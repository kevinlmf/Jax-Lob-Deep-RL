import jax
import jax.numpy as jnp
import optax
import sys
sys.path.append("/content")
from flax import linen as nn
from jax_lob_colab.env.base_env import OrderBookEnv

# === 策略网络（输出 mean + log_std）===
class PolicyNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.tanh(x)
        x = nn.Dense(128)(x)
        x = nn.tanh(x)
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        return mean, log_std

# === Q 网络 ===
class QNetwork(nn.Module):
    @nn.compact
    def __call__(self, x, a):
        x = jnp.concatenate([x, a], axis=-1)
        x = nn.Dense(128)(x)
        x = nn.tanh(x)
        x = nn.Dense(128)(x)
        x = nn.tanh(x)
        return nn.Dense(1)(x).squeeze()

# === 超参数 ===
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2  # entropy 正则系数
LR = 3e-4
NUM_UPDATES = 100
ROLLOUT_LENGTH = 32
BATCH_SIZE = 32

# === 合法动作转换 ===
def sanitize_action(logits, env):
    type_ = jnp.clip(jnp.round(logits[0]), 1, 3).astype(jnp.int32)
    side = jnp.where(logits[1] >= 0, 1, -1).astype(jnp.int32)
    price = jnp.clip(jnp.round(logits[2]), 0, env.price_levels - 1).astype(jnp.int32)
    size = jnp.clip(jnp.round(jnp.abs(logits[3])), 1, 10).astype(jnp.int32)
    return jnp.array([type_, side, price, size])

# === Replay Buffer ===
class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.storage = []
        self.max_size = max_size

    def add(self, transition):
        if len(self.storage) >= self.max_size:
            self.storage.pop(0)
        self.storage.append(transition)

    def sample(self, batch_size):
        idx = jax.random.choice(jax.random.PRNGKey(int(time.time() * 1e6) % 2**32), len(self.storage), (batch_size,), replace=False)
        batch = [self.storage[i] for i in idx]
        obs, act, rew, next_obs, done = map(jnp.array, zip(*batch))
        return obs, act, rew, next_obs, done

# === SAC 主函数 ===
def sac_train():
    env = OrderBookEnv(price_levels=100, l2_depth=5)
    rng = jax.random.PRNGKey(0)

    obs_dim = env.get_obs().shape[0]
    action_dim = 4

    policy_net = PolicyNetwork(action_dim=action_dim)
    q_net1 = QNetwork()
    q_net2 = QNetwork()
    q_target1 = QNetwork()
    q_target2 = QNetwork()

    dummy_obs = jnp.ones((obs_dim,))
    dummy_act = jnp.ones((action_dim,))
    rng, k1, k2, k3, k4, k5 = jax.random.split(rng, 6)

    params_policy = policy_net.init(k1, dummy_obs)
    params_q1 = q_net1.init(k2, dummy_obs, dummy_act)
    params_q2 = q_net2.init(k3, dummy_obs, dummy_act)
    target_q1 = q_target1.init(k4, dummy_obs, dummy_act)
    target_q2 = q_target2.init(k5, dummy_obs, dummy_act)

    opt_q1 = optax.adam(LR)
    opt_q2 = optax.adam(LR)
    opt_policy = optax.adam(LR)
    state_q1 = opt_q1.init(params_q1)
    state_q2 = opt_q2.init(params_q2)
    state_policy = opt_policy.init(params_policy)

    buffer = ReplayBuffer()
    obs = env.reset()

    for update in range(NUM_UPDATES):
        for _ in range(ROLLOUT_LENGTH):
            mean, log_std = policy_net.apply(params_policy, obs)
            std = jnp.exp(log_std)
            action_sample = mean + std * jax.random.normal(rng, shape=mean.shape)
            action = sanitize_action(action_sample, env)

            next_obs, reward, done, _ = env.step(action)
            buffer.add((obs, action, reward, next_obs, float(done)))
            obs = next_obs if not done else env.reset()

        if len(buffer.storage) < BATCH_SIZE:
            continue

        obs_b, act_b, rew_b, next_obs_b, done_b = buffer.sample(BATCH_SIZE)

        def q_loss(params_q, target_params_q, obs_b, act_b, rew_b, next_obs_b, done_b):
            next_mean, next_log_std = policy_net.apply(params_policy, next_obs_b)
            next_std = jnp.exp(next_log_std)
            next_action = next_mean + next_std * jax.random.normal(rng, shape=next_mean.shape)
            next_action = jax.vmap(sanitize_action, in_axes=(0, None))(next_action, env)

            target_q1_val = q_target1.apply(target_params_q[0], next_obs_b, next_action)
            target_q2_val = q_target2.apply(target_params_q[1], next_obs_b, next_action)
            target_min_q = jnp.minimum(target_q1_val, target_q2_val)
            target_value = rew_b + GAMMA * (1 - done_b) * (target_min_q - ALPHA * jnp.sum(next_log_std, axis=-1))

            pred_q = q_net1.apply(params_q, obs_b, act_b)
            return ((pred_q - target_value) ** 2).mean()

        grads_q1 = jax.grad(q_loss)(params_q1, (target_q1, target_q2), obs_b, act_b, rew_b, next_obs_b, done_b)
        updates_q1, state_q1 = opt_q1.update(grads_q1, state_q1)
        params_q1 = optax.apply_updates(params_q1, updates_q1)

        grads_q2 = jax.grad(q_loss)(params_q2, (target_q1, target_q2), obs_b, act_b, rew_b, next_obs_b, done_b)
        updates_q2, state_q2 = opt_q2.update(grads_q2, state_q2)
        params_q2 = optax.apply_updates(params_q2, updates_q2)

        def policy_loss_fn(params, obs_b):
            mean, log_std = policy_net.apply(params, obs_b)
            std = jnp.exp(log_std)
            action = mean + std * jax.random.normal(rng, shape=mean.shape)
            action = jax.vmap(sanitize_action, in_axes=(0, None))(action, env)
            q_val = q_net1.apply(params_q1, obs_b, action)
            log_prob = jnp.sum(log_std, axis=-1)
            return (ALPHA * log_prob - q_val).mean()

        grads_policy = jax.grad(policy_loss_fn)(params_policy, obs_b)
        updates_policy, state_policy = opt_policy.update(grads_policy, state_policy)
        params_policy = optax.apply_updates(params_policy, updates_policy)

        # soft target update
        target_q1 = jax.tree_util.tree_map(lambda t, s: (1 - TAU) * t + TAU * s, target_q1, params_q1)
        target_q2 = jax.tree_util.tree_map(lambda t, s: (1 - TAU) * t + TAU * s, target_q2, params_q2)

        print(f"Update {update}: reward mean = {jnp.mean(rew_b):.4f}")

if __name__ == "__main__":
    import time
    sac_train()
