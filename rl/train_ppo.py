import jax
import jax.numpy as jnp
import optax
import sys
sys.path.append("/content")
from flax import linen as nn
from jax_lob_colab.env.base_env import OrderBookEnv


# === 策略网络 ===
class PolicyNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.tanh(x)
        x = nn.Dense(128)(x)
        x = nn.tanh(x)
        logits = nn.Dense(self.action_dim)(x)
        return logits

# === 值函数网络 ===
class ValueNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.tanh(x)
        x = nn.Dense(128)(x)
        x = nn.tanh(x)
        value = nn.Dense(1)(x)
        return value.squeeze()

# === PPO 超参数 ===
GAMMA = 0.99
LR = 3e-4
NUM_UPDATES = 100
ROLLOUT_LENGTH = 32

# === 确保动作合法 ===
def sanitize_action(logits, env):
    """
    将策略输出 logits 转换为合法动作
    """
    type_ = jnp.clip(jnp.round(logits[0]), 1, 3).astype(jnp.int32)
    side = jnp.where(logits[1] >= 0, 1, -1).astype(jnp.int32)
    price = jnp.clip(jnp.round(logits[2]), 0, env.price_levels - 1).astype(jnp.int32)
    size = jnp.clip(jnp.round(jnp.abs(logits[3])), 1, 10).astype(jnp.int32)
    return jnp.array([type_, side, price, size])

# === PPO 训练 ===
def ppo_train():
    env = OrderBookEnv(price_levels=100, l2_depth=5)
    rng = jax.random.PRNGKey(0)

    obs_dim = env.get_obs().shape[0]
    action_dim = 4  # type, side, price, size

    # 初始化策略和值函数
    policy_net = PolicyNetwork(action_dim=action_dim)
    value_net = ValueNetwork()

    rng, init_rng1, init_rng2 = jax.random.split(rng, 3)
    params_policy = policy_net.init(init_rng1, jnp.ones((obs_dim,)))
    params_value = value_net.init(init_rng2, jnp.ones((obs_dim,)))

    optimizer_policy = optax.adam(LR)
    optimizer_value = optax.adam(LR)
    opt_state_policy = optimizer_policy.init(params_policy)
    opt_state_value = optimizer_value.init(params_value)

    obs = env.reset()

    for update in range(NUM_UPDATES):
        obs_buffer, act_buffer, rew_buffer, val_buffer = [], [], [], []
        for _ in range(ROLLOUT_LENGTH):
            logits = policy_net.apply(params_policy, obs)

            # ✅ 确保动作合法
            action = sanitize_action(logits, env)

            next_obs, reward, done, _ = env.step(action)

            obs_buffer.append(obs)
            act_buffer.append(action)
            rew_buffer.append(reward)
            val_buffer.append(value_net.apply(params_value, obs))

            obs = next_obs
            if done:
                obs = env.reset()

        # === Advantage 和目标值计算 ===
        vals = jnp.array(val_buffer)
        rewards = jnp.array(rew_buffer)
        next_vals = jnp.append(vals[1:], value_net.apply(params_value, obs))
        advantages = rewards + GAMMA * next_vals - vals

        # === 更新值函数 ===
        def loss_value(params, obs_batch, target_batch):
            pred_vals = jax.vmap(lambda o: value_net.apply(params, o))(obs_batch)
            return ((pred_vals - target_batch) ** 2).mean()

        grads_value = jax.grad(loss_value)(
            params_value, jnp.array(obs_buffer), rewards + GAMMA * next_vals
        )
        updates_value, opt_state_value = optimizer_value.update(grads_value, opt_state_value)
        params_value = optax.apply_updates(params_value, updates_value)

        # === 更新策略 ===
        def loss_policy(params, obs_batch, act_batch, adv_batch):
            def single_loss(obs, act, adv):
                logits = policy_net.apply(params, obs)
                log_prob = -((logits - act.astype(jnp.float32)) ** 2).sum()
                return -(log_prob * adv)
            losses = jax.vmap(single_loss)(obs_batch, act_batch, adv_batch)
            return losses.mean()

        grads_policy = jax.grad(loss_policy)(
            params_policy, jnp.array(obs_buffer), jnp.array(act_buffer), advantages
        )
        updates_policy, opt_state_policy = optimizer_policy.update(grads_policy, opt_state_policy)
        params_policy = optax.apply_updates(params_policy, updates_policy)

        print(f"Update {update}: Mean reward={jnp.mean(rewards):.4f}")

if __name__ == "__main__":
    ppo_train()