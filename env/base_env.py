import jax.numpy as jnp
from jax_lob_colab.lob.JaxOrderBookArrays import apply_message, get_l2_state
import sys
sys.path.append("/content")



class OrderBookEnv:
    def __init__(self, price_levels=100, l2_depth=5, max_steps=1000):
        """
        初始化环境参数
        """
        self.price_levels = price_levels
        self.l2_depth = l2_depth
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        """
        重置环境状态
        """
        self.book = jnp.zeros((self.price_levels, 2))
        self.pos = 0
        self.current_step = 0
        obs = self.get_obs()
        return obs

    def step(self, action):
        """
        执行 agent 动作，更新环境状态
        action: jnp.array([type, side, price, size])
        """
        self.book = apply_message(self.book, action)

        # 更新持仓
        side = action[1]
        size = action[3]
        self.pos += jnp.where(side == 1, size, -size)

        # 计算奖励（可以调整逻辑）
        reward = -jnp.abs(self.pos) * 0.01

        # 更新步数
        self.current_step += 1
        done = self.current_step >= self.max_steps

        obs = self.get_obs()
        return obs, reward, done, {}

    def get_obs(self):
        """提取 L2 订单簿状态"""
        return get_l2_state(self.book, depth=self.l2_depth)