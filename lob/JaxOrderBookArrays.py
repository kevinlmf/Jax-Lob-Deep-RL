import jax
import jax.numpy as jnp
import sys
sys.path.append("/content")


# === 初始化订单簿 ===
def build_order_book_array(price_levels):
    return jnp.zeros((price_levels, 2))

# === 消息处理 ===
@jax.jit
def apply_message(book_array, message):
    type_, side, price, size = message
    side_idx = jnp.where(side == 1, 0, 1)

    # === 限价单逻辑 ===
    def limit_order(book_array):
        return book_array.at[price, side_idx].add(size)

    # === 撤单逻辑 ===
    def cancel_order(book_array):
        current = book_array[price, side_idx]
        new_size = jnp.maximum(current - size, 0)
        return book_array.at[price, side_idx].set(new_size)

    # === 市价单逻辑（修复类型不一致问题）===
    def market_order(book_array):
        opp_idx = 1 - side_idx
        remaining = jnp.asarray(size, dtype=jnp.float32)  # ✅ 强制转换类型

        def body_fn(i, state):
            book_array, remaining = state
            level_qty = book_array[i, opp_idx]
            traded_qty = jnp.minimum(level_qty, remaining)
            book_array = book_array.at[i, opp_idx].add(-traded_qty)
            remaining -= traded_qty
            remaining = jnp.asarray(remaining, dtype=jnp.float32)  # ✅ 保持类型
            return (book_array, remaining)

        price_range = jax.lax.cond(
            side == 1,
            lambda _: jnp.arange(book_array.shape[0]),          # 买单扫卖盘
            lambda _: jnp.arange(book_array.shape[0])[::-1],    # 卖单扫买盘
            operand=None
        )

        book_array, _ = jax.lax.fori_loop(
            0, price_range.shape[0],
            lambda idx, state: body_fn(price_range[idx], state),
            (book_array, remaining)
        )

        return book_array

    # === 根据 type 选择逻辑 ===
    book_array = jax.lax.switch(
        type_ - 1,
        [limit_order, cancel_order, market_order],
        book_array
    )
    return book_array

# === 提取 L2 订单簿状态 ===
def get_l2_state(book_array, depth=10):
    prices = jnp.arange(book_array.shape[0])

    ask_sizes = book_array[:, 1]
    sorted_ask_idx = jnp.argsort(prices)
    top_ask_prices = prices[sorted_ask_idx][:depth]
    top_ask_sizes = ask_sizes[sorted_ask_idx][:depth]

    bid_sizes = book_array[:, 0]
    sorted_bid_idx = jnp.argsort(-prices)
    top_bid_prices = prices[sorted_bid_idx][:depth]
    top_bid_sizes = bid_sizes[sorted_bid_idx][:depth]

    l2_state = jnp.concatenate([
        top_ask_prices, top_ask_sizes,
        top_bid_prices, top_bid_sizes
    ])
    return l2_state