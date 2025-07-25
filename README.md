# 🧠 JAX LOB + Deep Reinforcement Learning (WIP)

This project builds upon [KangOxford/jax-lob](https://github.com/KangOxford/jax-lob) and explores how deep reinforcement learning (Deep RL) can be applied to high-frequency trading environments based on realistic limit order book (LOB) simulations.

The goal is to integrate a realistic market simulator with modular Deep RL agents such as PPO and SAC, and evaluate their performance in simulated trading tasks.

---

## 📂 Project Structure

```
├── AlphaTrade/       # Contains market data and simulation assets
├── env/              # Gym-compatible trading environments
├── lob/              # Limit Order Book logic and matching engine
├── rl/               # Deep RL agents (e.g. PPO, SAC)
```

---

## ✅ Highlights

- Built on a realistic LOB simulator using JAX (inspired by `jax-lob`)
- Modular environment and agent design
- Deep RL exploration in a market microstructure setting
- Early-stage experiment setup (WIP)

---

## 👨‍💻 Author

Mengfan Long ([GitHub](https://github.com/kevinlmf))

---

## 📜 License

MIT License – see the `LICENSE` file for details.
