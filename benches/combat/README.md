# Combat game module

本目录只维护 Combat 的任务生成、参考 GT、逐关模拟器和统一适配器 `game.py`。跨游戏编排、agent 生命周期与 RSI 指标请参阅项目根目录 [README](../README.md)，并通过 `python -m benchmark` 运行。

`task/` 中 seed=100 的 25 个任务是固定留出池；`batch_generate_tasks.py` 用于重新生成该池。
