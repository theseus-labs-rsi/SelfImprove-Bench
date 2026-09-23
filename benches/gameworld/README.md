# RSI-Bench CSD + GameWorld 对齐说明

## 目标

本目录把 GameWorld 作为 RSI-Bench CSD 的一个原生 benchmark 接入。整体执行链路与 CSD 自带的 `farm`、`combat`、`heros_journey`、`sweepmine` 保持一致：

```text
benchmark runner
  -> benchmark.tasks / task_manifest
  -> benchmark.game_adapters
  -> benches.gameworld.GameWorldBenchmark
  -> GameWorld main.py
  -> GameWorld BaseClient wrapper
  -> ACE / HyperAgents / AHE CSD agent
  -> GameWorld evaluator
```

因此 GameWorld 不是在 CSD 外部另起一套实验入口，而是和其他 CSD benchmark 共用任务编排、训练/refinement、候选验证、留出评测、结果汇总和错误语义。

## 当前游戏范围

当前示例选择两个 `explore` 游戏：

| 类别 | 游戏 | 训练任务 | 留出任务 |
|---|---|---|---|
| Exploring | `18_minecraft-clone-glm` | `18_02`, `18_03` | `18_01`, `18_05` |
| Exploring | `06_captaincallisto` | `06_01`, `06_03` | `06_02`, `06_04` |

训练和留出任务在配置中显式列出，不按数字 difficulty 随机生成，也不允许 train/holdout 重叠。

## 三种框架

公开代码同时支持三种方法，框架与模型是两个独立维度：

| CSD provider | GameWorld client | 输入保持 |
|---|---|---|
| `ace` | `ACEGameClient` | 原始 system prompt + 短 action history + 当前截图 + semantic controls |
| `hyperagents` | `HyperAgentsGameClient` | 同上 |
| `ahe` | `AHEGameClient` | 同上；若上游 AHE 尚未支持图片，会明确报兼容性错误 |

对应的 CSD adaptor 仍由固定 RSI-Bench checkout 提供：

- `rsi_methods.ACE.ACEAdaptor.ACEAdaptor`
- `rsi_methods.HyperAgents.HyperAgentsAdaptor.HyperAgentsAdaptor`
- `rsi_methods.AHE.AHEAdaptor.AHEAdaptor`

代码不写入 API key。模型由配置中的 `agent.model` / `game_options.gameworld.model_spec` 指定，endpoint 和 key 由外部环境变量提供。

## 配置与运行

三个可直接复制的配置模板位于：

- `benchmark/gameworld_explore_ace.example.json`
- `benchmark/gameworld_explore_hyperagents.example.json`
- `benchmark/gameworld_explore_ahe.example.json`

配置中的 `gameworld_root` 可以留空，此时从 `GAMEWORLD_ROOT` 读取 GameWorld checkout。典型运行方式：

```bash
export GAMEWORLD_ROOT=/path/to/GameWorld
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
export OPENAI_API_KEY=your-key

PYTHONPATH=.:/path/to/RSI-Bench-CSD python -m benchmark benchmark/gameworld_explore_ace.example.json
```

切换框架时，同时修改 `agent.provider` 和 `game_options.gameworld.framework`；二者不一致会在配置解析阶段直接拒绝，避免出现“使用 AHE refinement、实际却跑 ACE client”的隐性错配。

## 输入、评分与结果

GameWorld 的任务 prompt、截图、短历史、semantic action space 和 evaluator 仍由 GameWorld 原始 catalog/runtime 生成。CSD adapter 不额外展开实时结构化 `game_state` 到 user prompt，也不改变 GameWorld 的 SR/PG 评分定义。

每局结果保留：

- `summary.json` 的最终 `progress`、`task_status`、`metrics` 和 stop reason；
- `interactions.jsonl` 的 prompt、截图路径、模型输出、执行动作、usage、游戏状态和逐步 evaluator 结果；
- CSD 标准 `FeedbackResult`，供 ACE/HyperAgents/AHE refinement 使用；
- CSD runner 的 `result.json`、每局 run record 和 token/time telemetry。

“游戏达到目标”和“执行过程是否发生异常”是两个不同概念：正常跑完但未通关会记录为 `status=ok, success=false`，只有启动失败、summary 缺失或运行异常才记录为 `status=error`，不会把错误伪装成 0 分游戏结果。

## 固定版本

CSD adaptor 使用独立固定 checkout，默认路径：

```text
/mnt/shared-storage-user/yangjie/RSI-Bench-CSD
```

严格运行前会校验固定 commit 和工作区洁净状态；旧 GameWorld 实验目录与旧实现不会被覆盖。
