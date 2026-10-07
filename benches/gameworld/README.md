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

当前配置选择两个游戏，共用一份 GameWorld 核心：

| 类别 | 游戏 | 训练任务 | 留出任务 |
|---|---|---|---|
| Exploring | `18_minecraft-clone-glm` | 20 个训练任务，另保留 6 个 hard 任务资产 | 5 个原始留出任务 |
| Exploring | `03_astray` | 20 个 seed 参数化训练任务 | 5 个原始留出任务 |

训练和留出任务在配置中显式列出，不按数字 difficulty 随机生成，也不允许 train/holdout 重叠。

## 三种框架

公开代码同时支持三种方法，框架与模型是两个独立维度：

| CSD provider | GameWorld client | 输入保持 |
|---|---|---|
| `ace` | `ACEGameClient` | 原始 system prompt + 短 action history + 当前截图 + semantic controls |
| `hyperagents` | `HyperAgentsGameClient` | 同上 |
| `ahe` | `AHEGameClient` | 同上；若上游 AHE 尚未支持图片，会明确报兼容性错误 |

对应的 adaptor 使用本仓库的统一实现：

- `rsi_methods.ACE.ACEAdaptor.ACEAdaptor`
- `rsi_methods.HyperAgents.HyperAgentsAdaptor.HyperAgentsAdaptor`
- `rsi_methods.AHE.AHEAdaptor.AHEAdaptor`

代码不写入 API key。模型由配置中的 `agent.model` / `game_options.gameworld.model_spec` 指定，endpoint 和 key 由外部环境变量提供。

方法层默认配置使用 `azure/gpt-5.6-sol`，推理强度为 `high`。历史实验模板的模型与推理参数保持原样；其中 `game_options.gameworld.model_spec` 使用 catalog ID `azure-gpt-5.6-sol`。运行前按实际 API 准备本地配置，不要把 API key 写入 Git。

## 配置与运行

六份历史配置模板仍位于 `experiments/legacy_mm/configs/templates/`：

- `astray_{ace,ahe,hyperagents}.json`
- `minecraft_{ace,ahe,hyperagents}.json`

默认使用 `assets/core/`，任务 YAML 分别在 `assets/tasks/train/` 和 `assets/tasks/holdout/`。
显式 `gameworld_root` 优先于 `GAMEWORLD_ROOT`，环境变量优先于内置核心。
使用外部核心时可显式提供 `task_assets_roots.train` / `holdout`。
在仓库根目录运行，例如 PowerShell：

```powershell
$env:OPENAI_BASE_URL = "https://your-openai-compatible-endpoint/v1"
$env:OPENAI_API_KEY = "your-key"
python -m benchmark YOUR_LOCAL_CONFIG.json --max-new-train-runs 1
```

配置不写入 key 或 URL；方法 TOML 中的 `api_key_env` / `base_url_env` 指定读取的环境变量。
历史模板含 `${RSI_ROOT}`，主配置加载器不展开此占位符，不能直接当作主入口配置运行。
它们是完整实验模板（20 train / 5 holdout）；首次检查请在自己的本地配置中缩小任务数量与 `max_steps`。
GameWorld 运行产物保存在各自 `output_dir/gameworld/`，不依赖旧 launcher 初始化。
运行需要 Playwright 及 Chromium；HyperAgents 还需要可用的 Docker 环境。
本次仅完成代码接入，浏览器、容器、API 的实际连通性由下一轮 smoke test 确认。

切换框架时，同时修改 `agent.provider` 和 `game_options.gameworld.framework`；二者不一致会在配置解析阶段直接拒绝，避免出现“使用 AHE refinement、实际却跑 ACE client”的隐性错配。

## 输入、评分与结果

GameWorld 的任务 prompt、截图、短历史、semantic action space 和 evaluator 仍由 GameWorld 原始 catalog/runtime 生成。CSD adapter 不额外展开实时结构化 `game_state` 到 user prompt，也不改变 GameWorld 的 SR/PG 评分定义。

每局结果保留：

- `summary.json` 的最终 `progress`、`task_status`、`metrics` 和 stop reason；
- `interactions.jsonl` 的 prompt、截图路径、模型输出、执行动作、usage、游戏状态和逐步 evaluator 结果；
- CSD 标准 `FeedbackResult`，供 ACE/HyperAgents/AHE refinement 使用；
- CSD runner 的 `result.json`、每局 run record 和 token/time telemetry。

“游戏达到目标”和“执行过程是否发生异常”是两个不同概念：正常跑完但未通关会记录为 `status=ok, success=false`，只有启动失败、summary 缺失或运行异常才记录为 `status=error`，不会把错误伪装成 0 分游戏结果。

## 恢复与接入边界

任务级 `--resume` 和原有 `--target-dir` 的说明见 `benchmark/RESUME.md`。
GameWorld 步骤级恢复单独通过 `resumable_steps=true` 或 `RSI_STEP_RESUME=1` 启用，
目前支持 Astray / Minecraft，并保留浏览器状态、模型回复凭据和方法会话。
Astray 的默认 seed 42 与原始评估规则保留，训练 YAML 使用参数化 seed。
移动前 checkpoint 的绝对路径及代码 hash 不自动迁移；修改代码后请新开实验目录。

当前 GameWorld 仅接受 ACE / AHE / HyperAgents。HyperDSH 的图片转发和步骤恢复留待后续接入，
Hero / Sweepmine 的 HyperDSH 接入继续保留。指标与聚合实现沿用当前 CSD 版本。

待办：配置文件导入目录、相对路径基准与初始化方式留待讨论，本轮不迁移配置文件或改写加载器。
