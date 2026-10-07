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

## 安装依赖

在运行 benchmark 的 Python 环境中，从 `code` 目录执行：

```powershell
python -m pip install --upgrade -e ./benches/gameworld/assets/core/code --only-binary=greenlet,pillow,pyzmq,cryptography
python -m playwright install chromium
```

GameWorld 的依赖统一声明在 `assets/core/code/pyproject.toml`，其 `requirements.txt`
通过 `-e .` 引用该声明，需在文件所在目录使用。外层依赖仍由 `code/requirements.txt` 管理。
上述版本选择覆盖普通 CPython 3.12/3.13 的 Windows x64 环境；Playwright 1.49.1
使用 greenlet 3.1.1，Pillow 10.4.0 和 PyZMQ 26.2.0 提供 Python 3.13 预编译包。
升级 Playwright 后需重新安装对应的 Chromium。安装兼容性与 benchmark 运行结果仍需验证。

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

在本仓库的 RSI 接入中，执行与进化 agent 的 API 模型由各方法的 `config.toml` 控制。
benchmark JSON 控制任务清单、数量、seed、游戏步数与运行编排；其中的 `agent.model`
不覆盖这些 RSI 方法的模型设置。API key 和 endpoint 从方法 TOML 指定的环境变量读取。

`game_options.gameworld.model_spec` 是 **GameWorld catalog 配置条目的 ID**，
用于定位 `assets/core/code/catalog/models/<ID>.yaml`，提供提示模板、输出格式和默认客户端参数。
GameWorld wrapper 把实际模型调用交给 RSI agent，使用方法 TOML 中的执行模型；
refinement 同样使用方法 TOML 中对应的进化模型。

当前 `model_spec: "azure-gpt-5.6-sol"` 保留历史 catalog 条目以兼容现有客户端初始化。
这个 ID 的名称不表示实际 RSI 请求使用 GPT，也不要求它与方法 TOML 的模型名一致。
临时切换 DeepSeek 时应修改方法 TOML 的模型设置，保留可用的 GameWorld catalog ID。
未配置 `model_spec` 时，使用环境变量 `GAMEWORLD_MODEL_ID` 指定 catalog ID。

## 配置与运行

六份配置已放入 `configs/`，历史模板仍位于 `experiments/legacy_mm/configs/templates/`：

- `astray_{ace,ahe,hyperagents}.json`
- `minecraft_{ace,ahe,hyperagents}.json`

默认使用 `assets/core/`，任务 YAML 分别在 `assets/tasks/train/` 和 `assets/tasks/holdout/`。
历史配置的 `output_dir` 保留 `${RSI_ROOT}` 占位符，当前入口不展开环境变量，运行前需手动替换。
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
根据 2026-10-07 的手动运行日志，Astray + ACE 的最小训练和留出评测均为 `status=ok`；
AHE、HyperAgents 与 Minecraft 组合仍待验证，步骤级恢复也未在本轮 smoke 中验证。
视频回放需要系统 PATH 中的 FFmpeg；缺少时不影响已保存的评测结果。
不需要视频时，CMD 可执行 `set GAMEWORLD_DISABLE_VIDEO_REPLAY=1`，
PowerShell 可执行 `$env:GAMEWORLD_DISABLE_VIDEO_REPLAY = "1"`。

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
