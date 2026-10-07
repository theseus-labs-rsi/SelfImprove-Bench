# Minecraft hard-6 v1（本地扩展任务草案）

日期：2026-09-26。**不是 GameWorld 官方任务，也不是已有 14 轮结果的一部分；未进行付费游戏测试。**
将它们作为 14 轮训练完成后冻结的 **新增留出集**，不能先用于演化，再作为未见任务汇报。
原有 `18_01`～`18_25` 及 ACE / HyperAgents / AHE 的历史成绩均不修改。

## 六题与区分假设

| 顺序 | ID | 成功条件（100 动作内） | 预期考查点 |
|---:|:---|:---|:---|
| 1 | `18_h01` | 本局新增木头 ≥12 | 多棵树间搜索、保持目标、长程行动 |
| 2 | `18_h02` | 本局新增石头 ≥12 | 找到石面、连续定位（草/泥/圆石不计） |
| 3 | `18_h03` | 本局新增树叶 ≥12 | 视角与高度调整、跨树冠采集 |
| 4 | `18_h04` | **同一时刻库存**中不同物品种类 ≥3 | 由数量任务转为资源多样化（不限特定材料） |
| 5 | `18_h05` | 本局新增煤矿 ≥2 | 找矿、石/矿辨认；稀有资源定位 |
| 6 | `18_h06` | 本局新增铁矿 ≥2 | 地下探索、找矿与采矿 |

上表仅为**待检验的区分假说**，不是已经证实的难度等级。特意不写“采到木+石+铁各 N 个”，因为当前上游 evaluator 的 `aggregate_score_fields` 只求和，不检查每项下界；那种措辞会造成作弊捷径/假成功。`18_h04` 使用上游 `metrics.items_unique`，任何库存物品类型都算；“必须采集指定三种材料”**并未**被评分器强制保证。游戏规则说明不能制作工具，任务也不依赖合成或新的动作空间。

## 实现和验证范围

- 六份完整 YAML 在本目录；沿用 GameWorld 的 `game_api_metric`、`score_field`、`task_target_score_field`，只改任务定义，不改上游游戏引擎/评分代码、不改三个框架。
- 分数为 `min(1, score_best / target)` 的相对进度；达到目标通常提前终止。因此报告时并列 **raw_score / target、相对进度、达标率、用步数、完整 Token 和时长**。未达标的部分进度同样用于比较，不只看 0/1。库存增长计数遵循 GameWorld 已有 `item_gains` 定义，避免把起始库存当进展。
- 已用 `python self_evlove_game/tools/validate_gameworld_custom_tasks.py` 完成 6/6 离线校验：YAML 可解析、ID 不碰现有文件、零分/部分进度/成功阈值下 evaluator 状态合理。该测试**不验证**地图生成、矿脉可达性或三框架实际区分度。
- 上游任务 loader 将 YAML 固定从 `GAMEWORLD_ROOT/code/catalog/tasks/<game_id>/<task_id>.yaml` 读取；仅保存本目录文件**不会自动加载**。运行时必须将六份 YAML 安装到**独立的 GameWorld 可写副本或隔离的 catalog overlay**，并指向该副本。不要覆盖原始 GameWorld、不要向原有 14 轮结果目录写入、不要绕过旧实验的版本检查。安装方式和 GameWorld 文件指纹写入新 lineage 的 manifest/交接记录。**此步骤尚未执行。**
- 本地代码核对对象：`GameWorld/code/env/task_evaluator.py`（SHA256 `36f164c959334ae1eb57bd03307f3b9db83973657335f155f7a5f951498e2242`）以及 `code/games/benchmark/18_minecraft-clone-glm/game_api.js`（SHA256 `6e1254211b86f418d155bbb77b5529d14e3fde261929b075272dace603d2aaa1`）；以后上游换版本要重新校验字段和结果，不要推定通用。

## 若获准做付费校准 / 正式比较

1. **先冻结文件 hash、任务顺序、同一游戏种子序列、100 步与模型设置**，为三框架分别建立新 output_dir / holdout manifest，并核对 6 个任务、gameworld seed、prompt 与 scorer 配置一致（框架字段允许不同）。读取并校验三个框架已提交的第 14 轮检查点，避免重训或把新题写进旧 training；留出运行应禁止拿本次任务反馈继续演化。**但不能直接修改旧配置后用 `--resume`：当前 runner 严格核对 config/task_manifest 的 digest，必定拒绝。** 正式运行前需新增只读、可审计的“从旧检查点派生新评测 lineage”能力，或同等级的独立评测入口；保留方法状态、原始检查点和原目录不变。该能力目前未实现，不能宣称本批任务可以立即无损续跑。
2. 先做**隔离校准**：例如在开发用种子上用同一项任务跑三个框架各 1 局，确认 scorer 有非零/非饱和中间分数、无 `unknown/error`、三种环境能启动、Token/截图记录完整。此结果标注 development，不能回填正式留出；如果六题全过或全零，先修订 `v2`，正式评测只用新的未见种子，保留并报告开发历史。
3. 正式评测之前估算并经用户确认付费 Token 预算、单局 guard/失败费用核对、种子数（推荐 ≥3 个独立种子以观察方差），再对三个框架共享相同的 task × seed 矩阵。禁止在看到一个框架的分数后只为某个框架调阈值或挑选种子。
4. 报告逐题配对分数、达标率、步骤、实际完整账本以及不确定性；成功局和失败尝试单列。旧 14 轮属于**训练**，这六题属于**新增域内留出**；不等于跨游戏泛化或已证明进化增益。14 轮前不同框架演化节奏不同，跨框架成绩不能直接解释为“演化算法单一因素”差异。

> 注意：原 AHE 14 个成功训练轮完整账本为 11,654,512 Token（不包含一次用量不全的失败尝试）；新六题的费用和难度未知。创建 YAML/离线检查均不会调用付费模型；目前没有启动任何付费评测。

## h01/h02 隔离校准（2026-09-26，运行中；结果以 checkpoint 为准）

已新增 `tools/run_gameworld_hard2.py`、`tools/run_gameworld_hard2.sh`、`tools/supervise_gameworld_hard2.py`。
三框架第 14 轮训练检查点派生的**隔离副本**在
`outputs/gameworld_hard2_calibration_20260926/{ace,hyperagents,ahe}`，新任务的 GameWorld
catalog overlay 在同级 `gameworld-overlay/`；原始 GameWorld 源码和原始训练输出不写入。ACE 的方法配置要求 workspace 放在项目外的 `../outputs/ace/hard2_gameworld_hard2_calibration_20260926_ace/`，这里也仅是旧 workspace 的全量副本，不能写回旧 workspace。
各目录 `evaluation_manifest.json` 冻结原 checkpoint/config/manifest 的 SHA256、两题 YAML hash、
任务种子、方法配置和最大步数，`checkpoints/initial.json` 是只读派生状态。

三个独立 tmux 会话名 `gw_hard2_{ace,hyperagents,ahe}_0926`；每条线按 h01→h02 顺序，
每局提交 `checkpoints/holdout_000{1,2}.json` 和 `runs/holdout/*.json`，原始游戏交互/截图在
`results/`。监控状态 `logs/<method>.guard.json`、执行输出 `logs/<method>.log`。
AHE 每局验证总 Token ≤3,000,000，ACE/HyperAgents 每局 ≤1,500,000，动作增量 ≤120,000；
这些均仅是**已观测**用量的停止线，不能保证美元封顶。失败/缺失 summary 或 Token 时
`inflight.json` 保留，**不自动重试、不会进入该框架下一题**。手工续跑前需核查失败账单及
孤儿进程，并归档失败尝试；不得把失败费用计为零。正常中止、且仅有已提交 checkpoint
时，可用 `tools/run_gameworld_hard2.sh run --root outputs/gameworld_hard2_calibration_20260926 --method <method>`
继续未提交题目（外部守卫仍要启用，禁止裸跑）。

**本批仅为 development 校准**，不是预注册的独立多种子正式留出评测。
当前 `benches/gameworld/benchmark.py` 把 task seed 记入记录，却没有在
`gameworld_main(...)` 中显式传入游戏随机种子；即使三框架 manifest 的数值 seed 相同，
也不能据此声称底层随机地图严格相同。看完此次结果后不能把调过阈值的同一批题目
又当作“未见”正式测试。

## 2026-09-26 本轮调整：AHE 一题，ACE/HyperAgents 六题

用户在 h01 三线运行中调整范围：AHE **仅运行 h01**，不自动进入 h02；ACE 和
HyperAgents 继续 h01、h02，然后顺序运行 h03～h06。AHE 已启动的两题 runner 在内存中
持有旧 manifest，**不能通过改 YAML 或编辑磁盘 manifest 让它自然停在 h01**。独立 tmux
`gw_hard2_ahe_h01_stop_0926` 运行 `tools/stop_gameworld_ahe_after_h01.py`，每 25ms
检查 `ahe/checkpoints/holdout_0001.json`，待 h01 原子提交即冻结/终止 AHE runner 进程组，
保留 `inflight.json` 供核查；如已进入 h02 或停在提交瞬间，**禁止将未知费用当零**。
停机审计记录 `outputs/gameworld_hard2_calibration_20260926/logs/ahe.stop_after_h01.json`
和原 `logs/ahe.guard.json`。守护进程可能将这次人工停止标记为 incomplete，这不等于
h01 checkpoint 无效；应分别核验 checkpoint、h02 是否发生调用、GameWorld trace 和账本。
无需且不得在这个 lineage 重启 AHE 的双题 runner。

ACE/HyperAgents 后四题在独立 lineage `outputs/gameworld_hard4_calibration_20260926/`，
各自从冻结第 14 轮 checkpoint 派生，与前两题结果不相互训练、也不写回旧训练目录。
`tools/run_gameworld_hard2.py prepare --ids 18_h03,18_h04,18_h05,18_h06 --methods ace,hyperagents`
已创建独立 overlay，`check --method ...` 两者离线恢复均通过。三、四、五题沿用
原 manifest 第 3～5 个种子，原 manifest 只有五个 holdout seed；第六题使用确定性
SHA-256 派生种子 `386999964629840419`，两框架一致。注意 GameWorld 游戏主函数
未显式接收该种子，不可据此声称地图配对相同。

两个 tmux `gw_hard4_{ace,hyperagents}_0926` 各运行
`tools/chain_gameworld_hard4.py`：**仅当**同方法前两题 guard 状态 finished、两份
checkpoint 提交且无 inflight 时，自动 exec 有 Token 监督的
`tools/supervise_gameworld_hard2.py --expected-tasks 4`；如果前批失败则不发起后四题。
后四题每局结果记录于新 lineage 的 `runs/holdout/`、`checkpoints/holdout_000*.json`、
`results/`，监控在 `logs/<method>.guard.json`，队列在 `logs/<method>.chain.log`。
单步已观测 120k Token、每局末核验 1.5M Token、四题累计已观测 6M Token；
在途请求仍可能产生未观测费用，这些不是美元硬封顶。失败/Token 不全继续 fail-stop、
不自动重试。以上新题均属于开发校准，不可作为未见的正式多种子留出结果。
