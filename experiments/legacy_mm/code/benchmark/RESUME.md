# RSI 断点续跑（ACE / AHE / HyperAgents）

启动新实验需要**新的空 `output_dir`**，例如：

```bash
tools/run_gameworld.sh benchmark/<new-config>.json
```

额度恢复或进程中断后，配置文件（包括 `output_dir`）保持完全相同：

```bash
tools/run_gameworld.sh benchmark/<new-config>.json --resume
```

可以给三个框架各自独立的配置、输出目录并行运行；同一个输出目录有进程锁，不得并行运行两份。运行时形成 `checkpoints/initial.json` 及每轮 `checkpoints/train_NNNN.json`、`checkpoints/holdout_NNNN.json`。成功且已选择的轮次保存结果摘要及 ACE playbook / AHE 当前 workspace / HyperAgents 快照与演化计数；恢复时严格检查配置、task manifest、轮次顺序、结果摘要及持久状态，再跳过已提交轮次。`runs/` 下孤立的结果文件**不等于**已提交轮次。网络错误和候选验证错误停止当前实验，不推进下一轮。旧的未带这些 checkpoints 的实验输出不能使用 `--resume` 直接从下一轮继续。

未提交轮次的旧结果、部分进化目录及原有结果报告移至 `abandoned_attempts/`，`abandoned_attempts.jsonl` 保留尝试目录索引。GameWorld 的独立 attempt 目录仍位于原 `GAMEWORLD_RESULTS_ROOT` 下，由存档记录中的原始路径引用。当前轮从最后一次提交的模型状态重新执行；之前已经提交的轮次不重做。

**计费边界**：如果请求已到服务端、但进程在收到 usage 前中断，该请求是否计费、用掉多少 Token 无法可靠判定；当前轮重试可能再次计费。这些费用不能算作 0；出现重试存档后 `result.json` 的 `summary.full_token_coverage` 为 `false`，整场实验的 `metrics.evolution_cost.total_tokens` / `tokens_per_absolute_gain` 为 `null`。各已提交轮次记录中的 provider usage 仍可用于分析已知部分。自动续跑**不是** API 请求级的 exactly-once 事务。

`--resume` 不会从旧的无 checkpoint 的 2026-09-24 正式输出目录中推断进化状态。不得把修复前后的不同框架版本结果合并为同一次可比实验。代码/依赖升级后应新开实验输出目录。

`summary.attempted_*_runs` 包含最后一次失败尝试；`completed_*_runs` 只计成功，`committed_*_runs` 只计已有 checkpoint 的成功轮次。在额度不足时，不要把失败的那轮计为完成。
