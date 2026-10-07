# RSI 恢复入口（ACE / AHE / HyperAgents）

从仓库根目录运行；新实验使用空 `output_dir`：

```bash
python -m benchmark CONFIG --max-new-train-runs 1
python -m benchmark CONFIG --resume
```

`--max-new-train-runs N` 在 N 个新训练任务提交 checkpoint 后暂停。
`--resume` 继续相同输出目录，要求配置和任务 manifest 不变，校验连续轮次、
结果 hash 和方法状态后跳过已提交任务。孤立 `runs/` 文件不代表状态已提交。
同一目录由进程锁保护，不能同时运行两个实例。

原有新目录恢复仍保留：

```bash
python -m benchmark CONFIG --target-dir NEW_OUTPUT_DIR
```

`CONFIG.output_dir` 是原目录，`--target-dir` 是新的工作目录，保留已有 adaptor
的恢复规则。它不能与 `--resume` 或 `--max-new-train-runs` 同时使用。
旧实验没有 `checkpoints/initial.json` 时不能使用新的 `--resume`。
HyperDSH 暂不支持这两种恢复入口，显式传入会拒绝执行。

普通错误保留 `continue_on_error` 语义；AHE / HyperDSH 致命错误停止实验。
在未成功提交的任务之后继续运行，会停止追加 checkpoint 链，保留普通结果，
并在 summary 中标记 `checkpoint_chain_complete=false`。原目录恢复拒绝跳过
这类未提交任务后面的已执行记录，避免把失败之后的进化状态错误地当成已提交状态。

未提交任务重新执行前，旧结果和部分进化证据移至 `abandoned_attempts/`。
`attempt.json` 标记费用可能未知；运行记录中的已知 usage 不包含所有中断费用。
当前指标算法沿用 CSD，遇到这类存档需另行核对开销，不能将记录之和视为真实总费用。

GameWorld 步骤级恢复另由 `resumable_steps=true` 或 `RSI_STEP_RESUME=1` 启用。
启用后保留 `output_dir/gameworld/` 中的浏览器 checkpoint、已收到的模型回复和会话，
避免重新请求已有凭据的步骤。如果训练后的 refinement 已经开始但没有提交，
会拒绝自动重试该进化阶段。已发出而未取得 usage 的请求仍可能产生额外费用。

`--resume` 是任务状态恢复，GameWorld checkpoint 是局内步骤恢复，两者不能互相替代。
代码、依赖、资产路径改变后请新开实验目录；旧 checkpoint 不自动迁移。
