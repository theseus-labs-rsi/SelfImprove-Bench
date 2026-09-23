# AHE + GameWorld 对齐说明

## 当前范围

本实验新增 RSI-Bench `CSD` 分支中的 `AHE`（Agentic Harness Engineering）方法，
与已有 ACE/HyperAgents 实验隔离。CSD checkout 使用独立 worktree：

```text
/mnt/shared-storage-user/yangjie/RSI-Bench-CSD
commit: 2540eebae49ce75e4ebc74009105db4cd3087997
```

现阶段选取两个 `explore` 游戏：

| 类别 | 游戏 | 编排任务 | 留出任务 |
|---|---|---|---|
| Exploring | `18_minecraft-clone-glm` | `18_02`, `18_03` | `18_01`, `18_05` |
| Exploring | `06_captaincallisto` | `06_01`, `06_03` | `06_02`, `06_04` |

任务划分保存在
`experiments/configs/explore_ahe_csd_minecraft_captaincallisto.json`，不与已有
Simulation 划分混用。

## 输入口径

GameWorld 的有效输入仍然是原始游戏 system prompt、短 action history 和当前截图；
动作仍通过 GameWorld 注册的 semantic controls 执行；评测器和 `max_steps=100` 不变。
`experiments/ahe_game_client.py` 只负责把这一输入/输出契约接到 CSD `AHEAgent`。

## 当前阻塞

CSD 的 `AHEAgent.act()` 当前明确拒绝 `image_paths`，而 GameWorld 视觉游戏不能在没有截图的
情况下进行有效复现。因此在 AHE 的 NexAU 图像输入映射完成前，不能生成有效的 AHE 游戏分数；
不能用 text-only 调用替代，也不能把启动失败写成 0 分。

另外，CSD commit 中的
`rsi_methods/AHE/agentic-harness-engineering` 是 gitlink，但当前 worktree 没有 materialize
该上游目录。runner 会在启动前检查这一点并给出明确错误。

## 当前入口状态

本轮已落地 CSD bridge、GameWorld action client 和两游戏任务配置，但尚未开放正式
`run_ahe_evolution.py`：CSD AHE 上游 gitlink 尚未 materialize，且 NexAU 图像消息格式
尚未验证。正式运行前必须补齐这两项；否则只能做 import/dry-run，不能声称已经完成
GameWorld AHE 评测。
