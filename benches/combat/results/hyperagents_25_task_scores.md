# HyperAgents combat 25-task evaluation

`plan/evolve` 分别表示战斗阶段规划和 HyperAgents 自进化阶段。得分栏使用 `objective_score(relative_score)`；失败任务的参考分数来自原始任务文件，实际得分记为 `N/A`。本批次未记录自进化调用的 token，因此 evolve token 记为 `N/A`。

| 测试编号 | 参考分数 | 得分（objective(relative)） | 用时（plan/evolve） | token 消耗（plan/evolve） |
|---|---:|---:|---:|---:|
| difficulty_1_task_01 | 425 | N/A | 59/204 | 20781/N/A |
| difficulty_1_task_02 | 425 | N/A | 23/57 | 7115/N/A |
| difficulty_1_task_03 | 425 | 415(0.98) | 53/89 | 20781/N/A |
| difficulty_1_task_04 | 428 | 220(0.51) | 12/101 | 7302/N/A |
| difficulty_1_task_05 | 424 | N/A | 60/145 | 22467/N/A |
| difficulty_2_task_01 | 443 | 280(0.63) | 125/174 | 37587/N/A |
| difficulty_2_task_02 | 438 | N/A | 87/109 | 22647/N/A |
| difficulty_2_task_03 | 513 | 307(0.60) | 218/131 | 39672/N/A |
| difficulty_2_task_04 | 467 | 185(0.40) | 61/191 | 23186/N/A |
| difficulty_2_task_05 | 520 | 302(0.58) | 169/192 | 40689/N/A |
| difficulty_3_task_01 | 423 | N/A | 64/185 | 24030/N/A |
| difficulty_3_task_02 | 418 | 160(0.38) | 166/347 | 40934/N/A |
| difficulty_3_task_03 | 428 | N/A | 412/279 | 23775/N/A |
| difficulty_3_task_04 | 429 | 333(0.78) | 261/168 | 31570/N/A |
| difficulty_3_task_05 | 438 | 233(0.53) | 409/78 | 60017/N/A |
| difficulty_4_task_01 | 521 | N/A | 473/163 | 70122/N/A |
| difficulty_4_task_02 | 479 | 320(0.67) | 518/99 | 62141/N/A |
| difficulty_4_task_03 | 429 | N/A | 742/132 | 69260/N/A |
| difficulty_4_task_04 | 448 | N/A | 1053/229 | 68649/N/A |
| difficulty_4_task_05 | 480 | 280(0.58) | 1116/224 | 79927/N/A |
| difficulty_5_task_01 | 343 | N/A | 1080/192 | 70851/N/A |
| difficulty_5_task_02 | 490 | 167(0.34) | 359/472 | 67662/N/A |
| difficulty_5_task_03 | 560 | 208(0.37) | 707/253 | 69642/N/A |
| difficulty_5_task_04 | 530 | N/A | 639/104 | 46598/N/A |
| difficulty_5_task_05 | 483 | N/A | 505/381 | 60204/N/A |
