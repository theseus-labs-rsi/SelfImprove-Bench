# ACE combat 25-task evaluation

`plan/evolve` 分别表示战斗阶段规划和 ACE 自进化阶段。得分栏使用 `objective_score(relative_score)`；失败任务的参考分数来自原始任务文件，实际分数记为 `N/A`。

| 测试编号 | 参考分数 | 得分（objective(relative)） | 用时（plan/evolve） | token 消耗（plan/evolve） |
|---|---:|---:|---:|---:|
| difficulty_1_task_01 | 425 | 377(0.89) | 65/45 | 21274/26225 |
| difficulty_1_task_02 | 425 | 351(0.83) | 33/122 | 21947/26715 |
| difficulty_1_task_03 | 425 | 228(0.54) | 116/65 | 22083/27640 |
| difficulty_1_task_04 | 428 | N/A | 64/49 | 22614/25460 |
| difficulty_1_task_05 | 424 | 378(0.89) | 84/34 | 22581/26616 |
| difficulty_2_task_01 | 443 | N/A | 29/52 | 8104/20675 |
| difficulty_2_task_02 | 438 | 196(0.45) | 240/62 | 41232/34955 |
| difficulty_2_task_03 | 513 | 220(0.43) | 17/97 | 8305/23502 |
| difficulty_2_task_04 | 467 | 177(0.38) | 220/69 | 41144/34667 |
| difficulty_2_task_05 | 520 | 301(0.58) | 131/33 | 42339/33861 |
| difficulty_3_task_01 | 423 | 268(0.63) | 151/32 | 59333/41095 |
| difficulty_3_task_02 | 418 | N/A | 79/88 | 24739/27112 |
| difficulty_3_task_03 | 428 | 148(0.35) | 99/62 | 41525/34923 |
| difficulty_3_task_04 | 429 | 148(0.34) | 91/60 | 41569/34828 |
| difficulty_3_task_05 | 438 | N/A | 89/76 | 25123/26442 |
| difficulty_4_task_01 | 521 | N/A | 53/88 | 25232/26737 |
| difficulty_4_task_02 | 479 | 125(0.26) | 206/51 | 59842/42353 |
| difficulty_4_task_03 | 429 | N/A | 27/29 | 24847/26854 |
| difficulty_4_task_04 | 448 | N/A | 99/70 | 41869/34038 |
| difficulty_4_task_05 | 480 | N/A | 195/30 | 60309/39398 |
| difficulty_5_task_01 | 343 | 168(0.49) | 45/30 | 26145/29782 |
| difficulty_5_task_02 | 490 | N/A | 655/81 | 26054/28277 |
| difficulty_5_task_03 | 560 | N/A | 337/30 | 81220/46762 |
| difficulty_5_task_04 | 530 | 215(0.41) | 67/62 | 25986/29708 |
| difficulty_5_task_05 | 483 | N/A | 27/82 | 25929/28153 |
