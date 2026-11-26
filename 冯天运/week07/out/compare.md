# week7 抽取实验 · 结果对比

> 全部方案用同一套评测口径；entity F1 是跨数据集可比的主指标。


## 数据集：cmeee

| 方案 | n | entity F1 | P | R | 幻觉率 | JSON合法 | token | 耗时(s) |
|---|---|---|---|---|---|---|---|---|
| local-bert/bert-base-chinese·finetune-10ep | 16 | 0.9375 | 0.9375 | 0.9375 | 0.0000 | 16/16 | 0 | 31.7 |
| local-bert/chinese-roberta-wwm-ext·finetune-10ep | 16 | 0.8197 | 0.8621 | 0.7812 | 0.0000 | 16/16 | 0 | 45.7 |
| local-bert/bert-base-chinese·finetune-5ep | 16 | 0.3333 | 0.3235 | 0.3438 | 0.0000 | 16/16 | 0 | 22.5 |


最佳：**local-bert/bert-base-chinese·finetune-10ep**（entity F1 0.9375），比第二名高 0.1178。


## 数据集：duee_fin

| 方案 | n | entity F1 | P | R | trigger F1 | event F1 | 幻觉率 | JSON合法 | token | 耗时(s) |
|---|---|---|---|---|---|---|---|---|---|---|
| local-bert/bert-base-chinese·finetune-2ep | 300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 300/300 | 0 | 603.9 |
| local/Qwen3-0.6B·zero | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4/5 | 0 | 815.4 |


最佳：**local-bert/bert-base-chinese·finetune-2ep**（entity F1 0.0000），比第二名高 0.0000。
