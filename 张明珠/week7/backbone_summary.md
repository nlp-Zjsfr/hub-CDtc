# 换底座 A/B：RoBERTa-wwm-ext vs BERT-base-chinese

> 口径：`dev_extract.jsonl` 930 句，统一评测 `common/evaluate_events.py`；两组只改预训练权重，结构（12 层 / 768 维 / 149 标签）、解码器、max_len、epochs、lr、seed 全部相同。所有数字由本脚本从 `outputs/metrics/disc/*.json` 读取。

## 一、结果汇总

| 解码器 | 底座 | 触发词F1 | 论元F1 | 事件级F1 | 类型宏F1 | n200触发词 | n200论元 | 最优点 | 训练(s) | 溯源文件 |
|---|---|---|---|---|---|---|---|---|---|---|
| CRF 联合解码 | BERT-base-chinese | 0.8320 | 0.5815 | 0.1697 | 0.7741 | 0.8278 | 0.5672 | 3 | 2742 | `bert_crf_256.json` |
| CRF 联合解码 | RoBERTa-wwm-ext | 0.8105 | 0.5769 | 0.1676 | 0.7534 | 0.8082 | 0.5799 | 3 | 1751 | `roberta_crf_256.json` |
| Linear 逐位分类 | BERT-base-chinese | 0.8293 | 0.5630 | 0.1564 | 0.7654 | 0.8123 | 0.5598 | 3 | 637 | `bert_linear_256.json` |
| Linear 逐位分类 | RoBERTa-wwm-ext | 0.8177 | 0.5596 | 0.1692 | 0.7707 | 0.8244 | 0.5616 | 3 | 992 | `roberta_linear_256.json` |

## 二、差值（RoBERTa − BERT，百分点）

| 解码器 | Δ触发词F1 | Δ论元F1 | Δ事件级F1 | Δ类型宏F1 |
|---|---|---|---|---|
| CRF 联合解码 | -2.15 | -0.46 | -0.21 | -2.07 |
| Linear 逐位分类 | -1.16 | -0.34 | +1.28 | +0.53 |

## 三、配置核对（除底座外必须完全一致）

| 项 | BERT-base-chinese<br>+CRF | RoBERTa-wwm-ext<br>+CRF | BERT-base-chinese<br>+Linear | RoBERTa-wwm-ext<br>+Linear |
|---|---|---|---|---|
| epochs / 最优点 | 3 / 3 | 3 / 3 | 3 / 3 | 3 / 3 |
| max_len | 256 | 256 | 256 | 256 |
| lr | 2e-05 | 2e-05 | 2e-05 | 2e-05 |
| batch × 梯度累积 | 32 × 1 | 8 × 4 | 32 × 1 | 8 × 4 |
| seed | 42 | 42 | 42 | 42 |
| 可训练参数(M) | 102.4 | 102.4 | 102.4 | 102.4 |
| 内层验证 spanF1（选点用） | 0.6409 | 0.6373 | 0.6311 | 0.6170 |
| dev 合并 spanF1 | 0.6581 | 0.6586 | 0.6432 | 0.6401 |
| 训练(s) | 2742 | 1751 | 637 | 992 |
| 峰值显存(MB) | 1843 | 1650 | 1842 | 1649 |
| 推理延迟(ms/句) | 74.3 | 27.8 | 11.3 | 18.2 |

## 四、结论（由上面的数字自动生成）

- **CRF 联合解码**：四项指标全部为负（-2.15 ~ -0.21 个百分点，最差是触发词F1）→ 换底座没有一处受益
- **Linear 逐位分类**：有升有降 —— 最好的是事件级F1 +1.28，最差的是触发词F1 -1.16（百分点）
- 两个解码器的触发词 F1 都下降（-2.15 ~ -1.16 个百分点），方向一致 → 当前配置下 RoBERTa-wwm-ext **没有带来增益**。
- 但两项的 **dev 合并 span F1 基本打平**（BERT-CRF 0.6581 vs RoBERTa-CRF 0.6586，差 +0.05 个百分点）→ 整体抽取质量没有变差，差的是错误在各指标/各小类之间的分布。

> 读法提醒：batch 切分不同（BERT 32×1、RoBERTa 8×4，梯度累积在等长 micro-batch 下与 32×1 数学等价，但 dropout 随机流不同）；`训练(s)` / `推理延迟` 来自不同时间点的机器状态，**不可跨行比较**，只用于说明「换底座不增加成本」。

## 五、复现命令

```bash
python bert/train.py --decoder crf --max_len 256 --epochs 3 --run_name bert_crf_256
python bert/train.py --decoder crf --max_len 256 --epochs 3 --batch_size 8 --grad_accum 4 --seed 42
python bert/train.py --decoder linear --max_len 256 --epochs 3 --run_name bert_linear_256
python bert/train.py --decoder linear --max_len 256 --epochs 3 --batch_size 8 --grad_accum 4 --seed 42
```

> 注：BERT 那两行当年靠 `model_paths` 的默认值加载权重，而现在默认底座已改成 RoBERTa —— 复现 BERT 行时要显式加 `--bert_path <bert-base-chinese 的绝对路径>`，并保留 `--run_name bert_*`，否则会把 BERT 的权重结果写进 roberta 的文件名；RoBERTa 两行则以当前默认底座为准（默认已是 `chinese-roberta-wwm-ext`，换过默认值就补 `--bert_path pretrain_models/chinese-roberta-wwm-ext`）。

换底座三种方式（优先级从高到低）：`--bert_path <绝对路径>` → 环境变量 `EE_BERT_MODEL=<pretrain_models 下的目录名>` → 改 `common/model_paths.py` 的 `DEFAULT_BERT_MODEL_DIR`。改动只影响「加载哪份权重」，模型结构与标签体系不变。
