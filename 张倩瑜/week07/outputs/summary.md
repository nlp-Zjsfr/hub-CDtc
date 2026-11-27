# Week07 Data Lineage IE — method comparison

评测集：`data/val.jsonl`。统一口径：`common/evaluate.py`。

主对比指标：**role_macro_f1**（按论元角色平均的 F1）与 **type_macro_f1**。

| 层级 | 方法 | trigger F1 | argument F1 | event F1 | type_macro_f1 | role_macro_f1 |
|---|---|---:|---:|---:|---:|---:|
| L0 | 规则词典+模板 | 1.0000 | 0.6658 | 0.0500 | 1.0000 | 0.6515 |
| L1 | BiLSTM-CRF | 0.9167 | 0.7953 | 0.3833 | 0.9495 | 0.7745 |
| L2 | BERT-Linear BIO | 1.0000 | 0.9782 | 0.9250 | 1.0000 | 0.9821 |
| L3 | Qwen zero-shot | 0.0000 | 0.0259 | 0.0000 | 0.1343 | 0.0190 |
| L4 | Qwen LoRA SFT | 1.0000 | 0.9790 | 0.9000 | 1.0000 | 0.9801 |
| L5 | BERT+规则组合 | 1.0000 | 0.9782 | 0.9250 | 1.0000 | 0.9821 |

## L1–L5 by role_macro_f1 (desc)

1. L2 BERT-Linear BIO: **0.9821**
2. L5 BERT+规则组合: **0.9821**
3. L4 Qwen LoRA SFT: **0.9801**
4. L1 BiLSTM-CRF: **0.7745**
5. L3 Qwen zero-shot: **0.0190**
