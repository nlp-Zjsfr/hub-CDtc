# Week07 · Credit / Market Risk Data Lineage 信息抽取

对照课件 `week7_信息抽取` 的 L1–L5 方法阶梯，做 **风险报告 / 模型文档里的 data lineage 抽取**，并在同一评测口径下对比 **macro F1**。

任务例子：文本写明「数据 A 依赖数据 B，因为 method M 的 output 是 A、input 是 B」→ 抽出结构化事件（类型 + 触发词 + 论元角色）。

## 数据从哪来

检索结论：**没有**公开的「credit/market risk 报告 → data lineage 结构化标注」NLP 数据集。

相近但不可直接当本任务金标的材料：

| 资源 | 为什么不够 |
|---|---|
| CRISIL「Automated Data Lineage for Risk Models」案例 | 工业交付，解析 SQL/SAS，无公开标注语料 |
| BCBS 239 / FRTB / Deloitte RDARR 白皮书 | 讲合规与 lineage 要求，不是 span 标注集 |
| DependEval / DI-BENCH / SEOSS | 软件仓库依赖 / 需求追溯，不是风控报表叙事 |
| GRAB / 10-K risk NLP | 风险主题分类，不是「A←method←B」血缘槽位 |

因此用 `data/build_dataset.py` **按真实风控话术自建合成语料**（IFRS9 ECL、IRB EAD/LGD/PD、FRTB SBM/IMCC、BCBS 239 KDE、COREP/FINREP 字段等）。

- 规模：train **500** / val **120**（5 类事件均匀）
- 格式：一句一文事件，字符级 span（对齐课件 DuEE 派生口径）
- 事件类型：`DataDepends` / `DataProduces` / `ReportMaps` / `SystemFlow` / `TransformDerives`

```bash
python data/build_dataset.py
```

## 方法阶梯（对照课件）

| 层级 | 方法 | 脚本 |
|---|---|---|
| L0（对照） | 规则：触发词典 + 训练集论元词表 | `rule/run_rule.py` |
| **L1** | 字符级 BiLSTM-CRF | `bilstm_crf/train.py` |
| **L2** | BERT-base-uncased + Linear BIO | `bert/train.py` |
| **L3** | Qwen2.5-0.5B zero-shot | `llm_zero_shot/extract_zero_shot.py` |
| **L4** | Qwen2.5-0.5B + LoRA SFT | `llm_lora/train_lora.py` + `eval_lora.py` |
| **L5** | 系统组合：BERT 主抽 + 规则补论元 + schema 校验 | `service/extract_service.py` |

统一评测：`common/evaluate.py`（trigger / argument / event_exact + **type_macro_f1** + **role_macro_f1**）。  
作业主对比的 macro F1 = **role_macro_f1**（按论元角色平均）。

## 一键复现

```bash
# 环境：建议用已装 torch/transformers/peft 的 conda（本机 miniconda）
# 模型：复用 week06 的 junction → pretrain_models/

python data/build_dataset.py
python rule/run_rule.py
python bilstm_crf/train.py --epochs 8
python bert/train.py --epochs 5
python llm_zero_shot/extract_zero_shot.py
python llm_lora/train_lora.py --epochs 3
python llm_lora/eval_lora.py
python service/extract_service.py
python eval/eval_all.py
```

结果表：`outputs/summary.md`。

## 本机实测（val n=120）

| 层级 | 方法 | trigger F1 | argument F1 | event F1 | type_macro_f1 | role_macro_f1 |
|---|---|---:|---:|---:|---:|---:|
| L0 | 规则词典+模板 | 1.0000 | 0.6658 | 0.0500 | 1.0000 | 0.6515 |
| **L1** | BiLSTM-CRF | 0.9167 | 0.7953 | 0.3833 | 0.9495 | **0.7745** |
| **L2** | BERT-Linear BIO | 1.0000 | 0.9782 | 0.9250 | 1.0000 | **0.9821** |
| **L3** | Qwen2.5-0.5B zero-shot | 0.0000 | 0.0259 | 0.0000 | 0.1343 | **0.0190** |
| **L4** | Qwen2.5-0.5B + LoRA | 1.0000 | 0.9790 | 0.9000 | 1.0000 | **0.9801** |
| **L5** | BERT + 规则组合 | 1.0000 | 0.9782 | 0.9250 | 1.0000 | **0.9821** |

按 **role_macro_f1**：L2 ≈ L5 ≈ L4 ≫ L1 ≫ L0 ≫ L3。

简要结论：

1. **未见实体**让纯词典规则（L0）论元召回掉到 ~0.51，macro F1 只有 0.65。
2. **预训练（L2）**相对无预训练 BiLSTM（L1）论元/角色 macro 提升约 +20pp。
3. **0.5B zero-shot（L3）**几乎跟不住 schema（错类型、错触发词、缺论元）；**同模型 LoRA（L4）**拉到与 BERT 同级。
4. **L5 组合**在本合成集上与 L2 持平——BERT 已抽全时，规则补洞收益有限；工程上仍保留校验/兜底链路。

## 目录

```
week07/
├── data/           build_dataset.py + schema + train/val.jsonl
├── common/         BIO / CRF / evaluate / model_paths
├── rule/           L0
├── bilstm_crf/     L1
├── bert/           L2
├── llm_zero_shot/  L3
├── llm_lora/       L4
├── service/        L5
├── eval/           汇总
└── outputs/        预测与指标
```
