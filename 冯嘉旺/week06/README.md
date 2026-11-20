# Week06：中文短文本分类（TNEWS 15 类）

把 CLUE TNEWS 的 15 类中文新闻标题分类任务，用 7 条路线做横向对比：规则、TF-IDF、RoBERTa 微调、本地 Qwen3 零样本、Qwen3 + LoRA、Qwen3 全量微调、外部 API 零样本。

本次作业的核心改造：**把教师原始代码中的 BERT 骨干替换为中文 RoBERTa-base**，并完成全链路复现与统一评测。

## 环境

| 项 | 版本 / 配置 |
|---|---|
| Python | 3.12.14（conda 环境 `py312`） |
| PyTorch | 2.13.0+cu132 |
| Transformers | 5.23.0 |
| PEFT | 0.22.0 |
| scikit-learn | 1.5.1 |
| GPU | NVIDIA RTX 4060 Laptop，8 GB |

```powershell
conda activate py312
cd "C:\Users\fengj\Desktop\ai\hub-CDtc\冯嘉旺\week06"
```

## 目录结构

```
week06/
├── common/                  # 数据读取、标签映射、指标、模型路径
├── rule/                    # 方法一：规则 / 关键词
├── tfidf/                   # 方法二：TF-IDF + LinearSVC
├── roberta/                 # 方法三：中文 RoBERTa-base 微调（训练 + 推理）
├── llm_zero_shot/           # 方法四：本地 Qwen3-0.6B 零样本
├── llm_api_zero_shot/       # 方法四（外部版）：OpenAI 兼容 API 零样本
├── llm_lora/                # 方法五：Qwen3 LoRA / 全量微调 + 评测
├── eval/                    # 统一汇总
├── data/                    # train.json / val.json
└── outputs/                 # 评测结果与训练日志
```

`pretrain_models/` 放在上一级 `冯嘉旺/pretrain_models/`，供 week06 及后续周复用；体积约 1.9 GB，不提交。

## 数据与评测口径

- 训练集 53,360 条，验证集 10,000 条，15 类中文新闻标题。
- 验证集中有 **1,099 条（11%）与训练集完全重复**，直接用会导致指标虚高。
- 所有方法统一在**去重后的干净验证集**上评测；本次对比统一抽取 **500 条，seed=42**，保证同一批样本。

## 复现命令

```powershell
# 0) 规则（秒级）
python rule\classify_rule.py --n 500

# 1) TF-IDF（约 20 秒，CPU）
python tfidf\classify_tfidf.py --n 500

# 2) RoBERTa 微调（约 9 分钟，GPU）+ 推理（约 1 秒）
python roberta\train_roberta.py
python roberta\predict_roberta.py --n 500

# 3) 本地 Qwen3 zero-shot（约 40 秒）
python llm_zero_shot\classify_zero_shot.py --n 500

# 4) Qwen3 + LoRA（约 8 分钟）+ 评测（约 1 分钟）
python llm_lora\train_lora.py
python llm_lora\eval_lora.py --n 500

# 5) Qwen3 全量微调（约 5 分钟）+ 评测（约 1 分钟）
python llm_lora\train_lora.py --full_ft --num_train 3000 --epochs 2
python llm_lora\eval_lora.py --adapter outputs\full_ft_ckpt --n 500 --out outputs\full_ft_pred.json

# 6) API 零样本（约 40 秒，需 Key）
python llm_api_zero_shot\classify_api_zero_shot.py --n 500 --workers 8

# 7) 汇总
python eval\eval_all.py
```

> API 的 Key 写在 `week06/.env`（`LLM_API_KEY=...`），脚本会自动加载；`.env` 不随作业提交。

## 结果（500 条干净验证样本，seed=42）

| 方法 | 准确率 | Macro-F1 | 覆盖类别 | 解析失败 | 实测耗时 |
|---|---|---|---|---|---|
| 规则 / 关键词 | 0.126 | 0.0716 | 10/15 | 0 | 秒级 |
| TF-IDF + SVM | 0.512 | 0.4721 | 15/15 | 0 | 20 秒 |
| **RoBERTa 微调** | **0.568** | **0.5589** | 15/15 | 0 | 约 9 分钟 |
| 本地 Qwen3 zero-shot | 0.168 | 0.105 | 8/15 | 1 | 约 40 秒 |
| Qwen3 + LoRA SFT | 0.564 | 0.5115 | 14/15 | 0 | 约 8 分钟 |
| Qwen3 全量微调 | 0.534 | 0.4943 | 14/15 | 0 | 约 5 分钟 |
| API zero-shot（deepseek-v4-flash） | 0.528 | 0.5072 | 15/15 | 2 | 约 40 秒 |

完整逐条预测与训练日志见 `outputs/`。

## 结论

1. **封闭标签 + 短文本任务上，判别式模型最划算**：RoBERTa 微调 Macro-F1 0.5589，比 TF-IDF 高 8.7 个点，比最好的生成式（LoRA 0.5115）高 4.7 个点。
2. **本地 0.6B 零样本基本不可用**：只覆盖 8/15 类，预测塌缩到“文化”等少数类，Macro-F1 仅 0.105。
3. **SFT 能修好塌缩**：LoRA 把覆盖类别从 8/15 提到 14/15，F1 0.105 → 0.5115，但仍不如判别式。
4. **LoRA 与全量微调差距不大**：本次 LoRA 0.5115、全量 0.4943。注意两者训练数据量不同（LoRA 5000 条、全量 3000 条），这一组不能直接当作“全量不如 LoRA”的结论；成本上 LoRA 优势明显（显存约 3 GB、adapter 9 MB vs 6.4 GB、权重 1.2 GB）。
5. **API 零样本是低成本冷启动选项**：0.5072，略高于 TF-IDF，但仍低于 RoBERTa，且存在数据出网与按量计费问题。
6. **评测集必须先清洗**：验证集 11% 重复会让词面匹配方法虚高约 3 个点。

## 提交说明

- 已提交：全部源码、`data/`、`outputs/`（汇总表 + 7 份预测结果 + 3 份训练日志）。
- 未提交：`冯嘉旺/pretrain_models/`（约 1.9 GB）、`outputs/roberta_ckpt.pt`（390 MB）、`outputs/lora_adapter/`（9 MB）、`outputs/full_ft_ckpt/`（1.2 GB）、`.env`（含 API Key）。
- 模型权重可按上面的命令重训得到；评测结果已固化在 `outputs/*_pred.json`，不依赖权重即可查看。

## 备注

- `roberta/train_roberta.py` 的骨干为 ModelScope `damo/nlp_roberta_backbone_base_std`（12 层 / 768 维 / 12 头 / 约 102M），并非 Google `bert-base-chinese`。
- 训练脚本未固定随机种子，单次运行存在 1–2 个点的波动；本次结果为本机一次完整复现的实测值。
- `service/` 为可选的工程化演示，本次评测汇总不依赖它。
