# 第 6 周代码：数据接入与分流（五种方法 + 一个服务）

> 每个方法一个文件夹，互不依赖；`common/` 是共用的数据、标签口径和评测指标。
> 所有方法都跑在**同一批「去重后的干净验证样本」**上（500 条，seed=42），数字可直接横向比较。

## 目录

```
code/
├── common/                 # 共用：标签口径 / 数据切分 / 指标 / 模型路径
│   ├── labels.py           # 15 类 → 4 组主题 + 相关性 + 队列规则
│   ├── data.py             # 读数据、去重（剔除 11% 泄漏）、固定 seed 抽样
│   ├── metrics.py          # Accuracy / Macro-F1 / per-class / 混淆矩阵 / ECE
│   └── model_paths.py      # 定位本地模型（RoBERTa 在项目根目录，Qwen3 在 pretrain_models/）
├── rule/                   # 方法一：规则 / 关键词（L0）
├── tfidf/                  # 方法二：TF-IDF + LinearSVC（L1）
├── roberta/                # 方法三：RoBERTa 微调（L2，含训练与推理）
├── llm_zero_shot/          # 方法四：Qwen3-0.6B 本地零样本（L3）
├── llm_api_zero_shot/      # 方法四·外部版：调大模型 API 做零样本（同样不训练）
├── llm_lora/               # 方法五：Qwen3-0.6B + LoRA SFT（L4）
├── service/                # L5 系统组合：规则快筛 + 模型 + 阈值 + 队列；含早报 demo 与 CLI 分流服务
├── eval/                   # 统一汇总（eval_all.py）+ 换基座配对对比（compare_backbones.py）
├── data/                   # train.json / val.json（TNEWS）
└── outputs/                # 逐条预测 + 汇总表（ckpt / adapter 需自己跑训练生成，不随包分发）
```

> 目录 `roberta/` 原名 `bert/`（脚本 `train_roberta.py` / `predict_roberta.py`，模型类 `RobertaClassifier`）；
> 服务的后端 id 统一成 `roberta-cls-v1`，`--model bert` 已废弃，改用 `--model roberta`。

## 五分钟跑通

```bash
# 0) 依赖
pip install torch transformers peft scikit-learn matplotlib tqdm

# 1) 规则（秒级）
python rule/classify_rule.py --n 500

# 2) TF-IDF（20 秒）
python tfidf/classify_tfidf.py --n 500

# 3) RoBERTa：训练约 26 分钟（RTX 2060 6GB，3 epoch / seed=42），之后推理 500 条约 30 秒
python roberta/train_roberta.py
python roberta/predict_roberta.py --n 500
# 训练技巧消融（下面三个数字是在 BERT-base 基线上测的，换基座后未重跑）
python roberta/train_roberta.py --label_smoothing 0.1   # 软标签（实测 −0.7 个点）
python roberta/train_roberta.py --focal_gamma 2         #            Focal Loss（实测 ±0）
python roberta/train_roberta.py --fgm                   #            FGM 对抗训练（实测 +0.02，慢 1.5 倍）
python roberta/train_roberta.py --epochs 1 --max_steps 3 --max_eval 300 --no_save   # 冒烟：只验流程不落盘

# 4) 大模型两条路线（需要 GPU；0.6B 模型约 1.2 GB）
python llm_zero_shot/classify_zero_shot.py --n 500     # 约 5 分钟
python llm_lora/train_lora.py --num_train 5000 --epochs 2  # LoRA，约 44 分钟
python llm_lora/train_lora.py --full_ft --num_train 3000 --epochs 2 --gradient_checkpointing  # 全量微调，约 68 分钟
python llm_lora/eval_lora.py --adapter outputs/full_ft_ckpt --n 500 --out outputs/full_ft_pred.json
python llm_lora/eval_lora.py --n 500                   # 约 7 分钟
python llm_api_zero_shot/classify_api_zero_shot.py --n 500   # 调 API，约 8 分钟（需 key）

# 5) 汇总成一张表
python eval/eval_all.py

# 6) 组装成服务，看早报长什么样
python service/demo_early_report.py

# 7) 起一个分流服务（交互式 CLI）：输入标题 → 返回接口契约 JSON（对应课件 P43）
python service/classify_cli.py
python service/classify_cli.py --text "贵州茅台：2023 年年度业绩预告" --json
python service/classify_cli.py --file inputs.txt --json        # 批量，一条一行
python service/classify_cli.py --model tfidf                  # 没有 RoBERTa 权重也能跑
python service/classify_cli.py --use-rules                    # 打开规则快筛，演示 L5 级联
```

## 结果（500 条干净验证样本，seed=42）

| 方法 | 准确率(含失败) | Macro-F1 | 覆盖类别 | 训练成本 |
|---|---|---|---|---|
| 规则 / 关键词 | 0.126 | 0.072 | 10/15 | — |
| TF-IDF + SVM | 0.512 | 0.472 | 15/15 | 20 秒 |
| BERT-base（旧基座，历史数字） | 0.556 | 0.544 | 15/15 | ~14 分钟 |
| **RoBERTa-base（当前基座）** | **0.554** | **0.550** | 15/15 | ~26 分钟 |
| 本地 Qwen3 zero-shot | 0.168 | 0.105 | 8/15（塌缩） | — |
| 本地 Qwen3 + LoRA SFT | 0.530 | 0.484 | 14/15 | 44 分钟 / 3 GB |
| 本地 Qwen3 全量微调 | 0.536 | 0.492 | 14/15 | 68 分钟 / 5.58 GB |
| API zero-shot（deepseek-v4-flash） | 0.514 | 0.471 | 15/15 | 按 token 计费 |

> 结论：封闭标签 + 短文本 → 判别式（RoBERTa 这一路）最划算；本地 0.6B zero-shot 不可用，换 API 大模型能到 0.471（追平 TF-IDF、仍低于判别式）；LoRA 与全量微调只差 0.8 个点，成本差一个量级。
>
> **「换基座」到底涨了没有**：500 条上两行几乎一样（准确率 −0.2 个点、Macro-F1 +0.6 个点），
> 配对检验**测不出差异**——McNemar 精确 p=1.00，配对 bootstrap 差值 95%CI `[−3.8, +5.3]` 个点。
> 换到 8,901 条全量干净集才看出来：**RoBERTa 0.5548 vs BERT 0.534（+2.1 个点）**（RoBERTa 自身 95%CI `[0.543, 0.567]`）。
> 也就是说真实收益就是 **1~2 个点**量级，而 500 条的置信区间宽达 10 个点——**样本量决定了你能看见多大的差异**。
> 复现命令见 `eval/compare_backbones.py`（配对 McNemar + bootstrap + 逐类 recall 差值）。
>
> 补充（数据量实验，在旧基座 BERT 上做）：训练集砍到 15%（8,005 条）时，判别式与 TF-IDF 的差距从 **2.6 个点拉到 5.7 个点**
> —— TF-IDF 掉 4.9 个点、BERT 只掉 1.9 个点。也就是说，**TF-IDF 的主场是数据封闭的小样本场景**，
> 而通用短文本上它只是基线。详细表格见 [ARCHITECTURE.md](ARCHITECTURE.md) 的「数据量实验」一节。

## 三份文档

- [ARCHITECTURE.md](ARCHITECTURE.md)：方案对比、数据口径、选型理由、实验结果、踩坑表
- [USAGE_GUIDE.md](USAGE_GUIDE.md)：每个脚本的命令与预期输出、作为模块调用、FAQ
- [RESUME_GUIDE.md](RESUME_GUIDE.md)：这段工作怎么写进简历、面试会追问什么

