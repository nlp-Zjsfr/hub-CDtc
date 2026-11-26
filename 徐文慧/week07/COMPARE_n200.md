# Week7 事件抽取 · 同批 n=200 方法对比

- 评测口径：`common/evaluate_events.py`（trigger / argument / event_exact 三级 F1）
- 样本：dev 固定抽样 **n=200**，`seed=42`（`outputs/llm/gold_n200.jsonl`）
- 本次新增实验：DeepSeek API，`model=deepseek-chat`（未改代码，仅换环境变量重跑）

## 1. 主对比表

| 层次 | 方法 | trigger F1 | argument F1 | event F1 | 说明 |
|------|------|------------|-------------|----------|------|
| L0 | 规则（词典+正则） | 0.5750 | 0.1805 | 0.0200 | 零训练基线 |
| L1 | BiLSTM-CRF | 0.8167 | 0.4633 | 0.1056 | 无预训练 |
| L2 | BERT+Linear | 0.8123 | 0.5598 | 0.1645 | 联合抽取 |
| L2 | BERT+CRF | 0.8278 | 0.5672 | 0.1748 | 判别式主线 |
| L2 | BERT+CRF 硬约束 | 0.8402 | **0.6418** | **0.2268** | 同 ckpt，只换解码 |
| L2 | 两段式流水线 | 0.6522 | 0.3527 | 0.1304 | 检测→类型→论元 |
| L3 | DeepSeek zero（本次） | 0.8184 | 0.5610 | 0.0512 | `deepseek-chat` |
| L3 | DeepSeek few（本次） | **0.8477** | 0.5831 | 0.0812 | `deepseek-chat` + 6 例 |
| L4 | LoRA SFT（Qwen3-0.6B） | **0.9000** | **0.6971** | 0.1850 | 本地微调 |

指标来源：`outputs/metrics/disc/*_n200.json`、`outputs/metrics/llm/*.json`。

## 2. 本次 API vs 项目旧 API（同 n=200）

旧结果见 `NOTES_llm.md`（模型 `deepseek-v4-flash`）；本次覆盖为 `deepseek-chat`。

| 设置 | 模型 | trigger F1 | argument F1 | event F1 | 平均延迟 |
|------|------|------------|-------------|----------|----------|
| zero（旧） | v4-flash | 0.7789 | 0.5224 | 0.0402 | ~3.5 s/条 |
| zero（本次） | **chat** | **0.8184** | **0.5610** | **0.0512** | **0.70 s/条** |
| few（旧） | v4-flash | 0.8321 | 0.5395 | 0.0852 | ~4.1 s/条 |
| few（本次） | **chat** | **0.8477** | **0.5831** | 0.0812 | **0.91 s/条** |

## 3. 结论（简）

1. **主指标看 argument F1**：规则 0.18 → BiLSTM 0.46 → BERT/API ~0.56–0.58 → 硬约束 0.64 → LoRA 0.70。
2. **本次 `deepseek-chat` 优于旧 `v4-flash`**（论元约 +3～4 点，延迟明显下降）。
3. **API few 与普通 BERT+CRF 论元接近**，但 event F1 明显更低（整事件全对更难）。
4. **仍低于** BERT+CRF 硬约束与 LoRA SFT；落地精度上，带约束的判别式 / 任务微调更稳，API 优势是零训练、易换模型。

## 4. 本次产物路径

| 文件 | 作用 |
|------|------|
| `outputs/metrics/llm/api_zero_n200.json` | zero-shot 统一评测 |
| `outputs/metrics/llm/api_few_n200.json` | few-shot 统一评测 |
| `outputs/llm/report_api_zero_n200.json` | 工程统计（模型名/延迟/token） |
| `outputs/llm/report_api_few_n200.json` | 同上 |
| `outputs/llm/pred_api_zero_n200.jsonl` | 预测明细 |
| `outputs/llm/pred_api_few_n200.jsonl` | 预测明细 |
