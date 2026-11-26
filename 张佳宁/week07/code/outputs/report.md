# Week 07 金融事件抽取实验报告

## 本次完成的工作

1. 在 DuEE-Fin 上运行词典和正则规则，作为抽取基线。
2. 按本周序列标注主线，用中文 RoBERTa-wwm-ext 分别训练 Linear 和 CRF 两种解码器。两组使用完整的 12 层编码器、最大输入长度 256、批大小 32，在 RTX 4080 SUPER 上各训练 3 轮。使用全量训练文件的 5,747 条样本，其中 5,173 条用于训练，574 条用于内层验证。 [Linear 训练日志](logs/bert_linear_256_train.json) · [CRF 训练日志](logs/bert_crf_256_train.json)
3. 使用本地 Qwen3-0.6B 和 Qwen3-1.7B 做 zero-shot 事件抽取。两组使用相同的提示词、BF16、贪心解码、批大小 8，最多生成 384 个 token。 [0.6B 运行记录](llm/report_hw_final_qwen06_n64.json) · [1.7B 运行记录](llm/report_hw_final_qwen17_n64.json)
4. 通过 DeepSeek API 调用 `deepseek-flash` 做 zero-shot 事件抽取，沿用两档 Qwen 的提示词和 384 token 输出上限。 [DeepSeek 运行记录](llm/report_hw_final_deepseek_n64.json)
5. 从去掉训练集重复文本的 DuEE-Fin 验证集中，用种子 42 抽取 64 条样本。六组方法使用同一批金标和同一个评测脚本，均无缺失预测记录。 [评测金标](llm/gold_n64.jsonl) · [统一评测脚本](../common/evaluate_events.py)

## 整体评测结果

| 方法 | 样本数 | 触发词 F1 | 论元 F1 | 完整事件 F1 | 训练耗时（秒） | LLM 抽取耗时（秒） | JSON 解析失败数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [规则（词典与正则）](metrics/disc/hw_final_rule_n64.json) | 64 | 0.5827 | 0.1756 | 0.0315 | — | — | — |
| [RoBERTa + Linear](disc/pred_bert_linear_256.jsonl) | 64 | 0.8031 | 0.5357 | 0.2047 | 135.9 | — | — |
| [RoBERTa + CRF](metrics/disc/hw_final_roberta_crf_n64.json) | 64 | **0.8254** | **0.5691** | **0.2381** | 233.9 | — | — |
| [Qwen3-0.6B zero-shot](metrics/llm/hw_final_qwen06_n64.json) | 64 | 0.3680 | 0.1078 | 0.0000 | — | 32.9 | 0 |
| [Qwen3-1.7B zero-shot](metrics/llm/hw_final_qwen17_n64.json) | 64 | 0.4536 | 0.2420 | 0.0000 | — | 47.5 | 6 |
| [DeepSeek Flash API zero-shot](metrics/llm/hw_final_deepseek_n64.json) | 64 | 0.7460 | 0.5271 | 0.0635 | — | 49.9 | 0 |

论元按角色和原文片段匹配。完整事件要求事件类型、触发词和整组论元全部匹配。LLM 抽取耗时来自各自的运行记录，DeepSeek 计入 API 请求时间，本地 Qwen 使用批量推理。解析失败数按脚本尝试修复后的结果统计。

## 结果分析

- **RoBERTa + CRF 三项得分最高。** 与相同编码器的 Linear 组相比，CRF 的论元 F1 和完整事件 F1 均提高 3.34 个百分点，训练多用 98.0 秒。在这批样本上，规则组的触发词 F1 达到 0.5827，论元 F1 为 0.1756。
- **Qwen3-1.7B 的字段抽取有所改善。** 相比 0.6B，触发词 F1 提高 8.56 个百分点，论元 F1 提高 13.42 个百分点，两组完整事件 F1 均为 0。0.6B 有 19 条输出经结构修复后成功解析，1.7B 有 6 条最终解析失败。
- **DeepSeek Flash 在三组 zero-shot 方法中得分最高。** 相比本地 Qwen3-1.7B，论元 F1 提高 28.51 个百分点，完整匹配 4 条事件。64 条输出均直接解析成功，API 共用 60,364 个 token，其中输入 54,416 个，输出 5,948 个。这组对照同时改变了模型和运行方式。 [DeepSeek 评测](metrics/llm/hw_final_deepseek_n64.json) · [DeepSeek 运行记录](llm/report_hw_final_deepseek_n64.json)
- **错例体现了角色和格式对得分的影响。** 在周大生股东减持的样本中，1.7B 抽对了“减持”和“北极光投资”，但把减持后的持股比例误填为减持部分占总股本的比例。在华兴资本控股回购的样本中，1.7B 写对了触发词和每股交易价格，论元列表中的括号错误导致整条输出解析失败。 [金标样本](llm/gold_n64.jsonl) · [1.7B 原始输出](llm/raw_hw_final_qwen17_n64.jsonl)

本次实验覆盖 DuEE-Fin 的同一批 64 条样本，结论适用于本次抽样和当前 zero-shot 设置。跨数据集表现尚未验证。
