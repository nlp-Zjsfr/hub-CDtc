# CoNLL++ NER 消融实验汇总

## 1. 实验设置

- **数据集**：CoNLL++（4 列 CoNLL 格式，仅使用 token / NER 标签列）
  - 规模：train 14041 / dev 3250 / test 3453 句（已过滤 `-DOCSTART-`）
  - 4 类实体：PER / LOC / ORG / MISC，BIO 标注，共 9 个标签
- **评测口径**：词级 span 精确匹配（实体类型 + 边界均正确才计为 TP）；dev 选最优 checkpoint，test 仅评测一次
- **硬件**：8GB 统一内存，Apple Silicon MPS
- **范围说明**：BERT 4 组消融 + BiLSTM（40 轮版）4 组消融 + Qwen3-0.6B 零样本（base）。**不含 LoRA 与全量微调**——8GB 机器训练 Qwen 时发生 swap 颠簸（5.6GB/6GB 占用，~27s/step）无法在合理时间内完成，按决定终止/跳过（仅余一份 64 样本的冒烟产物，不计入对比）。

---

## 2. 总体结果

| 模型 | 消融 | dev F1 | test F1 | test P | test R | 参数量 | 训练时长 | 推理延迟 (ms/句) | test 非法序列 |
|------|------|--------|---------|--------|--------|--------|----------|------------------|---------------|
| BERT | full | 0.9496 | 0.9184 | 0.9165 | 0.9202 | 108.3M | 28.2 min | 18.75 | 34 |
| BERT | no_crf | 0.9492 | 0.9206 | 0.9176 | 0.9237 | 108.3M | 25.8 min | 15.26 | 46 |
| BERT | crf_hard | 0.9523 | 0.9178 | 0.9181 | 0.9176 | 108.3M | 38.6 min | 19.11 | 25 |
| BERT | len_half | 0.9495 | 0.9170 | 0.9153 | 0.9188 | 108.3M | 27.8 min | 18.08 | 29 |
| BiLSTM | full (40ep) | 0.8483 | 0.7749 | 0.8087 | 0.7438 | 4.5M | 26.8 min | 2.31 | 2 |
| BiLSTM | no_crf (40ep) | 0.8305 | 0.7512 | 0.7841 | 0.7210 | 4.5M | 13.7 min | 2.44 | 233 |
| BiLSTM | crf_hard (40ep) | 0.8489 | 0.7749 | 0.8231 | 0.7320 | 4.5M | 36.2 min | 3.77 | 0 |
| BiLSTM | len_half (40ep) | 0.8521 | 0.7749 | 0.7964 | 0.7546 | 4.5M | 35.4 min | 2.64 | 0 |
| Qwen3-0.6B | base (零样本) | 0.2561 | 0.2535 | 0.4814 | 0.1720 | 596.0M | — | 390.6 | — |

> 说明：
> - 「非法序列」指 BIO 转移非法的句子数（如 `B-X` 后接 `I-Y`）。BERT 用 CRF 软/硬约束解码后统计；BiLSTM 同理。
> - Qwen 为生成式（输出 JSON span 再反查回原词），口径不同：test 有 **15 个预测实体无法反查回原词**（unmatched），计入错误。
> - Qwen base 无训练（纯零样本推理），「训练时长」记为「—」；其 dev+test 两次推理评测共约 45.7 min。

---

## 3. 分类型 F1（test）

| 模型 | 消融 | PER | LOC | ORG | MISC |
|------|------|-----|-----|-----|------|
| BERT | full | 0.9685 | 0.9438 | 0.8969 | 0.8011 |
| BERT | no_crf | 0.9692 | 0.9461 | 0.9004 | 0.8044 |
| BERT | crf_hard | 0.9668 | 0.9432 | 0.8989 | 0.7959 |
| BERT | len_half | 0.9581 | 0.9430 | 0.8966 | 0.8158 |
| BiLSTM | full | 0.8025 | 0.8547 | 0.6955 | 0.7230 |
| BiLSTM | no_crf | 0.7537 | 0.8376 | 0.6823 | 0.7095 |
| BiLSTM | crf_hard | 0.7567 | 0.8659 | 0.7209 | 0.7302 |
| BiLSTM | len_half | 0.8128 | 0.8512 | 0.6897 | 0.7274 |
| Qwen3-0.6B | base | 0.4257 | 0.1921 | 0.1785 | 0.0000 |

---

## 4. 关键发现

### 4.1 编码器能力决定上限
- BERT（test F1 ≈ 0.92）显著优于 BiLSTM（≈ 0.77）和 Qwen 零样本（0.25）。
- Qwen3-0.6B 零样本以**漏抽为主**（R 0.17，P 0.48），MISC 完全抽不出（F1 = 0），说明未经微调的生成式大模型无法直接胜任封闭标签集的 span 抽取。

### 4.2 CRF 的价值随编码器强度变化
- **BERT 上 CRF 几乎无增益**：`no_crf`（0.9206）甚至略高于 `full`（0.9184），四组差距 < 0.004。强编码器的表示已足够，CRF 的结构约束边际收益很小。
- **BiLSTM 上 CRF 关键**：`no_crf`（0.7512）比带 CRF 三组（均 0.7749）低 0.024；且 `no_crf` 产生 **233 条**非法序列，CRF 组降至 0–2 条。对弱编码器，CRF 同时提升精度与输出合法性。

### 4.3 硬约束（crf_hard）的效果
- `crf_hard` 在两个模型上都把非法序列压到最低（BERT 25 / LSTM 0），但 F1 并未同步提升（BERT 0.9178、LSTM 0.7749）。说明「输出合法」≠「输出更准」，硬约束的主要价值在工程层面——保证下游拿到合法可用的 BIO 序列。

### 4.4 长度减半（len_half）
- BERT `len_half`：test F1 0.9170（-0.0014），训练时长与 full 相当，MISC 反而最高（0.8158）。
- BiLSTM `len_half`：test F1 0.7749 与 full 持平（best_epoch 35）。
- 注：BiLSTM 三组（`full` / `crf_hard` / `len_half`）test F1 恰好都是 **0.7749**，是 tp/fp/fn 组合不同导致的数值巧合，已核对非 bug。

### 4.5 效率对比

| 模型 | 参数量 | 推理延迟 | 相对 BERT |
|------|--------|----------|-----------|
| BiLSTM | 4.5M | ~2–4 ms/句 | 0.13× |
| BERT | 108.3M | ~15–19 ms/句 | 1× |
| Qwen3-0.6B | 596.0M | ~390 ms/句 | ~21× |

BiLSTM 最轻最快，适合资源受限场景；Qwen 生成式推理最慢（需逐 token 生成 + JSON 反查）。ORG 是所有模型共同的弱项（BERT ≈ 0.90，LSTM ≈ 0.69–0.72）。

---

## 5. 配置细节

- **BERT**：`bert-base-cased`，3 epoch，batch 8，lr 2e-5，dropout 0.1；`is_split_into_words=True`，词首子片背标签、续片 `-100`；CRF 单独维护 `crf_tags`（续片沿用同词标签，修复金标路径非法 bug）。
- **BiLSTM**：Embedding 200 → BiLSTM 128×2 → dropout 0.5 → CRF/Linear；40 epoch，batch 32，lr 1e-3，Adam；`min_freq=1`（训练集 UNK=0），保留标点入词表。
- **Qwen3-0.6B base**：零样本，prompt 要求输出 JSON `{"entities":[{"text","type"}]}`，`max_new_tokens=64`，bf16，`eval_batch=8`；预测 span 先精确匹配、再模糊匹配反查回原词区间（匹配窗口上限为整句长度）。
- 所有模型：dev 选最优 checkpoint，test 只跑一次；评测口径统一为词级 span 精确匹配。

---

## 6. 产物与未完成项

**已完成产物**（`homework/output/`）：
- `bert/`：4 组 model + log（`metrics_abl_*.json`）
- `lstm/`：4 组 model + log（40 轮版 `*_e40.json`，另含 20 轮版备用）
- `llm/`：base 评测 log（`metrics_base.json`）

**未完成**（按决定跳过）：
- Qwen3-0.6B **LoRA 微调**：训练中因 8GB 内存 swap 颠簸终止，仅余 `metrics_lora_n64.json`（64 样本冒烟，非有效结果）。
- Qwen3-0.6B **全量微调**：按决定未跑。
