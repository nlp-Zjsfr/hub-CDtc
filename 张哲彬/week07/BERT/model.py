import torch
import torch.nn as nn
from transformers import BertModel
import transformers
from torchcrf import CRF

class BERTNER(nn.Module):
    def __init__(self, bert_path, num_labels, dropout=0.1, use_crf=False):
        super().__init__()
        previous_verbosity = transformers.logging.get_verbosity()
        try:
            transformers.logging.set_verbosity_error()
            self.bert = BertModel.from_pretrained(bert_path)
        finally:
            transformers.logging.set_verbosity(previous_verbosity)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout)
        self.crf = CRF(num_labels, batch_first=True) if use_crf else None

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # 统一返回发射分数 logits: [B, L, num_labels]
        # 带 CRF 时也返回 logits：CRF 的损失/解码由下面两个方法完成，
        # 这样 train.py / evaluate.py 里普通模型和 CRF 模型可以共用同一套流程。
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        last_hidden_state = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(last_hidden_state)
        return logits

    def crf_loss(self, emissions, labels, attention_mask):
        """CRF 负对数似然损失。

        CRF 需要一条完整的标签链，所以这里用 attention_mask（而不是 labels != -100）
        作为 mask，并把 -100 的位置（[CLS]/[SEP] 等）当成 O 标签（标签表第 0 项）。
        这样训练时的标签链和 crf_decode 时用 attention_mask 解码的标签链完全一致；
        这些位置在计算 F1 时仍会被 calculate_f1 的 labels != -100 过滤掉。
        """
        tags = labels.masked_fill(labels < 0, 0)
        return -self.crf(emissions, tags, mask=attention_mask.bool(), reduction="mean")

    def crf_decode(self, emissions, attention_mask, pad_value=-100):
        """CRF 维特比解码，返回 [B, L]；padding 位置填 -100，可直接与 labels 对齐算 F1。"""
        if self.crf is None:
            return emissions.argmax(dim=-1)
        paths = self.crf.decode(emissions, mask=attention_mask.bool())   # List[List[int]]
        out = emissions.new_full(emissions.shape[:2], pad_value, dtype=torch.long)
        for b, path in enumerate(paths):
            out[b, :len(path)] = torch.as_tensor(path, device=emissions.device)
        return out

def build_model(bert_path, num_labels, dropout=0.1, crf=False):
    model = BERTNER(bert_path, num_labels, dropout=dropout, use_crf=crf)
    n_params = sum(p.numel() for p in model.parameters())
    n_bert = sum(p.numel() for p in model.bert.parameters())
    n_classifier = sum(p.numel() for p in model.classifier.parameters())
    info = f"模型参数总数: {n_params/1e6:.2f}M，其中 BERT 参数: {n_bert/1e6:.2f}M，分类器参数: {n_classifier/1e3:.2f}k"
    if model.crf is not None:
        n_crf = sum(p.numel() for p in model.crf.parameters())
        info += f"，CRF 参数: {n_crf/1e3:.2f}k"
    print(info)
    return model