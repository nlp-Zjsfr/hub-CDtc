import torch
from seqeval.metrics import f1_score, classification_report
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
import pathlib
import argparse

from dataset import PeopleDailyDataset, load_labels
from model import build_model

ROOT_DIR = pathlib.Path(__file__).parent.parent
DATA_PATH = ROOT_DIR / "peoples_daily"
TEST_PATH = DATA_PATH / "test.json"
LABELS_PATH = DATA_PATH / "label_names.json"
BERT_PATH = ROOT_DIR.parent / "pretrain_models" / "bert-base-chinese"
CKPT_DIR = ROOT_DIR / "checkpoints"

def calculate_f1(predictions, labels, idx2labels):
    """
    根据预测 ID 和真实标签 ID 计算实体级 F1。

    predictions: [B, L]，每个位置的预测标签 ID
    labels:      [B, L]，真实标签 ID；-100 位置会被忽略
    idx2labels:  标签 ID 到标签字符串的映射
    """
    predictions = torch.as_tensor(predictions).detach().cpu()
    labels = torch.as_tensor(labels).detach().cpu()

    if predictions.shape != labels.shape:
        raise ValueError(
            f"预测和标签形状必须一致，得到 {predictions.shape} 和 {labels.shape}"
        )

    true_sequences = []
    pred_sequences = []

    for pred_row, label_row in zip(predictions, labels):
        valid_mask = label_row != -100
        true_ids = label_row[valid_mask].tolist()
        pred_ids = pred_row[valid_mask].tolist()

        if not true_ids:
            continue

        true_sequences.append([idx2labels[i] for i in true_ids])
        pred_sequences.append([idx2labels[i] for i in pred_ids])

    if not true_sequences:
        return 0.0

    return f1_score(true_sequences, pred_sequences, zero_division=0)


def evaluate(model, loader, device, idx2labels, print_report=True):
    """遍历验证集并计算实体级 F1。"""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            if model.crf is None:
                predictions = logits.argmax(dim=-1)
            else:
                # CRF：维特比解码整条最优标签序列（padding 位置为 -100，会被 F1 过滤）
                predictions = model.crf_decode(logits, attention_mask)

            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())

    if not all_predictions:
        raise ValueError("验证集为空，无法计算 F1")

    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    score = calculate_f1(all_predictions, all_labels, idx2labels)

    if print_report:
        true_sequences = []
        pred_sequences = []

        for pred_row, label_row in zip(all_predictions, all_labels):
            valid_mask = label_row != -100
            true_ids = label_row[valid_mask].tolist()
            pred_ids = pred_row[valid_mask].tolist()

            if true_ids:
                true_sequences.append([idx2labels[i] for i in true_ids])
                pred_sequences.append([idx2labels[i] for i in pred_ids])

        print(classification_report(
            true_sequences,
            pred_sequences,
            digits=4,
            zero_division=0,
        ))

    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT NER 评估")
    parser.add_argument("--crf",            action="store_true", help="是否使用CRF")
    parser.add_argument("--max_length",     default=128, type=int)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"使用设备: {device}")

    # 模型验证
    tokenizer = BertTokenizerFast.from_pretrained(BERT_PATH)
    labels2idx, idx2labels = load_labels(LABELS_PATH)
    dataset = PeopleDailyDataset(TEST_PATH, tokenizer, labels2idx, max_length=args.max_length)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    model = build_model(
        bert_path=BERT_PATH,
        num_labels=len(idx2labels),
        dropout=0.1,
        crf=args.crf,
    )
    # 加载最优 checkpoint
    ckpt_path = CKPT_DIR / f"best_BERT.pt" if not args.crf else CKPT_DIR / f"best_BERT_CRF.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    f1 = evaluate(model, loader, device, idx2labels)
    print(f"F1 score: {f1:.4f}")