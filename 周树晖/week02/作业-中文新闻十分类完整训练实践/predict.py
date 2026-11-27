"""加载保存的完整流水线，预测新标题。示例：python predict.py --text 中国女排晋级决赛"""
import argparse
import json

import joblib
import numpy as np
import torch

from src.common import ROOT, read_json
from src.model import load_checkpoint
from src.prepare_data import normalize_text


class NewsPredictor:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.classes = read_json(ROOT / "models/classes.json")
        self.vectorizer = joblib.load(ROOT / "models/tfidf.joblib")
        self.model, self.checkpoint = load_checkpoint(ROOT / "models/best_model.pt", self.device)

    @torch.inference_mode()
    def predict(self, texts, top_k=3):
        if not 1 <= top_k <= len(self.classes):
            raise ValueError("top_k 必须在1到10之间")
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("请输入至少一条非空字符串标题")
        matrix = self.vectorizer.transform([normalize_text(t) for t in texts]).tocsr()
        results = []
        # 分批推理避免大型文件一次展开成密集矩阵耗尽内存。
        for start in range(0, len(texts), 256):
            part = matrix[start:start + 256]
            probs = torch.softmax(self.model(torch.from_numpy(part.toarray()).to(self.device)), dim=1).cpu().numpy()
            for offset, probability in enumerate(probs):
                index = start + offset
                if part[offset].nnz == 0:
                    results.append({"text": texts[index], "status": "insufficient_features", "top3": [],
                                    "message": "标题未命中训练词表，无法给出有效分类"})
                    continue
                order = np.argsort(-probability)[:top_k]
                results.append({"text": texts[index], "status": "ok", "prediction": self.classes[int(order[0])],
                                "top3": [{"label": self.classes[int(j)], "probability": float(probability[j])} for j in order]})
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", nargs="+", help="一条或多条标题，有空格时用引号括住")
    parser.add_argument("--file", help="UTF-8文本文件，每行一条标题")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    titles = args.text or []
    if args.file:
        from pathlib import Path
        titles.extend(t for t in Path(args.file).read_text(encoding="utf-8-sig").splitlines() if t.strip())
    if not titles:
        parser.error("需要 --text 或 --file")
    predictor = NewsPredictor(args.device)
    print(json.dumps(predictor.predict(titles), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
