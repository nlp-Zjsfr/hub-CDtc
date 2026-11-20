"""L5 分流 CLI 在 clean_val 上的分类指标。

评的是路由给出的手法标签，队列本身不另算一类。默认路径与 CLI 一致：TF-IDF + SGD。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_clean_val
from common.labels import LABELS
from common.metrics import classification_report
from service.classify_cli import Router

OUT = Path(__file__).resolve().parent.parent / "outputs"


def main():
    samples, meta = load_clean_val()
    router = Router()
    ps = [router.classify(d["sentence"])["label_id"] for d in samples]
    ys = [d["label"] for d in samples]
    report = classification_report(ys, ps, num_classes=len(LABELS))
    report["meta"] = meta
    report["method"] = "router"

    OUT.mkdir(parents=True, exist_ok=True)
    pred = [{"idx": d["idx"], "gold": d["label"], "pred": int(p), "sentence": d["sentence"]}
            for d, p in zip(samples, ps)]
    (OUT / "router_pred.json").write_text(json.dumps(pred, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "router_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[router] n={report['n']} micro_f1={report.get('micro_f1')} macro_f1={report.get('macro_f1')}")


if __name__ == "__main__":
    main()
