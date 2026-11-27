"""加载已保存的模型重新评估 CSV，不重新训练。"""

import argparse
import hashlib
import json
from pathlib import Path

from common import (ROOT, environment_info, make_loader, neural_predictions, read_data,
                    resolve_device, save_evaluation, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "test.csv")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size 必须为正数")
    output_dir = args.output_dir or args.model_dir / "re_evaluation"
    if output_dir.resolve() == args.model_dir.resolve():
        parser.error("评估输出目录必须与模型目录不同，以保留原始训练结果")
    try:
        metadata = json.loads((args.model_dir / "metadata.json").read_text(encoding="utf-8"))
        texts, names = read_data(args.data)
        labels = metadata["labels"]
        label_to_id = {name: index for index, name in enumerate(labels)}
        unknown = set(names) - set(labels)
        if unknown:
            raise ValueError(f"评估文件出现未知标签：{sorted(unknown)}")
        truth = [label_to_id[name] for name in names]
        if metadata["kind"] == "sklearn":
            import joblib

            model = joblib.load(args.model_dir / "model.joblib")
            predicted = model.predict(texts).tolist()
            device_name = "cpu"
        elif metadata["kind"] == "transformer":
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            device = resolve_device(args.device)
            tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, local_files_only=True).to(device)
            loader = make_loader(texts, truth, tokenizer, metadata["max_length"], args.batch_size)
            predicted = neural_predictions(model, loader, device)
            device_name = str(device)
        else:
            raise ValueError(f"不支持的模型类型：{metadata['kind']}")
        metrics = save_evaluation(output_dir, texts, truth, predicted, labels)
        write_json(output_dir / "evaluation_info.json", {
            "model_dir": str(args.model_dir.resolve()), "data": str(args.data.resolve()),
            "evaluation_file_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
            "training_data_sha256": metadata["data_sha256"], "device": device_name,
            "environment": environment_info(),
        })
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f"评估失败：{exc}\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"评估文件已保存：{output_dir}")


if __name__ == "__main__":
    main()
