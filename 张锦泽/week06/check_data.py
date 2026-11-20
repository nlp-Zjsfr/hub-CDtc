"""不安装机器学习依赖也可以检查 CSV 格式和数据划分。"""

import argparse
import json

from common import ROOT, load_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default=str(ROOT / "data"))
    args = parser.parse_args()
    try:
        _, labels, counts, digest = load_splits(args.data_dir)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"数据检查失败：{exc}\n")
    print(json.dumps({"labels": labels, "counts": counts, "data_sha256": digest}, ensure_ascii=False, indent=2))
    print("数据检查通过：非空、标签一致、集合内及集合间无规范化重复文本。")


if __name__ == "__main__":
    main()
