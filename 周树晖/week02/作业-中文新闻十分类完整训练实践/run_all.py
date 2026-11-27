"""一键复现。建议在 PowerShell 中执行：python -u run_all.py。"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# CUDA 确定性矩阵乘法需要在首次初始化 CUDA 前设置此环境变量。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONUTF8", "1")

ROOT = Path(__file__).resolve().parent


class Tee:
    def __init__(self, terminal, logfile):
        self.terminal, self.logfile = terminal, logfile

    def write(self, value):
        self.terminal.write(value)
        self.logfile.write(value)
        self.logfile.flush()

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "prepare", "train", "evaluate", "report"], default="all")
    parser.add_argument("--skip-robustness", action="store_true", help="快速复现时跳过额外随机种子实验")
    args = parser.parse_args()
    (ROOT / "logs").mkdir(exist_ok=True)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    with (ROOT / "logs" / f"run_{stamp}.log").open("w", encoding="utf-8") as logfile:
        previous_out, previous_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(previous_out, logfile), Tee(previous_err, logfile)
        try:
            print(f"作业目录: {ROOT}\n开始时间: {stamp} Asia/Shanghai", flush=True)
            if args.stage in ["all", "prepare"]:
                from src.prepare_data import prepare
                prepare()
            if args.stage in ["all", "train"]:
                from src.train import train_all
                train_all(skip_robustness=args.skip_robustness)
            if args.stage in ["all", "evaluate"]:
                from src.evaluate import evaluate_final
                evaluate_final()
            if args.stage in ["all", "report"]:
                from src.build_report import build_report
                build_report()
            print("指定流程执行完成。", flush=True)
        finally:
            sys.stdout, sys.stderr = previous_out, previous_err


if __name__ == "__main__":
    main()
