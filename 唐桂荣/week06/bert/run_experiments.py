"""依次跑完消融实验，重启时从第一个未完成的实验继续。"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from bert.experiments import MODEL_NAMES, RUN_ORDER

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_DIR / "outputs"


def read_completed(directory: Path, model: str, experiment: str) -> dict | None:
    report_path = directory / "result.json"
    if not report_path.is_file() or not (directory / "best.pt").is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["status"] != "complete":
        return None
    if report["model"] != model or report["experiment"] != experiment or report["split"] != "val":
        raise ValueError(f"结果与任务不匹配：{report_path}")
    if len(report["history"]) != report["config"]["epochs"]:
        raise ValueError(f"结果缺少训练轮次：{report_path}")
    return report


def archive_interrupted(directory: Path):
    archive = directory.parent / "_interrupted"
    archive.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = archive / f"{directory.name}-{stamp}"
    directory.rename(destination)
    print(f"保留中断记录到 {destination.relative_to(PROJECT_DIR)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="批量运行两套模型的消融对照")
    parser.add_argument("--experiments", nargs="+", choices=RUN_ORDER, default=RUN_ORDER)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="mps")
    parser.add_argument("--finalize", action="store_true", help="完成全部验证实验后统一评测测试集")
    args = parser.parse_args()

    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    root = OUTPUTS / "bert"
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".batch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("已有一批消融实验正在运行")
        failures = []
        for experiment in dict.fromkeys(args.experiments):
            for model in MODEL_NAMES:
                directory = root / model / experiment
                if read_completed(directory, model, experiment) is not None:
                    print(f"跳过已完成的 {model} {experiment}", flush=True)
                    continue
                if directory.exists():
                    archive_interrupted(directory)
                print(f"开始 {model} {experiment}", flush=True)
                result = subprocess.run(
                    [sys.executable, "-B", "-u", "-m", "bert.train_bert",
                     "--model", model, "--experiment", experiment, "--device", args.device],
                    cwd=PROJECT_DIR, env=env, pass_fds=(lock.fileno(),),
                )
                if result.returncode or read_completed(directory, model, experiment) is None:
                    print(f"失败 {model} {experiment}，进程状态 {result.returncode}", flush=True)
                    failures.append(f"{model} {experiment}")
                else:
                    print(f"完成 {model} {experiment}", flush=True)
        if failures:
            sys.exit("以下实验需续跑：" + "、".join(failures))
        if args.finalize:
            subprocess.run(
                [sys.executable, "-B", "-u", "-m", "eval.evaluate_experiments", "--device", args.device],
                cwd=PROJECT_DIR, env=env, check=True, pass_fds=(lock.fileno(),),
            )


if __name__ == "__main__":
    main()
