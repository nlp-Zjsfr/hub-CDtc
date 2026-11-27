"""训练与验证：本模块不会读取测试集标题、测试标签或测试特征。"""
import copy
import importlib.metadata
import math
import platform
import random
import shutil
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from .common import ROOT, dump_json, read_json, sha256
from .model import NewsClassifier


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


def metric_dict(labels, predictions):
    return {"accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, labels=list(range(10)), average="macro", zero_division=0))}


class FeatureCache:
    """显存足够时缓存特征，避免每批反复搬运；不足时仅把当前稀疏批次转为密集矩阵。"""
    def __init__(self, matrices, device):
        self.device = device
        self.matrices = matrices
        self.cached = {}
        required = sum(matrix.shape[0] * matrix.shape[1] * 4 for matrix in matrices.values())
        use_gpu_cache = device.type == "cuda" and required < torch.cuda.mem_get_info()[0] * 0.65
        self.mode = "GPU cached float32" if use_gpu_cache else "CSR mini-batch conversion"
        if use_gpu_cache:
            for name, matrix in matrices.items():
                self.cached[name] = torch.from_numpy(matrix.toarray()).to(device)
        print(f"特征读取方式: {self.mode}, 全部密集特征约 {required / 1024**3:.2f} GiB", flush=True)

    def get(self, name, indices):
        if name in self.cached:
            index_tensor = torch.as_tensor(indices, dtype=torch.long, device=self.device)
            return self.cached[name].index_select(0, index_tensor)
        array_indices = indices.numpy() if isinstance(indices, torch.Tensor) else indices
        return torch.from_numpy(self.matrices[name][array_indices].toarray()).to(self.device)


@torch.inference_mode()
def validate(model, cache, labels, split="val", batch_size=1024):
    model.eval()
    total_loss, predicted = 0.0, []
    for start in range(0, len(labels), batch_size):
        indices = np.arange(start, min(start + batch_size, len(labels)))
        xb = cache.get(split, indices)
        yb = torch.from_numpy(labels[indices]).to(cache.device)
        logits = model(xb)
        total_loss += nn.functional.cross_entropy(logits, yb, reduction="sum").item()
        predicted.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    return {**metric_dict(labels, predicted), "loss": total_loss / len(labels)}


def train_one(config, seed, cache, train_y, val_y, input_dim, data_signature, output_name=None):
    name = output_name or config["name"]
    folder = ROOT / "outputs/experiments" / name
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ROOT / "models" / f"{name}.pt"
    result_path = folder / "summary.json"
    if result_path.exists() and checkpoint_path.exists():
        old = read_json(result_path)
        if old["config"] == config and old["seed"] == seed and old["data_signature"] == data_signature:
            print(f"复用已完成实验: {name}", flush=True)
            return old
    seed_everything(seed)
    model = NewsClassifier(input_dim, config["hidden_dims"], 10, config["dropout"]).to(cache.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, threshold=config["min_delta"], threshold_mode="abs"
    ) if config["scheduler"] else None
    # DataLoader 每轮打乱样本；生成器独立于模型初始化，保证同 seed、同 batch 实验样本顺序一致。
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(range(len(train_y)), batch_size=config["batch_size"], shuffle=True,
                        num_workers=0, generator=generator, drop_last=False)
    labels_on_device = torch.from_numpy(train_y).to(cache.device)
    best_f1, best_loss, best_epoch, waiting = -1.0, math.inf, 0, 0
    patience_anchor = -1.0
    history = []
    start_time = time.perf_counter()
    print(f"\n开始实验 {name}: seed={seed}, hidden={config['hidden_dims']}, lr={config['lr']}, batch={config['batch_size']}", flush=True)
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        epoch_start = time.perf_counter()
        lr_used = optimizer.param_groups[0]["lr"]
        loss_sum = torch.zeros((), device=cache.device)
        correct_sum = torch.zeros((), dtype=torch.long, device=cache.device)
        max_grad_norm = torch.zeros((), device=cache.device)
        for indices in loader:
            xb = cache.get("train", indices)
            yb = labels_on_device.index_select(0, indices.to(cache.device))
            optimizer.zero_grad(set_to_none=True)    # 清除上一批梯度
            logits = model(xb)                      # 前向传播，形状 B×10
            loss = criterion(logits, yb)            # 整数标签形状 B，不提前 Softmax
            loss.backward()                        # 反向传播，得到各参数的梯度
            norm = nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
            optimizer.step()                       # Adam 根据梯度更新参数
            loss_sum += loss.detach() * len(indices)
            correct_sum += (logits.detach().argmax(dim=1) == yb).sum()
            max_grad_norm = torch.maximum(max_grad_norm, norm.detach())
        train_loss = loss_sum.item() / len(train_y)
        if not math.isfinite(train_loss):
            raise FloatingPointError(f"{name} 的训练损失不是有限数")
        val_metrics = validate(model, cache, val_y)
        if scheduler is not None:
            scheduler.step(val_metrics["macro_f1"])
        record = {"epoch": epoch, "train_loss": train_loss,
                  "train_accuracy_online": correct_sum.item() / len(train_y),
                  "val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"],
                  "val_macro_f1": val_metrics["macro_f1"], "lr_used": lr_used,
                  "lr_next": optimizer.param_groups[0]["lr"], "max_gradient_norm_before_clip": max_grad_norm.item(),
                  "seconds": time.perf_counter() - epoch_start}
        history.append(record)
        # 最佳 checkpoint 按精确分数排名；早停使用 min_delta，避免微小抖动无限延长训练。
        improved = val_metrics["macro_f1"] > best_f1 + 1e-12 or (
            abs(val_metrics["macro_f1"] - best_f1) <= 1e-12 and val_metrics["loss"] < best_loss)
        if improved:
            best_f1, best_loss, best_epoch = val_metrics["macro_f1"], val_metrics["loss"], epoch
            torch.save({"state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "config": config, "input_dim": input_dim, "num_classes": 10,
                        "seed": seed, "best_epoch": epoch, "validation": val_metrics,
                        "data_signature": data_signature}, checkpoint_path)
        if val_metrics["macro_f1"] > patience_anchor + config["min_delta"]:
            patience_anchor, waiting = val_metrics["macro_f1"], 0
        else:
            waiting += 1
        dump_json(folder / "history.json", history)
        print(f"{name} epoch {epoch:02d}: train_loss={train_loss:.4f}, val_loss={val_metrics['loss']:.4f}, "
              f"val_acc={val_metrics['accuracy']:.4%}, val_F1={val_metrics['macro_f1']:.4%}, lr={lr_used:g}, {record['seconds']:.1f}s", flush=True)
        if waiting >= config["early_stopping_patience"]:
            print(f"{name}: 连续{waiting}轮验证Macro-F1未显著改善，提前停止；恢复第{best_epoch}轮。", flush=True)
            break
    best_record = history[best_epoch - 1]
    summary = {"name": name, "config": config, "seed": seed, "data_signature": data_signature,
               "best_epoch": best_epoch, "epochs_ran": len(history),
               "best_val_macro_f1": best_record["val_macro_f1"], "best_val_accuracy": best_record["val_accuracy"],
               "best_val_loss": best_record["val_loss"], "parameter_count": sum(p.numel() for p in model.parameters()),
               "seconds": time.perf_counter() - start_time,
               "checkpoint": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
               "stopped_early": len(history) < config["max_epochs"],
               "steps_per_epoch": len(loader), "total_steps": len(loader) * len(history)}
    dump_json(result_path, summary)
    del model, optimizer
    return summary


def train_all(skip_robustness=False):
    started = time.perf_counter()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    packages = {name: importlib.metadata.version(name) for name in ["torch", "numpy", "scipy", "scikit-learn", "matplotlib", "joblib"]}
    try:
        packages["tzdata"] = importlib.metadata.version("tzdata")
    except importlib.metadata.PackageNotFoundError:
        pass  # 部分系统直接使用系统时区数据库。
    environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                   "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                   "cuda_build": torch.version.cuda, "packages": packages, "torch_num_threads": torch.get_num_threads(),
                   "deterministic_algorithms": True, "timezone": "Asia/Shanghai",
                   "training_started": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}
    dump_json(ROOT / "logs/environment.json", environment)
    (ROOT / "requirements-lock.txt").write_text("\n".join(f"{k}=={v}" for k, v in packages.items()) + "\n", encoding="utf-8")
    matrices = {s: sparse.load_npz(ROOT / "data/features" / f"{s}.npz") for s in ["train", "val"]}
    train_y, val_y = [np.load(ROOT / "data/features" / f"{s}_labels.npy") for s in ["train", "val"]]
    signature = {"train": sha256(ROOT / "data/processed/train.jsonl"), "val": sha256(ROOT / "data/processed/val.jsonl"),
                 "vectorizer": sha256(ROOT / "models/tfidf.joblib")}
    cache = FeatureCache(matrices, device)
    environment["feature_cache_mode"] = cache.mode
    dump_json(ROOT / "logs/environment.json", environment)
    suite = read_json(ROOT / "configs/experiments.json")
    dump_json(ROOT / "outputs/experiment_plan.json", suite)
    majority_label = int(np.bincount(train_y, minlength=10).argmax())
    majority_val = metric_dict(val_y, np.full(len(val_y), majority_label))
    summaries = []
    for item in suite["experiments"]:
        config = {**suite["defaults"], **item}
        summaries.append(train_one(config, suite["seed"], cache, train_y, val_y, matrices["train"].shape[1], signature))
        dump_json(ROOT / "outputs/experiment_summary.json", summaries)
    mlps = [summary for summary in summaries if summary["config"]["hidden_dims"]]
    rank = lambda x: (x["best_val_macro_f1"], -x["best_val_loss"])
    selected = max(mlps, key=rank)
    linear = next(summary for summary in summaries if summary["name"] == "linear_adam")
    # 额外种子只审查选中配置的稳定性，不择优挑选随机种子，也不访问测试集。
    robustness = []
    if not skip_robustness:
        for seed in suite["robustness_seeds"]:
            robustness.append(train_one(copy.deepcopy(selected["config"]), seed, cache, train_y, val_y,
                                        matrices["train"].shape[1], signature, output_name=f"robustness_seed_{seed}"))
    dump_json(ROOT / "outputs/robustness_summary.json", robustness)
    shutil.copy2(ROOT / selected["checkpoint"], ROOT / "models/best_model.pt")
    shutil.copy2(ROOT / linear["checkpoint"], ROOT / "models/linear_baseline.pt")
    selection = {"frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                 "selection_rule": suite["selection"], "main_model": selected, "linear_baseline": linear,
                 "majority_label": majority_label, "majority_validation": majority_val,
                 "main_checkpoint_sha256": sha256(ROOT / "models/best_model.pt"),
                 "linear_checkpoint_sha256": sha256(ROOT / "models/linear_baseline.pt"),
                 "vectorizer_sha256": sha256(ROOT / "models/tfidf.joblib"),
                 "test_processed_sha256": sha256(ROOT / "data/processed/test.jsonl"),
                 "test_evaluation_plan": ["majority_baseline", "linear_baseline", "selected_mlp"],
                 "total_training_seconds": time.perf_counter() - started}
    dump_json(ROOT / "outputs/selection.json", selection)
    print(f"\n模型已冻结：主模型 {selected['name']}, 验证Macro-F1={selected['best_val_macro_f1']:.4%}。", flush=True)
    print("训练阶段完成，尚未读取测试集做模型评估。", flush=True)
    return selection


if __name__ == "__main__":
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    train_all()
