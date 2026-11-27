# -*- coding: utf-8 -*-
"""统一评测口径：trigger / argument / event_exact + type_macro_f1 + role_macro_f1。"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from io_utils import as_event_list, load_jsonl, row_key


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": tp, "fp": fp, "fn": fn}


def compare_events(gold_events, pred_events):
    g_by_type, p_by_type = defaultdict(list), defaultdict(list)
    for e in gold_events:
        g_by_type[e["type"]].append(e)
    for e in pred_events:
        p_by_type[e["type"]].append(e)

    tr_tp = tr_fp = tr_fn = 0
    ar_tp = ar_fp = ar_fn = 0
    ev_tp = ev_fp = ev_fn = 0

    for t in set(g_by_type) | set(p_by_type):
        gs, ps = g_by_type.get(t, []), p_by_type.get(t, [])
        n = max(len(gs), len(ps))
        for i in range(n):
            g = gs[i] if i < len(gs) else None
            p = ps[i] if i < len(ps) else None
            if g is None:
                ev_fp += 1
                tr_fp += 1 if p["trigger"] else 0
                ar_fp += len(p["args"])
                continue
            if p is None:
                ev_fn += 1
                tr_fn += 1 if g["trigger"] else 0
                ar_fn += len(g["args"])
                continue
            if p["trigger"] and p["trigger"] == g["trigger"]:
                tr_tp += 1
            else:
                tr_fp += 1 if p["trigger"] else 0
                tr_fn += 1 if g["trigger"] else 0
            gc, pc = Counter(g["args"]), Counter(p["args"])
            inter = sum((gc & pc).values())
            ar_tp += inter
            ar_fp += sum(pc.values()) - inter
            ar_fn += sum(gc.values()) - inter
            if p["trigger"] == g["trigger"] and gc == pc:
                ev_tp += 1
            else:
                ev_fp += 1
                ev_fn += 1
    return (tr_tp, tr_fp, tr_fn), (ar_tp, ar_fp, ar_fn), (ev_tp, ev_fp, ev_fn)


def type_macro_f1(gold_rows, pred_rows):
    g_map, p_map = {}, {}
    for r in gold_rows:
        g_map[row_key(r)] = Counter(e["type"] for e in as_event_list(r))
    for r in pred_rows:
        p_map[row_key(r)] = Counter(e["type"] for e in as_event_list(r))
    types = set(t for c in g_map.values() for t in c) | set(t for c in p_map.values() for t in c)
    per = {}
    for t in sorted(types):
        tp = sum(min(g_map[i].get(t, 0), p_map.get(i, Counter()).get(t, 0)) for i in g_map)
        gp = sum(g_map[i].get(t, 0) for i in g_map)
        pp = sum(p_map.get(i, Counter()).get(t, 0) for i in g_map)
        per[t] = prf(tp, pp - tp, gp - tp)
    macro = sum(v["f1"] for v in per.values()) / len(per) if per else 0.0
    return round(macro, 4), per


def role_macro_f1(per_role: dict) -> float:
    if not per_role:
        return 0.0
    return round(sum(v["f1"] for v in per_role.values()) / len(per_role), 4)


def evaluate(gold_rows, pred_rows, verbose=True):
    p_map = {row_key(r): r for r in pred_rows}
    missing = 0
    tr = [0, 0, 0]
    ar = [0, 0, 0]
    ev = [0, 0, 0]
    per_role_tp, per_role_gp, per_role_pp = Counter(), Counter(), Counter()
    per_type_tp, per_type_gp, per_type_pp = Counter(), Counter(), Counter()

    for g in gold_rows:
        gold_events = as_event_list(g)
        p = p_map.get(row_key(g))
        if p is None:
            missing += 1
            pred_events = []
        else:
            pred_events = as_event_list(p)

        t, a, e = compare_events(gold_events, pred_events)
        for i in range(3):
            tr[i] += t[i]
            ar[i] += a[i]
            ev[i] += e[i]

        gc = Counter(x["type"] for x in gold_events)
        pc = Counter(x["type"] for x in pred_events)
        for ty in set(gc) | set(pc):
            per_type_tp[ty] += min(gc[ty], pc[ty])
            per_type_gp[ty] += gc[ty]
            per_type_pp[ty] += pc[ty]

        gargs = Counter((x["type"], r, txt) for x in gold_events for (r, txt) in x["args"])
        pargs = Counter((x["type"], r, txt) for x in pred_events for (r, txt) in x["args"])
        for key in gargs:
            per_role_gp[key[1]] += gargs[key]
        for key in pargs:
            per_role_pp[key[1]] += pargs[key]
        for key in set(gargs) | set(pargs):
            per_role_tp[key[1]] += min(gargs[key], pargs[key])

    res = {
        "num_samples": len(gold_rows),
        "missing_predictions": missing,
        "trigger": prf(*tr),
        "argument": prf(*ar),
        "event_exact": prf(*ev),
    }

    per_type = {}
    for ty in sorted(per_type_gp):
        per_type[ty] = prf(per_type_tp[ty], per_type_pp[ty] - per_type_tp[ty],
                           per_type_gp[ty] - per_type_tp[ty])
    per_role = {}
    for role in sorted(per_role_gp):
        per_role[role] = prf(per_role_tp[role], per_role_pp[role] - per_role_tp[role],
                             per_role_gp[role] - per_role_tp[role])
    res["per_type"] = per_type
    res["per_role"] = per_role
    res["type_macro_f1"], _ = type_macro_f1(gold_rows, pred_rows)
    res["role_macro_f1"] = role_macro_f1(per_role)
    # 作业主对比指标：论元级 macro（按角色平均）+ 类型 macro
    res["macro_f1"] = res["role_macro_f1"]

    if verbose:
        print(f"样本数 {res['num_samples']}（缺预测 {missing} 条）")
        for k in ("trigger", "argument", "event_exact"):
            m = res[k]
            print(f"  {k:12s} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")
        print(f"  type_macro_f1 = {res['type_macro_f1']:.4f}")
        print(f"  role_macro_f1 = {res['role_macro_f1']:.4f}  (作业主对比 macro F1)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--json", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    res = evaluate(load_jsonl(args.gold), load_jsonl(args.pred), verbose=not args.quiet)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print("written:", args.json)


if __name__ == "__main__":
    main()
