# -*- coding: utf-8 -*-
"""
自建 Credit / Market Risk 报告的 Data Lineage 信息抽取语料。

网上无公开「风控报告 → lineage 结构化标注」集；本脚本按 BCBS 239 / FRTB / IFRS9
文档话术生成带字符级 span 的单事件句。val 故意混入 train 未见实体与噪声从句，
避免规则词典近满分、对比失真。

用法：
  python data/build_dataset.py
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCHEMA_PATH = ROOT / "schema.json"

# train 可见实体
OUTPUT_DATA_TR = [
    "EAD", "LGD", "PD", "Expected_Credit_Loss", "RWA_Credit", "CVA_charge",
    "EL_bucket", "NPE_ratio", "Stage2_ECL", "IRB_capital", "CCF_weighted_EAD",
    "VaR_99", "ES_97_5", "FRTB_SBM_charge", "Market_RWA", "IR_Delta_risk",
    "FX_Vega_bucket", "CSR_non_sec_Delta", "GIRR_Curvature", "Stressed_VaR",
]
INPUT_DATA_TR = [
    "loan_book_snapshot", "facility_limit", "drawn_balance", "undrawn_commitment",
    "collateral_haircut", "rating_grade", "days_past_due", "macro_scenario_table",
    "LGD_floor_table", "PD_term_structure", "yield_curve_EOD", "FX_spot_matrix",
    "implied_vol_surface", "CDS_spread_curve", "bond_price_feed", "equity_spot",
    "position_inventory", "risk_factor_sensitivities", "historical_PnL_vector",
]
METHODS_TR = [
    "calc_ead_irb", "compute_lgd_downturn", "estimate_pd_ttc", "run_ecl_ifrs9",
    "aggregate_rwa_credit", "apply_ccf_lookup", "map_stage_allocation",
    "calc_var_historical", "compute_es_frtb", "run_sbm_delta", "build_girr_bucket",
    "aggregate_imcc", "stress_var_window", "map_rf_to_bucket", "etl_risk_mart",
]
TRANSFORMS_TR = [
    "currency_conversion", "netting_set_aggregation", "maturity_ladder_binning",
    "haircut_adjustment", "scenario_shock_apply", "bucket_mapping",
    "correlation_matrix_multiply", "floor_cap_clip", "time_weighted_average",
]
REPORT_FIELDS_TR = [
    "COREP_C07_RWA", "FINREP_F18_NPE", "FRTB_IMA_IMCC", "FRTB_SA_SBM",
    "BCBS239_KDE_EAD", "ICAAP_credit_capital", "MR_daily_VaR_pack",
]
DATASETS_TR = [
    "crd_exposure_mart", "mkt_risk_factor_store", "pnl_attrib_cube",
    "counterparty_master", "frtb_sensitivity_store", "ecl_run_output",
]
SOURCE_SYSTEMS_TR = [
    "LOS_Core", "Collateral_DB", "Rating_Engine", "IFRS9_Engine",
    "Murex", "Bloomberg_BPIPE", "RiskWatch", "EDW_Risk",
]
TARGET_SYSTEMS_TR = [
    "Risk_Data_Mart", "FRTB_Cube", "BCBS239_Repository",
    "Regulatory_Reporting_Hub", "Market_Risk_Dashboard",
]

# val 专属未见实体（迫使模型泛化，削弱纯词典规则）
OUTPUT_DATA_VA = ["DFCVaR", "IMCC", "Default_Risk_Charge", "Aggregated_PnL", "SA_CCR_EAD"]
INPUT_DATA_VA = ["repo_rate_grid", "liquidity_horizon_map", "exposure_at_default_raw", "option_surface_otc"]
METHODS_VA = ["compute_cva_ba", "refresh_frtb_cube", "publish_bcbs239_kdes", "run_sa_ccr"]
TRANSFORMS_VA = ["look_through_lookback", "log_return_transform", "tenor_interpolation"]
REPORT_FIELDS_VA = ["TRIM_model_inventory_line", "Pillar3_market_risk_table", "STE_CVA_template"]
DATASETS_VA = ["instrument_ref_data", "liquidity_gap_ladder", "xva_engine_out"]
SOURCE_SYSTEMS_VA = ["Calypso", "Refinitiv_Elektron", "FRTB_Engine"]
TARGET_SYSTEMS_VA = ["ICAAP_Workbench", "Model_Inventory", "XVA_Store"]

RISK_DOMAINS = ["credit_risk", "market_risk", "counterparty_credit", "cva_risk", "liquidity_risk"]

TRIGGERS = {
    "DataDepends": ["depends on", "is dependent on", "relies on", "is driven by"],
    "DataProduces": ["produces", "outputs", "computes", "generates"],
    "ReportMaps": ["maps from", "is sourced from", "is populated from", "traces back to"],
    "SystemFlow": ["flows from", "is ingested from", "is loaded from", "is transferred from"],
    "TransformDerives": ["is derived via", "is transformed by", "is obtained through", "is calculated using"],
}

NOISE = [
    "Per the Q3 model inventory review, ",
    "After the TRIM remediation workshop, ",
    "According to the BCBS 239 KDE register, ",
    "In the attached lineage appendix, ",
    "As confirmed by the second line of defence, ",
    "",
    "",
]


def _span(text: str, frag: str) -> dict:
    pat = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(frag)}(?![A-Za-z0-9_])")
    m = pat.search(text)
    if m:
        return {"text": frag, "start": m.start(), "end": m.end()}
    start = text.index(frag)
    return {"text": frag, "start": start, "end": start + len(frag)}


def _pick(rng: random.Random, items):
    return rng.choice(items)


def _pool(tr, va, holdout_p, rng):
    """以 holdout_p 概率从 val 专属池取样。"""
    if rng.random() < holdout_p and va:
        return _pick(rng, va)
    return _pick(rng, tr)


def make_DataDepends(rng, holdout_p=0.0):
    out = _pool(OUTPUT_DATA_TR, OUTPUT_DATA_VA, holdout_p, rng)
    inp = _pool(INPUT_DATA_TR, INPUT_DATA_VA, holdout_p, rng)
    method = _pool(METHODS_TR, METHODS_VA, holdout_p, rng)
    domain = _pick(rng, RISK_DOMAINS)
    trig = _pick(rng, TRIGGERS["DataDepends"])
    noise = _pick(rng, NOISE)
    templates = [
        f"{noise}In the {domain} lineage note, {out} {trig} {inp} because method {method} takes {inp} as input and emits {out}.",
        f"{noise}For {domain} reporting, metric {out} {trig} dataset {inp}; the code path {method} reads {inp} and writes {out}.",
        f"{noise}BCBS 239 evidence for {domain}: {out} {trig} {inp} via {method}, whose output field is {out} and whose input feed is {inp}.",
        f"{noise}Under {domain}, {out} {trig} {inp}. The owning procedure is {method}.",
        f"{noise}{out} {trig} {inp} in {domain} scope; see procedure {method}.",
    ]
    text = _pick(rng, templates).strip()
    return {
        "text": text,
        "event": {
            "type": "DataDepends",
            "trigger": _span(text, trig),
            "args": [
                {"role": "OutputData", **_span(text, out)},
                {"role": "InputData", **_span(text, inp)},
                {"role": "Method", **_span(text, method)},
                {"role": "RiskDomain", **_span(text, domain)},
            ],
        },
    }


def make_DataProduces(rng, holdout_p=0.0):
    out = _pool(OUTPUT_DATA_TR, OUTPUT_DATA_VA, holdout_p, rng)
    inp = _pool(INPUT_DATA_TR, INPUT_DATA_VA, holdout_p, rng)
    method = _pool(METHODS_TR, METHODS_VA, holdout_p, rng)
    transform = _pool(TRANSFORMS_TR, TRANSFORMS_VA, holdout_p, rng)
    trig = _pick(rng, TRIGGERS["DataProduces"])
    noise = _pick(rng, NOISE)
    templates = [
        f"{noise}The routine {method} {trig} {out} from {inp} after applying {transform}.",
        f"{noise}In the ETL spec, {method} {trig} risk measure {out}; inputs include {inp} and the step {transform} is mandatory.",
        f"{noise}Pipeline note: {method} {trig} {out}. Upstream table {inp} is reshaped by {transform} before the write.",
        f"{noise}{method} {trig} {out}; feed={inp}; transform={transform}.",
        f"{noise}Risk engineering log: {method} {trig} {out} using {inp}, with intermediate {transform}.",
    ]
    text = _pick(rng, templates).strip()
    return {
        "text": text,
        "event": {
            "type": "DataProduces",
            "trigger": _span(text, trig),
            "args": [
                {"role": "OutputData", **_span(text, out)},
                {"role": "InputData", **_span(text, inp)},
                {"role": "Method", **_span(text, method)},
                {"role": "Transform", **_span(text, transform)},
            ],
        },
    }


def make_ReportMaps(rng, holdout_p=0.0):
    field = _pool(REPORT_FIELDS_TR, REPORT_FIELDS_VA, holdout_p, rng)
    src = _pool(OUTPUT_DATA_TR + DATASETS_TR, OUTPUT_DATA_VA + DATASETS_VA, holdout_p, rng)
    method = _pool(METHODS_TR, METHODS_VA, holdout_p, rng)
    domain = _pick(rng, RISK_DOMAINS)
    trig = _pick(rng, TRIGGERS["ReportMaps"])
    noise = _pick(rng, NOISE)
    templates = [
        f"{noise}Regulatory cell {field} {trig} {src} in the {domain} pack; assembler {method} fills {field} from {src}.",
        f"{noise}Lineage worksheet: {field} {trig} {src}. Method {method} copies {src} into {field} for {domain}.",
        f"{noise}Audit trail for {domain} shows {field} {trig} {src} because {method} binds report field {field} to source {src}.",
        f"{noise}For {domain} submission, {field} {trig} {src}; code {method} is the declared producer of {field}.",
        f"{noise}{field} {trig} {src} ({domain}); binder={method}.",
    ]
    text = _pick(rng, templates).strip()
    return {
        "text": text,
        "event": {
            "type": "ReportMaps",
            "trigger": _span(text, trig),
            "args": [
                {"role": "ReportField", **_span(text, field)},
                {"role": "SourceData", **_span(text, src)},
                {"role": "Method", **_span(text, method)},
                {"role": "RiskDomain", **_span(text, domain)},
            ],
        },
    }


def make_SystemFlow(rng, holdout_p=0.0):
    ds = _pool(DATASETS_TR, DATASETS_VA, holdout_p, rng)
    src = _pool(SOURCE_SYSTEMS_TR, SOURCE_SYSTEMS_VA, holdout_p, rng)
    tgt = _pool(TARGET_SYSTEMS_TR, TARGET_SYSTEMS_VA, holdout_p, rng)
    method = _pool(METHODS_TR, METHODS_VA, holdout_p, rng)
    trig = _pick(rng, TRIGGERS["SystemFlow"])
    noise = _pick(rng, NOISE)
    templates = [
        f"{noise}Dataset {ds} {trig} {src} into {tgt} by job {method}.",
        f"{noise}Interface catalogue: {ds} {trig} {src}; landing zone is {tgt} and the mover is {method}.",
        f"{noise}Ops runbook: {ds} {trig} {src} overnight, then {method} publishes {ds} to {tgt}.",
        f"{noise}System lineage: {ds} {trig} {src} and lands in {tgt}. Control point is method {method}.",
        f"{noise}{ds} {trig} {src} -> {tgt} via {method}.",
    ]
    text = _pick(rng, templates).strip()
    return {
        "text": text,
        "event": {
            "type": "SystemFlow",
            "trigger": _span(text, trig),
            "args": [
                {"role": "Dataset", **_span(text, ds)},
                {"role": "SourceSystem", **_span(text, src)},
                {"role": "TargetSystem", **_span(text, tgt)},
                {"role": "Method", **_span(text, method)},
            ],
        },
    }


def make_TransformDerives(rng, holdout_p=0.0):
    out = _pool(OUTPUT_DATA_TR, OUTPUT_DATA_VA, holdout_p, rng)
    inp = _pool(INPUT_DATA_TR, INPUT_DATA_VA, holdout_p, rng)
    transform = _pool(TRANSFORMS_TR, TRANSFORMS_VA, holdout_p, rng)
    method = _pool(METHODS_TR, METHODS_VA, holdout_p, rng)
    trig = _pick(rng, TRIGGERS["TransformDerives"])
    noise = _pick(rng, NOISE)
    templates = [
        f"{noise}Measure {out} {trig} {transform} on top of {inp}; orchestrator {method} schedules the step.",
        f"{noise}Derivation rule: {out} {trig} {transform}. Input series {inp} enters {method} before {out} is stored.",
        f"{noise}FRTB note says {out} {trig} {transform} applied to {inp}. Owner method is {method}.",
        f"{noise}Credit analytics: {out} {trig} {transform} using {inp}; implementation hook {method} owns the lineage edge.",
        f"{noise}{out} {trig} {transform}; base input {inp}; runner {method}.",
    ]
    text = _pick(rng, templates).strip()
    return {
        "text": text,
        "event": {
            "type": "TransformDerives",
            "trigger": _span(text, trig),
            "args": [
                {"role": "OutputData", **_span(text, out)},
                {"role": "InputData", **_span(text, inp)},
                {"role": "Transform", **_span(text, transform)},
                {"role": "Method", **_span(text, method)},
            ],
        },
    }


MAKERS = [
    make_DataDepends,
    make_DataProduces,
    make_ReportMaps,
    make_SystemFlow,
    make_TransformDerives,
]


def _validate(row: dict) -> None:
    text = row["text"]
    ev = row["event"]
    trig = ev["trigger"]
    assert text[trig["start"]:trig["end"]] == trig["text"], (text, trig)
    for a in ev["args"]:
        assert text[a["start"]:a["end"]] == a["text"], (text, a)


def build_split(n: int, rng: random.Random, prefix: str, holdout_p: float) -> list:
    rows = []
    for i in range(n):
        maker = MAKERS[i % len(MAKERS)]
        row = maker(rng, holdout_p=holdout_p)
        _validate(row)
        uid = f"{prefix}_{i:04d}"
        rows.append({
            "id": uid,
            "sent_id": uid,
            "uid": uid,
            "text": row["text"],
            "event": row["event"],
        })
    rng.shuffle(rows)
    return rows


def dump_jsonl(path: Path, rows: list) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=500)
    ap.add_argument("--val", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    train = build_split(args.train, rng, "tr", holdout_p=0.0)
    val = build_split(args.val, rng, "va", holdout_p=0.55)

    dump_jsonl(ROOT / "train.jsonl", train)
    dump_jsonl(ROOT / "val.jsonl", val)

    from collections import Counter
    stats = {
        "train": len(train),
        "val": len(val),
        "train_types": dict(Counter(r["event"]["type"] for r in train)),
        "val_types": dict(Counter(r["event"]["type"] for r in val)),
        "val_holdout_p": 0.55,
        "schema": json.loads(SCHEMA_PATH.read_text(encoding="utf-8")),
        "note": (
            "Synthetic credit/market risk data-lineage IE corpus. "
            "No public annotated lineage-from-risk-report dataset found. "
            "Val mixes held-out entity names so lexicon rules cannot saturate."
        ),
    }
    (ROOT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"wrote train={len(train)} val={len(val)} -> {ROOT}")
    print("train types:", stats["train_types"])
    print("val types:", stats["val_types"])


if __name__ == "__main__":
    main()
