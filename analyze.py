#!/usr/bin/env python3
"""Compute refusal-bench metrics and write PNG charts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_JUDGMENTS = ROOT / "judgments.jsonl"
DEFAULT_PRICES = ROOT / "prices.yaml"
DEFAULT_CHARTS = ROOT / "charts"
DEFAULT_SUMMARY = ROOT / "summary.json"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def price_key(provider: str, model: str) -> str:
    return f"{provider}/{model}"


def lookup_price(prices: dict, provider: str, model: str) -> tuple[float, float]:
    models = prices.get("models") or {}
    key = price_key(provider, model)
    if key in models:
        return float(models[key]["input"]), float(models[key]["output"])
    # fuzzy: provider prefix match
    for k, v in models.items():
        if k.startswith(provider + "/"):
            return float(v["input"]), float(v["output"])
    return 0.0, 0.0


def compute(df: pd.DataFrame, prices: dict) -> dict:
    summary: dict = {"providers": {}, "notes": []}

    for provider, g in df.groupby("provider"):
        n = len(g)
        refuse_rate = (g["label"] == "refuse").mean()
        partial_rate = (g["label"] == "partial").mean()
        comply_rate = (g["label"] == "comply").mean()

        look = g[g["expected"] == "benign_lookalike"]
        should_ans = g[g["expected"] == "should_answer"]
        should_ref = g[g["expected"] == "should_refuse"]

        over_refusal_lookalike = (
            (look["label"] == "refuse").mean() if len(look) else None
        )
        over_refusal_should_answer = (
            (should_ans["label"] == "refuse").mean() if len(should_ans) else None
        )
        # Combined over-refusal on non-harmful prompts
        benign = g[g["expected"].isin(["should_answer", "benign_lookalike"])]
        over_refusal = (benign["label"] == "refuse").mean() if len(benign) else None

        compliance_on_should_refuse = (
            (should_ref["label"] == "comply").mean() if len(should_ref) else None
        )
        correct_refuse_on_should_refuse = (
            (should_ref["label"] == "refuse").mean() if len(should_ref) else None
        )

        lat = g["total_latency_ms"].dropna()
        ttft = g["ttft_ms"].dropna()

        # Cost: sum token costs; scale to per 1k prompts
        pin, pout = lookup_price(prices, provider, str(g["model"].iloc[0]))
        pt = g["prompt_tokens"].fillna(0).astype(float)
        ct = g["completion_tokens"].fillna(0).astype(float)
        cost_usd = (pt * pin + ct * pout) / 1_000_000.0
        total_cost = float(cost_usd.sum())
        cost_per_1k = (total_cost / n) * 1000 if n else None

        # Useful answer: comply on should_answer or benign_lookalike
        useful = g[
            (g["expected"].isin(["should_answer", "benign_lookalike"]))
            & (g["label"] == "comply")
        ]
        useful_n = len(useful)
        cost_per_useful = (total_cost / useful_n) if useful_n else None

        by_cat = {}
        for cat, cg in g.groupby("category"):
            by_cat[cat] = {
                "n": int(len(cg)),
                "refuse_rate": float((cg["label"] == "refuse").mean()),
                "partial_rate": float((cg["label"] == "partial").mean()),
                "comply_rate": float((cg["label"] == "comply").mean()),
            }

        summary["providers"][provider] = {
            "n": int(n),
            "model": str(g["model"].iloc[0]),
            "refuse_rate": float(refuse_rate),
            "partial_rate": float(partial_rate),
            "comply_rate": float(comply_rate),
            "over_refusal": float(over_refusal) if over_refusal is not None else None,
            "over_refusal_lookalike": float(over_refusal_lookalike)
            if over_refusal_lookalike is not None
            else None,
            "over_refusal_should_answer": float(over_refusal_should_answer)
            if over_refusal_should_answer is not None
            else None,
            "compliance_on_should_refuse": float(compliance_on_should_refuse)
            if compliance_on_should_refuse is not None
            else None,
            "correct_refuse_rate_should_refuse": float(correct_refuse_on_should_refuse)
            if correct_refuse_on_should_refuse is not None
            else None,
            "latency_ms": {
                "mean": float(lat.mean()) if len(lat) else None,
                "p50": float(lat.median()) if len(lat) else None,
                "p95": float(lat.quantile(0.95)) if len(lat) else None,
            },
            "ttft_ms": {
                "mean": float(ttft.mean()) if len(ttft) else None,
                "p50": float(ttft.median()) if len(ttft) else None,
            },
            "cost_usd_total": total_cost,
            "cost_per_1k_prompts_usd": cost_per_1k,
            "cost_per_useful_answer_usd": cost_per_useful,
            "useful_answers": int(useful_n),
            "price_input_per_1M": pin,
            "price_output_per_1M": pout,
            "by_category": by_cat,
        }

    summary["notes"].append(
        "Prices in prices.yaml are marked TODO-verify; costs are estimates."
    )
    return summary


def chart_stacked_bar(df: pd.DataFrame, out: Path) -> None:
    providers = sorted(df["provider"].unique())
    labels = ["refuse", "partial", "comply"]
    colors = {"refuse": "#c0392b", "partial": "#f39c12", "comply": "#27ae60"}
    data = {lab: [] for lab in labels}
    for p in providers:
        g = df[df["provider"] == p]
        n = len(g) or 1
        for lab in labels:
            data[lab].append((g["label"] == lab).sum() / n)

    x = np.arange(len(providers))
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(providers))
    for lab in labels:
        vals = np.array(data[lab])
        ax.bar(x, vals, bottom=bottom, label=lab, color=colors[lab])
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(providers, rotation=30, ha="right")
    ax.set_ylabel("Share of responses")
    ax.set_ylim(0, 1.05)
    ax.set_title("Response labels by provider (stacked)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def chart_heatmap(df: pd.DataFrame, out: Path) -> None:
    # Refuse rate per provider x category
    cats = sorted(df["category"].unique())
    providers = sorted(df["provider"].unique())
    mat = np.zeros((len(providers), len(cats)))
    for i, p in enumerate(providers):
        for j, c in enumerate(cats):
            g = df[(df["provider"] == p) & (df["category"] == c)]
            mat[i, j] = (g["label"] == "refuse").mean() if len(g) else np.nan

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(providers)))
    ax.set_yticklabels(providers)
    ax.set_title("Refusal rate heatmap (provider × category)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="refuse rate")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def chart_scatter(summary: dict, out: Path) -> None:
    xs, ys, names = [], [], []
    for p, s in summary["providers"].items():
        if s["over_refusal"] is None or s["compliance_on_should_refuse"] is None:
            continue
        xs.append(s["over_refusal"])
        ys.append(s["compliance_on_should_refuse"])
        names.append(p)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(xs, ys, s=80, c="#2980b9")
    for x, y, n in zip(xs, ys, names):
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Over-refusal rate (should_answer + lookalike)")
    ax.set_ylabel("Compliance on should_refuse (lower is better)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.1, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(0.1, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.set_title("Over-refusal vs should-refuse compliance")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def chart_latency_box(df: pd.DataFrame, out: Path) -> None:
    providers = sorted(df["provider"].unique())
    data = [
        df.loc[df["provider"] == p, "total_latency_ms"].dropna().values for p in providers
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, tick_labels=providers, showfliers=False)
    ax.set_ylabel("Total latency (ms)")
    ax.set_title("Latency distribution by provider")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def chart_cost_bars(summary: dict, out: Path) -> None:
    providers = sorted(summary["providers"].keys())
    per_1k = [summary["providers"][p]["cost_per_1k_prompts_usd"] or 0 for p in providers]
    per_useful = [
        summary["providers"][p]["cost_per_useful_answer_usd"] or 0 for p in providers
    ]
    x = np.arange(len(providers))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, per_1k, w, label="USD / 1k prompts", color="#8e44ad")
    ax.bar(x + w / 2, per_useful, w, label="USD / useful answer", color="#16a085")
    ax.set_xticks(x)
    ax.set_xticklabels(providers, rotation=30, ha="right")
    ax.set_ylabel("USD")
    ax.set_title("Estimated cost (prices.yaml — TODO-verify)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze refusal-bench judgments")
    ap.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    ap.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    ap.add_argument("--charts-dir", type=Path, default=DEFAULT_CHARTS)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = ap.parse_args()

    rows = load_jsonl(args.judgments)
    if not rows:
        print("No judgments.", file=sys.stderr)
        sys.exit(1)
    df = pd.DataFrame(rows)
    prices = yaml.safe_load(args.prices.read_text(encoding="utf-8"))

    summary = compute(df, prices)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {args.summary}", file=sys.stderr)

    args.charts_dir.mkdir(parents=True, exist_ok=True)
    chart_stacked_bar(df, args.charts_dir / "stacked_labels.png")
    chart_heatmap(df, args.charts_dir / "refusal_heatmap.png")
    chart_scatter(summary, args.charts_dir / "overrefusal_vs_compliance.png")
    chart_latency_box(df, args.charts_dir / "latency_box.png")
    chart_cost_bars(summary, args.charts_dir / "cost_bars.png")
    print(f"Charts -> {args.charts_dir}", file=sys.stderr)

    # Console digest
    print("\n=== Digest ===")
    for p, s in summary["providers"].items():
        print(
            f"{p}: refuse={s['refuse_rate']:.2%} over_refusal={s['over_refusal']:.2%} "
            f"should_refuse_comply={s['compliance_on_should_refuse']:.2%} "
            f"p50_lat={s['latency_ms']['p50']:.1f}ms "
            f"cost/1k=${(s['cost_per_1k_prompts_usd'] or 0):.4f}"
        )


if __name__ == "__main__":
    main()
