"""
논문용 그래프 생성 스크립트 (5-seed mean ± std 기반)
=======================================================
입력: results/multiseed/summary.csv
출력: academic_papers/images/ 에 논문 품질 그래프 저장

생성 그래프:
  fig_01_cagr_by_frequency.png  - 리밸런싱 주기별 CAGR 그룹 막대그래프 (에러바)
  fig_02_cagr_all.png           - 전체 전략 CAGR 수평 막대그래프 (에러바)
  fig_03_sharpe_all.png         - 전체 전략 Sharpe Ratio (에러바)
  fig_04_risk_return.png        - 리스크-수익 산점도 (에러바)
  fig_05_summary_4panel.png     - CAGR / Sharpe / MDD / Total Return 4분할
"""

import matplotlib
matplotlib.use("Agg")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKTESTING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKTESTING_DIR, ".."))

SUMMARY_CSV = os.path.join(BACKTESTING_DIR, "results", "multiseed", "summary.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "academic_papers", "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 스타일 설정 (Elsevier/IEEE 스타일) ────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "legend.fontsize": 10,
    "legend.framealpha": 0.9,
    "errorbar.capsize": 4,
})

# ── 색상 팔레트 ────────────────────────────────────────────────────────────────
COLORS = {
    "Benchmark": "#6B6B6B",
    "TGNN":      "#2196F3",   # Blue
    "DDPG":      "#FF9800",   # Orange
    "HYBRID":    "#4CAF50",   # Green
}

FREQ_ORDER = ["Monthly", "Quarterly", "Semiannual", "Annual"]
FREQ_LABELS = {"Monthly": "Monthly", "Quarterly": "Quarterly",
               "Semiannual": "Semi-annual", "Annual": "Annual"}

# ── 데이터 로드 ────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY_CSV)
    df = df[df["Strategy"] != "Benchmark"].copy()

    # Model / Frequency 파싱
    def parse_strategy(s: str):
        parts = s.split("_", 1)
        return parts[0].upper(), parts[1] if len(parts) > 1 else ""

    df[["Model", "Frequency"]] = df["Strategy"].apply(
        lambda s: pd.Series(parse_strategy(s))
    )
    df = df[df["Frequency"].isin(FREQ_ORDER)].copy()
    df["Frequency"] = pd.Categorical(df["Frequency"], categories=FREQ_ORDER, ordered=True)
    return df.sort_values(["Model", "Frequency"]).reset_index(drop=True)


def load_benchmark() -> dict:
    raw = pd.read_csv(SUMMARY_CSV)
    row = raw[raw["Strategy"] == "Benchmark"].iloc[0]
    return {
        "CAGR": float(row["CAGR (%) Mean"]),
        "Sharpe": float(row["Sharpe Ratio Mean"]),
        "MDD": float(row["MDD (%) Mean"]),
        "Return": float(row["Total Return (%) Mean"]),
    }


print("[*] Loading data...")
df = load_data()
bench = load_benchmark()
print(f"  OK: {len(df)} strategies loaded")
print(f"  Benchmark CAGR: {bench['CAGR']:.2f}%")

MODELS = ["TGNN", "DDPG", "HYBRID"]


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 01: 리밸런싱 주기별 CAGR 그룹 막대그래프 (에러바 포함) — 핵심 그림
# ═══════════════════════════════════════════════════════════════════════════════
def fig01_cagr_by_frequency():
    fig, ax = plt.subplots(figsize=(10, 6))

    n_freq = len(FREQ_ORDER)
    n_model = len(MODELS)
    group_width = 0.7
    bar_width = group_width / n_model
    x = np.arange(n_freq)

    for i, model in enumerate(MODELS):
        sub = df[df["Model"] == model].set_index("Frequency")
        means = [sub.loc[f, "CAGR (%) Mean"] if f in sub.index else np.nan for f in FREQ_ORDER]
        stds  = [sub.loc[f, "CAGR (%) Std"]  if f in sub.index else 0.0       for f in FREQ_ORDER]
        offset = (i - n_model / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, means, bar_width * 0.9,
            label=model, color=COLORS[model], alpha=0.85,
            edgecolor="white", linewidth=0.8,
        )
        ax.errorbar(
            x + offset, means, yerr=stds,
            fmt="none", color="black", linewidth=1.2,
            capsize=4, capthick=1.2,
        )
        # 값 라벨
        for xpos, mean in zip(x + offset, means):
            if not np.isnan(mean):
                va = "bottom" if mean >= 0 else "top"
                dy = 0.15 if mean >= 0 else -0.15
                ax.text(xpos, mean + dy, f"{mean:.1f}",
                        ha="center", va=va, fontsize=7.5, fontweight="bold")

    # Benchmark line
    ax.axhline(bench["CAGR"], color=COLORS["Benchmark"], linestyle="--",
               linewidth=1.5, label=f"Benchmark ({bench['CAGR']:.1f}%)")
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([FREQ_LABELS[f] for f in FREQ_ORDER])
    ax.set_xlabel("Rebalancing Frequency")
    ax.set_ylabel("CAGR (%)")
    ax.set_title("CAGR by Rebalancing Frequency\n(5-Seed Mean ± Std, 2015–2024)", pad=10)
    ax.legend(loc="upper left")

    out = os.path.join(OUTPUT_DIR, "fig_01_cagr_by_frequency.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 02: 전체 전략 CAGR 수평 막대그래프 (에러바)
# ═══════════════════════════════════════════════════════════════════════════════
def fig02_cagr_all():
    plot_df = df.sort_values("CAGR (%) Mean", ascending=True).reset_index(drop=True)
    colors_list = [COLORS[m] for m in plot_df["Model"]]

    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(plot_df))

    ax.barh(y, plot_df["CAGR (%) Mean"], xerr=plot_df["CAGR (%) Std"],
            color=colors_list, alpha=0.85, edgecolor="white", linewidth=0.8,
            error_kw={"elinewidth": 1.2, "capsize": 4, "ecolor": "black"})

    ax.axvline(bench["CAGR"], color=COLORS["Benchmark"], linestyle="--",
               linewidth=1.5, label=f"Benchmark ({bench['CAGR']:.1f}%)")
    ax.axvline(0, color="gray", linestyle="-", linewidth=0.6, alpha=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["Strategy"], fontsize=9)
    ax.set_xlabel("CAGR (%)")
    ax.set_title("CAGR Comparison — All Strategies\n(5-Seed Mean ± Std)", pad=10)

    legend_patches = [
        mpatches.Patch(color=COLORS["TGNN"], label="TGNN"),
        mpatches.Patch(color=COLORS["DDPG"], label="DDPG"),
        mpatches.Patch(color=COLORS["HYBRID"], label="Hybrid"),
        Line2D([0], [0], color=COLORS["Benchmark"], linestyle="--",
               label=f"Benchmark ({bench['CAGR']:.1f}%)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right")

    out = os.path.join(OUTPUT_DIR, "fig_02_cagr_all.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 03: Sharpe Ratio 수평 막대그래프 (에러바)
# ═══════════════════════════════════════════════════════════════════════════════
def fig03_sharpe_all():
    plot_df = df.sort_values("Sharpe Ratio Mean", ascending=True).reset_index(drop=True)
    colors_list = [COLORS[m] for m in plot_df["Model"]]

    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(plot_df))

    ax.barh(y, plot_df["Sharpe Ratio Mean"], xerr=plot_df["Sharpe Ratio Std"],
            color=colors_list, alpha=0.85, edgecolor="white", linewidth=0.8,
            error_kw={"elinewidth": 1.2, "capsize": 4, "ecolor": "black"})

    ax.axvline(bench["Sharpe"], color=COLORS["Benchmark"], linestyle="--",
               linewidth=1.5, label=f"Benchmark ({bench['Sharpe']:.2f})")
    ax.axvline(0, color="gray", linestyle="-", linewidth=0.6, alpha=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["Strategy"], fontsize=9)
    ax.set_xlabel("Sharpe Ratio")
    ax.set_title("Sharpe Ratio Comparison — All Strategies\n(5-Seed Mean ± Std)", pad=10)

    legend_patches = [
        mpatches.Patch(color=COLORS["TGNN"], label="TGNN"),
        mpatches.Patch(color=COLORS["DDPG"], label="DDPG"),
        mpatches.Patch(color=COLORS["HYBRID"], label="Hybrid"),
        Line2D([0], [0], color=COLORS["Benchmark"], linestyle="--",
               label=f"Benchmark ({bench['Sharpe']:.2f})"),
    ]
    ax.legend(handles=legend_patches, loc="lower right")

    out = os.path.join(OUTPUT_DIR, "fig_03_sharpe_all.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 04: 리스크-수익 산점도 (MDD vs CAGR, 에러바 포함)
# ═══════════════════════════════════════════════════════════════════════════════
def fig04_risk_return():
    fig, ax = plt.subplots(figsize=(9, 7))

    for model in MODELS:
        sub = df[df["Model"] == model]
        ax.errorbar(
            sub["MDD (%) Mean"], sub["CAGR (%) Mean"],
            xerr=sub["MDD (%) Std"], yerr=sub["CAGR (%) Std"],
            fmt="o", color=COLORS[model], label=model,
            markersize=8, linewidth=1.0,
            capsize=3, capthick=1, alpha=0.85,
            markeredgecolor="white", markeredgewidth=0.8,
        )
        # 전략명 라벨
        for _, row in sub.iterrows():
            freq = row["Frequency"]
            short = {"Monthly": "M", "Quarterly": "Q", "Semiannual": "SA", "Annual": "A"}
            ax.annotate(
                f"{model[0]}-{short.get(freq, freq)}",
                (row["MDD (%) Mean"], row["CAGR (%) Mean"]),
                xytext=(6, 3), textcoords="offset points",
                fontsize=8, color=COLORS[model],
            )

    # Benchmark
    ax.scatter(bench["MDD"], bench["CAGR"], s=200, c=COLORS["Benchmark"],
               marker="D", zorder=5, label="Benchmark", edgecolors="white", linewidth=1.5)
    ax.annotate("Benchmark", (bench["MDD"], bench["CAGR"]),
                xytext=(6, 3), textcoords="offset points", fontsize=9, fontweight="bold")

    # 기준선
    ax.axhline(bench["CAGR"], color=COLORS["Benchmark"], linestyle="--",
               linewidth=1, alpha=0.5)
    ax.axvline(bench["MDD"], color=COLORS["Benchmark"], linestyle="--",
               linewidth=1, alpha=0.5)

    ax.set_xlabel("Maximum Drawdown (%) →  Higher = More Risk")
    ax.set_ylabel("CAGR (%)")
    ax.set_title("Risk–Return Profile\n(5-Seed Mean ± Std; Upper-Left = Better)", pad=10)

    # 사분면 라벨
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    ax.text(xlim[0] + 0.5, ylim[1] - 1.5, "Low Risk\nHigh Return",
            fontsize=8, color="green", alpha=0.6)
    ax.text(xlim[1] - 8, ylim[0] + 0.5, "High Risk\nLow Return",
            fontsize=8, color="red", alpha=0.6)

    ax.legend(loc="upper left")
    out = os.path.join(OUTPUT_DIR, "fig_04_risk_return.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 05: 4분할 종합 요약 (CAGR / Sharpe / MDD / Total Return)
# ═══════════════════════════════════════════════════════════════════════════════
def fig05_summary_4panel():
    metrics = [
        ("CAGR (%) Mean",         "CAGR (%) Std",         "CAGR (%)",          bench["CAGR"]),
        ("Sharpe Ratio Mean",      "Sharpe Ratio Std",     "Sharpe Ratio",      bench["Sharpe"]),
        ("MDD (%) Mean",           "MDD (%) Std",          "MDD (%)",           bench["MDD"]),
        ("Total Return (%) Mean",  "Total Return (%) Std", "Total Return (%)",  bench["Return"]),
    ]
    titles = ["(a) CAGR", "(b) Sharpe Ratio", "(c) Max Drawdown", "(d) Total Return"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    n_freq = len(FREQ_ORDER)
    n_model = len(MODELS)
    group_width = 0.72
    bar_width = group_width / n_model
    x = np.arange(n_freq)

    for ax, (mean_col, std_col, ylabel, bench_val), title in zip(axes, metrics, titles):
        for i, model in enumerate(MODELS):
            sub = df[df["Model"] == model].set_index("Frequency")
            means = [sub.loc[f, mean_col] if f in sub.index else np.nan for f in FREQ_ORDER]
            stds  = [sub.loc[f, std_col]  if f in sub.index else 0.0   for f in FREQ_ORDER]
            offset = (i - n_model / 2 + 0.5) * bar_width
            ax.bar(x + offset, means, bar_width * 0.9,
                   label=model, color=COLORS[model], alpha=0.85,
                   edgecolor="white", linewidth=0.7)
            ax.errorbar(x + offset, means, yerr=stds,
                        fmt="none", color="black", linewidth=1.0, capsize=3, capthick=1.0)

        ax.axhline(bench_val, color=COLORS["Benchmark"], linestyle="--",
                   linewidth=1.2, alpha=0.8)
        ax.axhline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([FREQ_LABELS[f] for f in FREQ_ORDER], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")

    # 공통 범례 (하단)
    legend_patches = [mpatches.Patch(color=COLORS[m], label=m) for m in MODELS]
    legend_patches.append(
        Line2D([0], [0], color=COLORS["Benchmark"], linestyle="--", label="Benchmark")
    )
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), fontsize=10)

    fig.suptitle("Performance Summary by Rebalancing Frequency\n(5-Seed Mean ± Std, 2015–2024)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    out = os.path.join(OUTPUT_DIR, "fig_05_summary_4panel.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 06: Hybrid vs TGNN 직접 비교 (주기별) — 핵심 기여 강조
# ═══════════════════════════════════════════════════════════════════════════════
def fig06_hybrid_vs_tgnn():
    """HYBRID와 TGNN 직접 비교 — 앙상블 효과 강조용"""
    fig, ax = plt.subplots(figsize=(9, 6))

    bar_width = 0.28
    x = np.arange(n_freq := len(FREQ_ORDER))

    for i, model in enumerate(["TGNN", "HYBRID"]):
        sub = df[df["Model"] == model].set_index("Frequency")
        means = [sub.loc[f, "CAGR (%) Mean"] if f in sub.index else np.nan for f in FREQ_ORDER]
        stds  = [sub.loc[f, "CAGR (%) Std"]  if f in sub.index else 0.0   for f in FREQ_ORDER]
        offset = (i - 0.5) * bar_width
        bars = ax.bar(x + offset, means, bar_width * 0.92,
                      label=model, color=COLORS[model], alpha=0.85,
                      edgecolor="white", linewidth=0.8)
        ax.errorbar(x + offset, means, yerr=stds,
                    fmt="none", color="black", linewidth=1.2, capsize=4, capthick=1.2)
        for xpos, mean in zip(x + offset, means):
            if not np.isnan(mean):
                va = "bottom" if mean >= 0 else "top"
                dy = 0.2 if mean >= 0 else -0.3
                ax.text(xpos, mean + dy, f"{mean:.1f}",
                        ha="center", va=va, fontsize=8.5, fontweight="bold")

    ax.axhline(bench["CAGR"], color=COLORS["Benchmark"], linestyle="--",
               linewidth=1.5, label=f"Benchmark ({bench['CAGR']:.1f}%)")
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([FREQ_LABELS[f] for f in FREQ_ORDER])
    ax.set_xlabel("Rebalancing Frequency")
    ax.set_ylabel("CAGR (%)")
    ax.set_title("Hybrid vs. TGNN: CAGR Comparison\n(5-Seed Mean ± Std)", pad=10)
    ax.legend()

    out = os.path.join(OUTPUT_DIR, "fig_06_hybrid_vs_tgnn.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out)}")


# ── 실행 ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n[*] Generating paper figures...\n")
    fig01_cagr_by_frequency()
    fig02_cagr_all()
    fig03_sharpe_all()
    fig04_risk_return()
    fig05_summary_4panel()
    fig06_hybrid_vs_tgnn()

    print(f"\nAll figures saved to:\n   {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.startswith("fig_"):
            size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) // 1024
            print(f"  [fig] {f}  ({size_kb} KB)")
