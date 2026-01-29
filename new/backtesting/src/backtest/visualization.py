import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Any
from .metrics import compute_metrics


class Visualizer:
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.results_dir = results_dir
        self.plot_dir = os.path.join(results_dir, "plots")
        self.log_dir = os.path.join(results_dir, "logs")

        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        # Font settings for Korean support
        plt.rcParams["font.family"] = "Malgun Gothic"
        plt.rcParams["axes.unicode_minus"] = False

    def save_logs(self, results_map: Dict[str, Dict], symbols: List[str]):
        """
        Saves separate trade logs into a CSV.
        """
        all_logs = []
        for name, res in results_map.items():
            all_logs.extend(res["trade_logs"])

        df_log = pd.DataFrame(all_logs)
        if not df_log.empty:
            path = os.path.join(self.log_dir, "trade_logs.csv")
            # Ensure nice column ordering
            cols = ["Date", "Strategy", "Type"] + symbols
            # Filter cols that actually exist in logs
            existing_cols = [c for c in cols if c in df_log.columns]
            df_log = df_log[existing_cols]

            df_log.to_csv(path, index=False)
            print(f"✅ Trade logs saved to {path}")

    def plot_comparison(
        self, results_map: Dict[str, Dict], title: str = "Backtest Comparison"
    ):
        """
        Plots comprehensive comparison: Cumulative Returns, CAGR, and MDD.
        Layout:
          [ Cumulative Returns (Line) ]
          [ CAGR (Bar) ] [ MDD (Bar) ]
        """
        # Data Prep
        names = []
        cagrs = []
        mdds = []
        colors = [
            "#555555",
            "#2E86AB",
            "#A23B72",
            "#F18F01",
            "#C73E1D",
            "#63A375",
            "#000000",
        ]

        plt.figure(figsize=(16, 12))
        gs = matplotlib.gridspec.GridSpec(2, 2)

        ax1 = plt.subplot(gs[0, :])
        ax2 = plt.subplot(gs[1, 0])
        ax3 = plt.subplot(gs[1, 1])

        # 1. Cumulative Returns (Line)
        idx = 0
        for name, res in results_map.items():
            dates = pd.to_datetime(res["dates"])
            values = res["portfolio_values"]

            if not values:
                continue

            initial = values[0]
            returns = [(v / initial - 1) * 100 for v in values]

            # Metrics
            metrics = compute_metrics(values, initial_capital=initial)
            cagr = metrics["CAGR"]
            mdd = metrics["MDD"]

            names.append(name)
            cagrs.append(cagr)
            mdds.append(mdd)

            color = colors[idx % len(colors)]
            label = f"{name} (CAGR: {cagr:.1f}%)"

            ax1.plot(dates, returns, label=label, linewidth=2.5, color=color)
            idx += 1

        ax1.set_title(
            f"{title} (Cumulative Return)", fontsize=16, fontweight="bold", pad=20
        )
        ax1.set_ylabel("Cumulative Return (%)", fontsize=12, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.3)
        ax1.legend(loc="upper left", fontsize=11, framealpha=0.9)

        # 2. CAGR Comparison (Bar)
        bars = ax2.bar(
            names,
            cagrs,
            color=colors[: len(names)],
            alpha=0.85,
            edgecolor="black",
            width=0.6,
        )
        ax2.set_title("CAGR (%)", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Return (%)")
        ax2.grid(axis="y", alpha=0.3)
        ax2.tick_params(axis="x", rotation=45)

        for bar, val in zip(bars, cagrs):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 3. MDD Comparison (Bar)
        bars = ax3.bar(
            names,
            mdds,
            color=colors[: len(names)],
            alpha=0.85,
            edgecolor="black",
            width=0.6,
        )
        ax3.set_title("Max Drawdown (MDD)", fontsize=14, fontweight="bold")
        ax3.set_ylabel("Drawdown (%)")
        ax3.grid(axis="y", alpha=0.3)
        ax3.invert_yaxis()  # Depends on if MDD is positive or negative. Usually MDD is positive number in reports.
        # Check metrics.py: usually MDD is returned as positive percentage like 15.2.
        # Legacy code inverted axis, implying MDD values were positive.
        ax3.tick_params(axis="x", rotation=45)

        for bar, val in zip(bars, mdds):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"-{val:.1f}%",
                ha="center",
                va="top",
                fontweight="bold",
                color="red",
            )

        plt.tight_layout()
        plot_path = os.path.join(self.plot_dir, "comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"✅ Comparison plot saved to {plot_path}")
        plt.close()
