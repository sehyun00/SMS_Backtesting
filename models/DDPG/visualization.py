"""
DDPG 백테스팅 결과 시각화
리밸런싱 빈도별 성과 비교
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False


class BacktestVisualizer:
    """백테스팅 결과 시각화 클래스"""
    
    def __init__(self, save_dir='results/02_DDPG_Only'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_rebalancing_comparison(
        self,
        buy_and_hold: dict,
        monthly: dict,
        quarterly: dict,
        semiannual: dict,
        annual: dict
    ):
        """
        리밸런싱 빈도별 누적 수익률 비교 그래프
        
        Args:
            각 dict는 {'dates': [], 'portfolio_values': []} 형태
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2)
        
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        
        strategies = {
            '1/N 매수 후 보유': buy_and_hold,
            'DDPG (월간)': monthly,
            'DDPG (분기)': quarterly,
            'DDPG (반기)': semiannual,
            'DDPG (연간)': annual
        }
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
        
        # 1. 누적 수익률 비교
        all_returns = []
        
        for (strategy_name, data), color in zip(strategies.items(), colors):
            dates = pd.to_datetime(data['dates'])
            returns = (np.array(data['portfolio_values']) / data['portfolio_values'][0] - 1) * 100
            all_returns.extend(returns)
            ax1.plot(dates, returns, label=strategy_name, linewidth=2.5, color=color)
        
        ax1.set_xlabel('연도', fontsize=12, fontweight='bold')
        ax1.set_ylabel('누적 수익률 (%)', fontsize=12, fontweight='bold')
        ax1.set_title('리밸런싱 빈도별 누적 수익률 비교 (DDPG)', 
                     fontsize=16, fontweight='bold', pad=20)
        ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
        
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        
        y_min, y_max = min(all_returns), max(all_returns)
        ax1.set_ylim(y_min - 10, y_max * 1.1)
        
        # 2. 연평균 수익률 (CAGR) 비교
        cagr_values = []
        labels = ['Buy&Hold', '월간', '분기', '반기', '연간']
        
        for data in [buy_and_hold, monthly, quarterly, semiannual, annual]:
            dates_list = pd.to_datetime(data['dates'])
            days = (dates_list.max() - dates_list.min()).days
            years = days / 365.25
            if years > 0:
                final_value = data['portfolio_values'][-1]
                initial_value = data['portfolio_values'][0]
                cagr = (pow(final_value / initial_value, 1/years) - 1) * 100
            else:
                cagr = 0
            cagr_values.append(cagr)
        
        bars = ax2.bar(range(5), cagr_values, color=colors, alpha=0.85, edgecolor='black', width=0.6)
        ax2.set_xticks(range(5))
        ax2.set_xticklabels(labels, fontsize=10)
        ax2.set_ylabel('연평균 수익률 (%)', fontsize=11, fontweight='bold')
        ax2.set_title('CAGR 비교', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, value in zip(bars, cagr_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 3. 최대 낙폭 (MDD) 비교
        mdd_values = []
        for data in [buy_and_hold, monthly, quarterly, semiannual, annual]:
            portfolio = np.array(data['portfolio_values'])
            running_max = np.maximum.accumulate(portfolio)
            drawdown = (portfolio - running_max) / running_max * 100
            mdd = drawdown.min()
            mdd_values.append(abs(mdd))
        
        bars = ax3.bar(range(5), mdd_values, color=colors, alpha=0.85, edgecolor='black', width=0.6)
        ax3.set_xticks(range(5))
        ax3.set_xticklabels(labels, fontsize=10)
        ax3.set_ylabel('최대 낙폭 (%)', fontsize=11, fontweight='bold')
        ax3.set_title('Maximum Drawdown 비교 (낮을수록 좋음)', fontsize=14, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        for bar, value in zip(bars, mdd_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'-{value:.1f}%', ha='center', va='bottom', fontsize=10, 
                    fontweight='bold', color='#D32F2F')
        
        plt.tight_layout()
        
        save_path = self.save_dir / 'rebalancing_frequency_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 그래프 저장: {save_path}")
        
        plt.show()
        
    def plot_performance_metrics(self, metrics_df: pd.DataFrame):
        """
        성과 지표 히트맵
        
        Args:
            metrics_df: 전략별 성과지표 DataFrame
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 정규화 (0-1)
        normalized = metrics_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        
        sns.heatmap(
            normalized.T, 
            annot=metrics_df.T, 
            fmt='.2f',
            cmap='RdYlGn',
            cbar_kws={'label': '정규화 점수'},
            ax=ax,
            linewidths=0.5
        )
        
        ax.set_title('전략별 성과 지표 비교 (DDPG)', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('전략', fontsize=12)
        ax.set_ylabel('지표', fontsize=12)
        
        plt.tight_layout()
        save_path = self.save_dir / 'performance_heatmap.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 히트맵 저장: {save_path}")
        
        plt.show()
