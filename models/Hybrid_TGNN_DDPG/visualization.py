"""
Hybrid TGNN-DDPG 백테스팅 결과 시각화
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
    
    def __init__(self, save_dir='results/03_Hybrid_TGNN_DDPG'):
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
        리밸런싱 빈도별 성과 비교 그래프
        Args:
            각 dict는 {'dates': [], 'portfolio_values': [], 'metrics': {}} 형태
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2)
        
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        
        strategies = {
            '1/N Buy & Hold': buy_and_hold,
            'Hybrid (월간)': monthly,
            'Hybrid (분기)': quarterly,
            'Hybrid (반기)': semiannual,
            'Hybrid (연간)': annual
        }
        
        colors = ['#555555', '#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        # 1. 누적 수익률 비교
        all_returns = []
        for (name, data), color in zip(strategies.items(), colors):
            dates = pd.to_datetime(data['dates'])
            initial_val = data['portfolio_values'][0]
            returns = [(v/initial_val - 1)*100 for v in data['portfolio_values']]
            all_returns.extend(returns)
            
            final_cagr = data['metrics']['CAGR']
            label = f"{name} (CAGR: {final_cagr:.1f}%)"
            
            ax1.plot(dates, returns, label=label, linewidth=2.5, color=color)
            
        ax1.set_title('Hybrid Model: 리밸런싱 빈도별 수익률 비교 (2018-2025)', fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('누적 수익률 (%)', fontsize=12, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # 2. CAGR 비교
        cagr_values = [d['metrics']['CAGR'] for d in [buy_and_hold, monthly, quarterly, semiannual, annual]]
        labels = ['Buy&Hold', '월간', '분기', '반기', '연간']
        
        bars = ax2.bar(labels, cagr_values, color=colors, alpha=0.85, edgecolor='black', width=0.6)
        ax2.set_title("연평균 수익률 (CAGR)", fontsize=14, fontweight='bold')
        ax2.set_ylabel("수익률 (%)")
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, cagr_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                     f"{val:.1f}%", ha='center', va='bottom', fontweight='bold')
            
        # 3. MDD 비교
        mdd_values = [d['metrics']['MDD'] for d in [buy_and_hold, monthly, quarterly, semiannual, annual]]
        
        bars = ax3.bar(labels, mdd_values, color=colors, alpha=0.85, edgecolor='black', width=0.6)
        ax3.set_title("최대 낙폭 (MDD)", fontsize=14, fontweight='bold')
        ax3.set_ylabel("낙폭 (%)")
        ax3.grid(axis='y', alpha=0.3)
        ax3.invert_yaxis()
        
        for bar, val in zip(bars, mdd_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                     f"-{val:.1f}%", ha='center', va='top', fontweight='bold', color='red')
            
        plt.tight_layout()
        save_path = self.save_dir / 'hybrid_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 그래프 저장: {save_path}")
