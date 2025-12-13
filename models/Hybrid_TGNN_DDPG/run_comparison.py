"""
Hybrid TGNN-DDPG 학습 및 성과 비교 스크립트
- TGNN의 그래프 생성 로직과 DDPG의 강화학습 로직을 결합하여 학습 및 테스트를 수행합니다.
- 학습 기간: ~2017년 (3년)
- 테스트 기간: 2018년~2025년 (8년)
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from model import HybridAgent
from visualization import BacktestVisualizer

# 프로젝트 루트 및 데이터 경로 설정
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_PATH = (
    ROOT_DIR / "data" / "processed_daily_5factor_model_10stocks_10years_20251127.csv"
)

# 한글 폰트 설정 (그래프 출력용)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ============ Hybrid 데이터셋 (그래프 구조 + 윈도우 데이터) ============

class HybridDataset:
    """
    Hybrid 모델을 위한 데이터셋 클래스
    - 시계열 윈도우 데이터와 종목 간 관계 그래프(Adjacency Matrix)를 생성하여 제공합니다.
    """
    def __init__(self, df, window_size=12, feature_cols=None):
        self.df = df.copy()
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.window_size = window_size
        self.feature_cols = feature_cols
        self.symbols = sorted(df["Symbol"].unique())
        
        # 1. 월별 데이터로 변환 (Resampling)
        # - Daily 데이터를 월말 기준으로 합칩니다.
        self.monthly_df = (
            self.df.set_index("Date")
            .groupby(["Symbol", pd.Grouper(freq="ME")])
            .last()
            .reset_index()
        )
        # 목표 수익률로 Momentum1M 사용 (필요시 교체 가능)
        self.monthly_df["Return_Raw"] = self.monthly_df["Momentum1M"].copy()
        
        # 2. 학습/테스트 데이터 분할 기준일 (2017년 12월 31일)
        self.split_date = pd.Timestamp("2017-12-31")
        
        # 3. 스케일러 학습 (Train 데이터 기준)
        # - Test 데이터의 정보 유출(Look-ahead Bias) 방지
        self._fit_scaler_on_train_data()
        
        # 4. 윈도우 데이터 생성 (그래프 포함)
        self.windows = self._create_windows()
        
        # 5. 테스트 시작 인덱스 찾기
        self.test_start_idx = 0
        for i, w in enumerate(self.windows):
            if w["date"] > self.split_date:
                self.test_start_idx = i
                break
                
        print(f"   📊 Hybrid 데이터셋 생성 완료: 총 {len(self.windows)}개월")
        print(f"   📈 학습 데이터: {self.test_start_idx}개월 (~2017)")
        print(f"   📉 테스트 데이터: {len(self.windows) - self.test_start_idx}개월 (2018~)")

    def _fit_scaler_on_train_data(self):
        """학습 데이터에 대해서만 스케일러(StandardScaler)를 학습합니다."""
        train_data = self.monthly_df[self.monthly_df["Date"] <= self.split_date]
        self.scaler = StandardScaler()
        self.scaler.fit(train_data[self.feature_cols].values)
        
        # 전체 데이터 변환
        self.monthly_df[self.feature_cols] = self.scaler.transform(
            self.monthly_df[self.feature_cols].values
        )

    def _create_graph(self, snapshot_df):
        """
        TGNN 로직: 상관계수(Correlation) * 산업 유사도(Industry Similarity)
        - 두 종목 간의 관계를 정의하여 인접 행렬을 생성합니다.
        """
        n = len(self.symbols)
        corr_matrix = np.eye(n)
        
        # 1. 상관계수 계산
        for i, sym1 in enumerate(self.symbols):
            data1 = snapshot_df[snapshot_df["Symbol"] == sym1][self.feature_cols].values.flatten()
            for j, sym2 in enumerate(self.symbols):
                if i >= j: continue
                data2 = snapshot_df[snapshot_df["Symbol"] == sym2][self.feature_cols].values.flatten()
                
                if len(data1) > 0 and len(data2) > 0:
                    corr = np.corrcoef(data1, data2)[0, 1]
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
        
        # 2. 산업 유사도 반영
        if "Sector" in snapshot_df.columns:
            sector_map = snapshot_df.set_index("Symbol")["Sector"].to_dict()
            industry_sim = np.zeros((n, n))
            for i, sym1 in enumerate(self.symbols):
                for j, sym2 in enumerate(self.symbols):
                    if sym1 in sector_map and sym2 in sector_map:
                        industry_sim[i, j] = 1.0 if sector_map[sym1] == sector_map[sym2] else 0.5
            
            edge_weights = corr_matrix * industry_sim
        else:
            edge_weights = corr_matrix

        # 임계값(0.35) 이상인 연결만 남김 (Binary Adjacency Matrix)
        adj = (edge_weights >= 0.35).astype(float) 
        return adj

    def _create_masked_graph(self, snapshot_df, active_mask):
        """존재하지 않는 종목(상장 전/폐지 등)의 연결을 제거한 그래프를 생성합니다."""
        n = len(self.symbols)
        adj = self._create_graph(snapshot_df) if not snapshot_df.empty else np.zeros((n, n))
        
        # 마스크 매트릭스를 이용하여 비활성 종목의 연결 제거
        mask_matrix = np.outer(active_mask, active_mask)
        return adj * mask_matrix

    def _create_windows(self):
        """전체 기간에 대한 윈도우 데이터를 생성합니다."""
        dates = sorted(self.monthly_df["Date"].unique())
        windows = []
        
        for i in range(len(dates) - self.window_size):
            window_dates = dates[i : i + self.window_size]
            target_date = window_dates[-1] # 현재 시점 (포트폴리오 구성 시점)
            next_date = dates[i + self.window_size] # 다음 시점 (수익률 확인 시점)
            
            window_df = self.monthly_df[self.monthly_df["Date"].isin(window_dates)]
            next_df = self.monthly_df[self.monthly_df["Date"] == next_date]
            
            features = []
            active_mask = []
            
            for symbol in self.symbols:
                stock_data = window_df[window_df["Symbol"] == symbol]
                # 현재 시점에 종목 데이터가 존재하는지 확인
                is_active = not stock_data[stock_data["Date"] == target_date].empty
                
                if is_active:
                    vals = stock_data[self.feature_cols].values
                    # 데이터 길이가 부족하면(상장 초기 등) 0으로 패딩
                    if len(vals) < self.window_size:
                        pad = np.zeros((self.window_size - len(vals), len(self.feature_cols)))
                        vals = np.vstack([pad, vals])
                    features.append(vals)
                    active_mask.append(True)
                else:
                    # 비활성 종목은 0으로 채움
                    features.append(np.zeros((self.window_size, len(self.feature_cols))))
                    active_mask.append(False)
            
            # 그래프 구성 (Mask 적용)
            active_mask = np.array(active_mask)
            adj = self._create_masked_graph(window_df[window_df["Date"] == target_date], active_mask)
            
            # 레이블 (다음 달 수익률)
            labels = []
            for symbol in self.symbols:
                val = next_df[next_df["Symbol"] == symbol]["Return_Raw"].values
                labels.append(val[0] if len(val) > 0 else 0.0)
                
            windows.append({
                "features": np.array(features), # (N, T, F)
                "adj_matrix": adj,              # (N, N)
                "labels": np.array(labels),     # (N,)
                "date": target_date,
                "active_mask": active_mask
            })
            
        return windows

    def get_state(self, idx):
        """RL 에이전트의 입력으로 사용할 상태 벡터를 반환합니다."""
        w = self.windows[idx]
        features = w["features"] # (N, T, F)
        adj = w["adj_matrix"]    # (N, N)
        
        # Flatten 및 결합: [특징 벡터..., 인접 행렬 벡터...]
        state = np.concatenate([features.flatten(), adj.flatten()])
        return state.astype(np.float32)

    def get_train_windows(self):
        return self.windows[:self.test_start_idx]

    def get_test_windows(self):
        return self.windows[self.test_start_idx:]
        
    def __len__(self):
        return len(self.windows)

# ============ 포트폴리오 환경 (Environment) ============

class HybridPortfolioEnv:
    """
    강화학습 환경 (Environment)
    - 상태(State), 행동(Action), 보상(Reward) 상호작용을 정의합니다.
    """
    def __init__(self, dataset, windows=None, initial_cash=1_000_000):
        self.dataset = dataset
        self.windows = windows if windows else dataset.windows
        self.initial_cash = initial_cash
        self.portfolio_value = initial_cash
        self.current_step = 0
        self.n_steps = len(self.windows)
        self.gamma = 2.0 # 위험 회피 성향 (Risk Aversion)
        self.cost_bps = 0.0005 # 거래 비용 (5bp)
        
        self.n_stocks = len(dataset.symbols)
        self.prev_weights = np.zeros(self.n_stocks)
        
    def reset(self):
        """환경 초기화"""
        self.current_step = 0
        self.portfolio_value = self.initial_cash
        self.prev_weights = np.zeros(self.n_stocks)
        return self._get_state(0) 

    def _get_state(self, idx):
        """현재 스텝의 상태 벡터 생성"""
        w = self.windows[idx]
        features = w["features"]
        adj = w["adj_matrix"]
        state = np.concatenate([features.flatten(), adj.flatten()])
        return state.astype(np.float32)

    def step(self, action):
        """
        행동(포트폴리오 비중)을 수행하고 다음 상태와 보상을 반환합니다.
        """
        w = self.windows[self.current_step]
        returns = w["labels"]
        
        # 포트폴리오 수익률 계산
        portfolio_return_pct = np.dot(action, returns)
        portfolio_return = portfolio_return_pct / 100.0
        
        # 거래 비용 계산 (Turnover * Cost)
        turnover = np.sum(np.abs(action - self.prev_weights))
        cost = turnover * self.cost_bps
        
        # 순수익률 및 포트폴리오 가치 갱신
        net_return = portfolio_return - cost
        self.portfolio_value *= (1 + net_return)
        
        self.current_step += 1
        done = (self.current_step >= self.n_steps)
        
        # 보상 함수 (CRRA Utility)
        # - 위험(변동성)을 고려한 효용 함수 사용
        safe_return = max(net_return, -0.99) # -100% 손실 방지
        exponent = 1.0 - self.gamma
        reward = ((1.0 + safe_return) ** exponent) / exponent
        
        self.prev_weights = action
        # 다음 상태 반환 (종료 시 0 벡터)
        next_state = self._get_state(self.current_step) if not done else np.zeros_like(self._get_state(0))
        
        info = {
            'portfolio_value': self.portfolio_value,
            'date': w["date"],
            'turnover': turnover,
            'cost': cost
        }
        
        return next_state, reward, done, info

# ============ 학습 및 실행 로직 ============

def train_hybrid(agent, env, num_episodes=100):
    """Hybrid 에이전트 학습 루프"""
    print(f"\n🚀 Hybrid 모델 학습 시작: 총 {num_episodes} 에피소드")
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        # 탐색 노이즈 감소 (Exploration scheduling)
        noise_std = max(0.01, 0.2 - episode * 0.002)
        
        while True:
            # 행동 선택 및 환경 상호작용
            action = agent.select_action(state, noise_std)
            next_state, reward, done, info = env.step(action)
            
            # 경험 저장
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # 학습 (배치 크기 64)
            if len(agent.replay_buffer) > 256:
                agent.train(batch_size=64)
                
            episode_reward += reward
            state = next_state
            
            if done: break
            
        if (episode+1) % 10 == 0:
            print(f"[{episode+1:3d}/{num_episodes}] 보상(Reward): {episode_reward:.2f}")

def calculate_metrics(ts_data, dates, strategy_name):
    """성과 지표 계산 (CAGR, MDD, Sharpe Ratio)"""
    df = pd.DataFrame(ts_data)
    df['date'] = pd.to_datetime(dates)
    
    initial = df['portfolio_value'].iloc[0]
    final = df['portfolio_value'].iloc[-1]
    
    days = (df['date'].max() - df['date'].min()).days
    years = days / 365.25
    cagr = ((final/initial)**(1/years) - 1) * 100 if years > 0 else 0
    
    mdd = abs(min(df['drawdown'])) * 100
    
    # Sharpe Ratio (연율화)
    r = df['return'] / 100
    vol = r.std() * np.sqrt(12)
    sharpe = (cagr/100) / (vol + 1e-8)
    
    return {
        "Strategy": strategy_name,
        "CAGR": cagr,
        "MDD": mdd,
        "Sharpe": sharpe,
        "Final_Value": final
    }

def run_hybrid_rebalancing(agent, dataset, freq="monthly"):
    """
    주기적 리밸런싱 전략 실행
    - freq: monthly(1개월), quarterly(3개월), semiannual(6개월), annual(12개월)
    """
    freq_map = {"monthly": 1, "quarterly": 3, "semiannual": 6, "annual": 12}
    interval = freq_map[freq]
    
    test_windows = dataset.get_test_windows()
    start_idx = dataset.test_start_idx 
    
    capital = 1_000_000
    peak = capital
    current_weights = np.ones(10) / 10 # 초기 비중 1/N
    
    ts_data = {"portfolio_value": [], "return": [], "drawdown": [], "turnover": []}
    dates = []
    trade_logs = []
    
    for i, w in enumerate(test_windows):
        state = dataset.get_state(start_idx + i)
        
        # 리밸런싱 주기가 되었을 때만 행동 수행
        if i % interval == 0:
            action = agent.select_action(state, noise_std=0.0)
            current_weights = action
            
            log = {"Date": w["date"], "Strategy": f"Hybrid({freq})", "Type": "Rebalance"}
            for sym, val in zip(dataset.symbols, current_weights):
                log[sym] = round(float(val), 4)
            trade_logs.append(log)
        else:
            # 리밸런싱 주기가 아닐 경우 현재 비중 유지
            # (실제로는 가격 변동에 따라 비중이 변하지만, 단순화를 위해 고정 비중 가정)
            log = {"Date": w["date"], "Strategy": f"Hybrid({freq})", "Type": "Hold"}
            for sym, val in zip(dataset.symbols, current_weights):
                log[sym] = round(float(val), 4)
            trade_logs.append(log)
            
        # 수익률 계산 및 자산 갱신
        ret = np.dot(current_weights, w["labels"])
        capital *= (1 + ret/100)
        
        # MDD 계산
        peak = max(peak, capital)
        dd = (capital - peak) / peak
        
        dates.append(w["date"])
        ts_data["portfolio_value"].append(capital)
        ts_data["return"].append(ret)
        ts_data["drawdown"].append(dd)
        ts_data["turnover"].append(0) 
        
    metrics = calculate_metrics(ts_data, dates, f"Hybrid({freq})")
    
    return {
        "dates": dates,
        "portfolio_values": ts_data["portfolio_value"],
        "metrics": metrics,
        "trade_logs": trade_logs
    }

def run_fixed_weights(dataset, strategy_name="1/N Buy & Hold"):
    """
    벤치마크: 고정 비중 전략 (Monthly Rebalancing 가정)
    """
    test_windows = dataset.get_test_windows()
    
    capital = 1_000_000
    peak = capital
    num_stocks = len(dataset.symbols)
    
    ts_data = {"portfolio_value": [], "return": [], "drawdown": [], "turnover": []}
    dates = []
    trade_logs = []
    
    for i, w in enumerate(test_windows):
        # 매월 1/N로 리밸런싱한다고 가정 (표준 벤치마크)
        current_weights = np.ones(num_stocks) / num_stocks
        
        if i == 0:
             log = {"Date": w["date"], "Strategy": strategy_name, "Type": "Init"}
             for sym, val in zip(dataset.symbols, current_weights):
                 log[sym] = round(float(val), 4)
             trade_logs.append(log)

        ret = np.dot(current_weights, w["labels"])
        capital *= (1 + ret/100)
        
        peak = max(peak, capital)
        dd = (capital - peak) / peak
        
        dates.append(w["date"])
        ts_data["portfolio_value"].append(capital)
        ts_data["return"].append(ret)
        ts_data["drawdown"].append(dd)
        ts_data["turnover"].append(0)
        
    metrics = calculate_metrics(ts_data, dates, strategy_name)
    return {
        "dates": dates,
        "portfolio_values": ts_data["portfolio_value"],
        "metrics": metrics,
        "trade_logs": trade_logs
    }


def main(mode="compare"):
    df = pd.read_csv(DATA_PATH)
    feature_cols = [
        "Beta", "MarketCap", "Momentum1M", "Momentum6M", "Volatility", "RSI",
        "Beta_Factor", "Value_Factor", "Size_Factor", "Momentum_Factor", "Volatility_Factor",
    ]
    
    print("\n[초기화] Hybrid 데이터셋 준비 중...")
    dataset = HybridDataset(df, feature_cols=feature_cols)
    
    num_stocks = 10
    window_size = 12
    num_features = len(feature_cols)
    
    # GPU 자동 감지
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[시스템] 사용 장치: {device}")
    
    agent = HybridAgent(num_stocks, window_size, num_features, device=device)

    model_path = Path(__file__).parent / "best_hybrid.pth"

    if mode == "train":
        print("\n[학습] 2015-2017 데이터를 사용하여 모델 학습 시작...")
        if model_path.exists():
            print("⚠️ 기존 모델을 발견했습니다. 삭제 후 재학습합니다...")
            model_path.unlink()
        
        train_windows = dataset.get_train_windows()
        train_env = HybridPortfolioEnv(dataset, windows=train_windows)
        
        # 실제 학습 (50 에포크)
        train_hybrid(agent, train_env, num_episodes=50)
        
        # Actor 모델 저장
        torch.save(agent.actor.state_dict(), model_path)
        print("✅ 모델 저장 완료 ({})".format(model_path))
        return

    elif mode == "compare":
        if not model_path.exists():
            print("⚠️ 학습된 모델이 없습니다. 먼저 'python run_comparison.py train'을 실행하세요.")
            return
            
        print("\n[테스트] 저장된 모델 로드 중...")
        agent.actor.load_state_dict(torch.load(model_path))
        
        print("\n[테스트] 2018-2025 데이터 백테스팅 수행...")
        
        # 전략별 백테스팅 실행
        buy_and_hold = run_fixed_weights(dataset, "1/N Buy & Hold")
        monthly = run_hybrid_rebalancing(agent, dataset, "monthly")
        quarterly = run_hybrid_rebalancing(agent, dataset, "quarterly")
        semiannual = run_hybrid_rebalancing(agent, dataset, "semiannual")
        annual = run_hybrid_rebalancing(agent, dataset, "annual")
        
        # 결과 저장 경로
        save_dir = ROOT_DIR / "results" / "03_Hybrid_TGNN_DDPG"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 결과 시각화
        visualizer = BacktestVisualizer(save_dir=save_dir)
        visualizer.plot_rebalancing_comparison(buy_and_hold, monthly, quarterly, semiannual, annual)
    
    print("\n=== Hybrid 모델 최종 성과 ===")
    results_list = [buy_and_hold, monthly, quarterly, semiannual, annual]
    
    # 요약 메트릭 저장
    summary_data = [res['metrics'] for res in results_list]
    pd.DataFrame(summary_data).to_csv(save_dir / "summary_metrics.csv", index=False)
    
    for res in results_list:
        m = res['metrics']
        print(f"{m['Strategy']:<15} | CAGR: {m['CAGR']:>6.1f}% | MDD: {m['MDD']:>6.1f}% | Final: ${m['Final_Value']:,.0f}")

    # 거래 로그 저장
    all_logs = []
    for res in results_list:
        all_logs.extend(res['trade_logs'])
    pd.DataFrame(all_logs).to_csv(save_dir / "hybrid_trade_logs.csv", index=False)
    print(f"\n✅ 거래 로그 저장 완료: {save_dir / 'hybrid_trade_logs.csv'}")


if __name__ == "__main__":
    import sys
    # 인자가 없으면 기본값으로 'compare' 사용
    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    main(mode=mode)
