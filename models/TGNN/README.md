# TGNN (Temporal Graph Neural Network)

## 📖 개요

이 디렉토리는 **TGNN (Temporal Graph Neural Network)** 모델을 사용해 주가 모멘텀을 예측하고, 이를 기반으로 리밸런싱 전략을 백테스트하는 코드를 포함합니다.  
TGNN은 종목 간 **그래프 구조(상관관계, 섹터)**와 **시계열 패턴(12개월 히스토리)**을 동시에 학습하여, 단순 시계열 모델보다 구조 정보를 더 잘 반영하는 것을 목표로 합니다.

이 TGNN 단일 모델은 이후 **TGNN + DDPG 하이브리드 모델**과의 성능 비교를 위한 **베이스라인**으로 사용됩니다.

---

## 🧠 모델 구조

TGNN 모델은 `model.py` 및 이 디렉토리의 스크립트에서 다음 세 가지 핵심 모듈로 구성됩니다.

1. **Graph Convolution (GCN)**

   - 종목 간 수익률 Feature의 상관계수와 섹터 정보를 바탕으로 인접 행렬(Adjacency Matrix)을 생성합니다.
   - 연결된 종목들 간의 정보를 전파하면서 포트폴리오 내 종목 관계 구조를 학습합니다.

2. **Temporal Attention**

   - 각 종목에 대해 과거 **12개월**의 시계열을 입력으로 받아, 시점별 중요도를 학습합니다.
   - 코로나 이후와 같은 국면 전환(Regime Change) 구간에서 중요한 구간에 더 큰 가중치를 부여합니다.

3. **Multi-Head Predictor (Momentum Heads)**
   - 하나의 TGNN 인코더에서 나온 임베딩을 이용해
     - **Momentum1M / Momentum3M / Momentum6M / Momentum12M**  
       네 개의 기간별 모멘텀을 동시에 예측합니다.
   - 백테스트 시 리밸런싱 주기(월간/분기/반기/연간)에 맞는 헤드를 선택하여 사용합니다.

---

## 📊 데이터 및 전처리

- **Universe (종목)**

  - AAPL, MSFT, NVDA, TSLA, GOOGL, AMZN, META, UNH, PLTR, IONQ 등 기술/성장주 중심 종목군이 사용됩니다.

- **Feature 컬럼 예시**  
  (실제 사용 컬럼은 코드의 `feature_cols` 정의 기준)

  - 기술 지표:
    - Volatility, RSI, MACD, Signal, MACD_Hist
  - 팩터/펀더멘털 기반 지표:
    - Beta_Factor, Value_Factor, Momentum_Factor, Volatility_Factor
  - 기타:
    - weighted_score, Mkt_RF, SMB, HML, RMW, CMA
  - 타겟:
    - Momentum1M, Momentum3M, Momentum6M, Momentum12M
      - 값은 모두 **소수 비율** (예: 0.05 = 5%) 형태로 사용됩니다.

- **기간 분할**

  - `data/train_data.csv`: 2006-01-01 ~ 2020-12-31
    - 학습(train): 2006-01-01 ~ 2018-12-31
    - 검증(val): 2019-01-01 ~ 2020-12-31
  - `data/test_data.csv`: 2021-01-01 ~ 2025-12-31

- **전처리 규칙**

  - Feature 컬럼만 **표준화(z-score)** 적용
    - 평균/표준편차는 학습 구간(2006–2020)에서 계산하고,
      - `results/01_TGNN_Only/scaler_params.npz`에 저장합니다.
    - 테스트 시 같은 값으로 스케일링을 적용합니다.
  - Momentum 컬럼은 **정규화 없이 원본 소수 값 사용**, 다만
    - 극단값만 `clip(-0.4, 0.5)`로 제한하여 이상치 영향을 줄입니다.

- **Dynamic Universe**
  - 상장 전에는 해당 종목을 유니버스에서 자동 제외하고, 상장 이후부터만 포트폴리오 구성 대상에 포함하는 **동적 종목 유니버스**를 지원합니다.
  - `TGNNDataset` / `TGNN_Dataset`의 `active_mask`와 그래프 생성 로직을 통해 구현됩니다.

---

## 🧪 리밸런싱 전략 및 백테스트

백테스트는 `backtest_tgnn.py`와 `Backtester` 클래스를 통해 수행됩니다.

1. **리밸런싱 주기**

   - 월간(Monthly) → Momentum1M 헤드 사용
   - 분기(Quarterly) → Momentum3M 헤드 사용
   - 반기(Semiannual) → Momentum6M 헤드 사용
   - 연간(Annual) → Momentum12M 헤드 사용

2. **종목 선정 (Selection)**

   - 각 리밸런싱 시점마다, 해당 기간 헤드에서 예측한 모멘텀 상위 **Top-K (기본 5개)** 종목을 선택합니다.
   - 거래 가능한 종목 수가 K 미만인 경우, 유효 종목 전체를 매수합니다.

3. **비중 할당 (Weighting)**

   - 현재 구현은 `weighting_method='equal'`을 사용한 **동일 비중(equal weight)** 포트폴리오입니다.
   - (추후 Softmax 기반 가중치 등으로 확장 가능)

4. **비교 대상 전략**
   - **Buy & Hold**: 초기 동일 비중으로 매수 후, 추가 리밸런싱 없이 보유.
   - **TGNN 1M/3M/6M/12M**: TGNN multi-head 예측에 기반한 리밸런싱 전략.

---

## 📂 파일 구조 (feat/models/TGNN 브랜치 기준)

models/TGNN/
├── model.py # 기본 TGNN 모델 및 단일 타겟 Dataset 정의
├── backtester.py # Backtester, BacktestConfig, 메트릭 계산 로직
├── train_tgnn.py # Multi-head TGNN 학습 스크립트 (Train/Val)
├── backtest_tgnn.py # 학습된 TGNN으로 백테스트 수행
└── README.md # 현재 파일

## 🚀 실행 방법

### 1. TGNN 학습 (Train only)

TGNN Multi-Head 모델을 학습하고, 모델 가중치 및 스케일러를 저장합니다.

python -m models.TGNN.train_tgnn

- 생성 파일:
  - `results/01_TGNN_Only/best_tgnn_multi.pth`
  - `results/01_TGNN_Only/scaler_params.npz`

### 2. TGNN 백테스트 (Backtest only)

학습된 TGNN 모델을 이용해 Buy & Hold 대비 TGNN 전략 성과를 비교합니다.

python -m models.TGNN.backtest_tgnn

- 생성 파일(예시):
  - `results/01_TGNN_Only/comparison_metrics.csv`
  - `results/01_TGNN_Only/metrics_comparison.csv`
  - `results/01_TGNN_Only/timeseries_*.csv`
  - `results/01_TGNN_Only/comparison_graph.png`

---

## 📈 현재 TGNN 단일 모델 결과 (예시)

테스트 구간(2021–2025) 기준 예시 결과:

- **Buy & Hold**

  - 누적수익률: 약 0.5%
  - Sharpe: 약 -17.6

- **TGNN 12M (연간 리밸런싱)**
  - 누적수익률: 약 5.7%
  - CAGR: 약 1.4%
  - Sharpe: 약 -3.3

해석:

- TGNN은 단순 Buy & Hold 대비 **조금 더 높은 누적 수익률과 덜 나쁜 샤프지수**를 보이지만,
- 절대적인 의미에서는 Sharpe가 여전히 음수로, **리스크 대비 수익이 충분히 만족스럽지는 못한 상태**입니다.
- 따라서 본 TGNN 모델은 이후 구현되는 **TGNN + DDPG 하이브리드 모델**과의 비교에서
  - TGNN 단독의 한계를 보여주고
  - 강화학습을 통한 의사결정(policy) 개선 효과를 검증하기 위한 **베이스라인 모델**로 사용됩니다.
