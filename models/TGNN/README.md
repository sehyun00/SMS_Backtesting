# TGNN (Temporal Graph Neural Network)

## 📖 개요
이 디렉토리는 **TGNN (Temporal Graph Neural Network)** 모델을 사용하여 주가 수익률을 예측하고, 이를 기반으로 포트폴리오를 구성하는 코드를 포함합니다.
TGNN은 종목 간의 **상관관계(Spatial)**와 **시계열 패턴(Temporal)**을 동시에 학습하여, 전통적인 시계열 모델(LSTM, Transformer)보다 우수한 예측 성능을 목표로 합니다.

---

## 🧠 모델 구조 (Architecture)
TGNN 모델은 크게 세 가지 핵심 모듈로 구성됩니다 (`model.py`).

1.  **Graph Convolution (GCN)**:
    -   종목 간의 상관계수(Correlation)를 기반으로 인접 행렬(Adjacency Matrix)을 생성합니다.
    -   연결된 종목들 간의 정보를 교환하여 시장의 구조적 패턴을 학습합니다.

2.  **Temporal Attention**:
    -   과거 12개월의 데이터 중에서 현재 예측에 중요한 시점에 가중치를 부여합니다.
    -   시장의 국면 변화(Regime Change)를 포착하는 데 유리합니다.

3.  **Predictor**:
    -   최종적으로 다음 달의 **1개월 모멘텀(Momentum1M)**을 예측합니다.

---

## 📊 데이터 및 전처리
-   **입력 데이터**: 10개 주요 기술주 (AAPL, MSFT, NVDA, TSLA, GOOGL, AMZN, META, UNH, PLTR, IONQ)
-   **Features (11개)**:
    -   기본 지표: Beta, MarketCap, Momentum(1M, 6M), Volatility, RSI
    -   **5-Factor**: Beta_F, Value_F, Size_F, Momentum_F, Volatility_F
    -   *(Note: PBR은 미래 정보 누수(Look-ahead Bias) 방지를 위해 제외됨)*
-   **기간**: 2015.01 ~ 2025.11

---

## 🧪 백테스팅 전략 (Dynamic Universe)
본 프로젝트는 실제 투자 환경을 모사하기 위해 **Dynamic Universe** 방식을 채택했습니다.

1.  **동적 종목 관리**:
    -   2015년에는 상장되지 않은 종목(PLTR, IONQ)은 포트폴리오에서 자동으로 제외됩니다.
    -   해당 종목이 상장되는 시점(2020년, 2021년)부터 자동으로 유니버스에 편입됩니다.

2.  **리밸런싱 전략 (`run_comparison.py`)**:
    -   **주기**: 월간 / 분기 / 반기 / 연간 리밸런싱 비교
    -   **종목 선정 (Selection)**:
        -   TGNN 모델이 예측한 **수익률 점수(Score)** 상위 5개 종목 선정.
        -   (거래 가능 종목이 5개 미만일 경우 전체 매수)
    -   **비중 할당 (Weighting)**:
        -   **Softmax 가중치**: 예측 점수가 높을수록 더 많은 비중을 할당합니다.
        -   AI가 5-Factor를 분석하여 확신이 강한 종목에 집중 투자하는 효과를 냅니다.

---

## 📂 파일 구조
models/TGNN/
├── model.py # TGNN 모델 정의 및 데이터셋(Dynamic Universe) 클래스
├── run_comparison.py # 학습(Train) 및 백테스팅(Compare) 실행 스크립트
├── best_tgnn_model.pth # (생성됨) 학습된 모델 가중치 파일
└── README.md # 현재 파일

---

## 🚀 실행 방법

### 1. 모델 학습 (Train)
PBR을 제외한 11개 Feature로 모델을 새로 학습합니다.

python run_comparison.py train

-   학습이 완료되면 `best_tgnn_model.pth` 파일이 생성됩니다.

### 2. 백테스팅 비교 (Compare)
Buy & Hold 전략과 TGNN의 리밸런싱 주기별 성과를 비교합니다.

python run_comparison.py compare

-   **결과물**:
    -   `rebalancing_comparison.png`: 누적 수익률 및 MDD 비교 그래프
    -   로그 출력: 리밸런싱 내역 및 활성 종목 수 변화 확인 가능

---

## 📈 주요 결과 요약
-   **안정성**: TGNN 전략(월간 리밸런싱)은 Buy & Hold 대비 **MDD(최대 낙폭)를 약 6~9%p 감소**시키는 효과가 있습니다.
-   **수익률**: 대세 상승장(Bull Market)에서는 Buy & Hold의 승자 독식(Winner-take-all) 효과로 인해 단순 보유 전략의 수익률이 더 높게 나타날 수 있습니다.
-   **결론**: TGNN 단일 모델은 **리스크 관리(Risk Management)** 측면에서 우수하며, 이를 기반으로 DDPG(강화학습)를 결합하여 수익률을 극대화하는 하이브리드 모델로 발전시킬 필요가 있습니다.
