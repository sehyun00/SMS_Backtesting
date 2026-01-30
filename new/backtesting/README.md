# SMS Backtesting Framework (Research Edition)

> 🔬 **학술 연구용**으로 설계된 하이브리드(TGNN + DDPG) 주가 예측 및 포트폴리오 최적화 프레임워크입니다.

이 프로젝트는 **Temporal Graph Neural Networks (TGNN)**의 강력한 시장 예측 능력과 **Deep Deterministic Policy Gradient (DDPG)**의 포트폴리오 최적화 능력을 결합하여, 기존 방법론 대비 우수한 위험 조정 수익률(Risk-adjusted Return)을 달성하는 것을 목표로 합니다.

## 🌟 핵심 기능 (Key Features)

1.  **Hybrid Architecture**: 지도 학습(TGNN)의 Feature Extraction 능력을 강화 학습(DDPG)의 State로 활용.
2.  **Horizon Matching Strategy**: 예측 주기(1M, 3M, 6M, 12M)와 리밸런싱 주기를 일치시켜 학술적 정합성 확보.
3.  **Transfer Learning**: 학습된 유니버스(Training Universe)와 다른 테스트 유니버스(Test Universe)에 대해 파라미터 전이 및 미세 조정(Fine-tuning) 지원.
4.  **Realistic Backtesting**: 거래 비용, 슬리피지(Slippage)는 물론 **Portfolio Drift(가격 변동에 따른 비중 변화)**와 **Turnover**를 정밀하게 시뮬레이션.

---

## 📦 디렉토리 구조 (Directory Structure)

```bash
new/backtesting/
├── config/              # 실험 설정 (Hyperparameters)
├── src/
│   ├── models/          # TGNN, DDPG, Hybrid 모델 정의
│   ├── training/        # 학습 루프 (Trainer, RL Trainer)
│   ├── preprocessing/   # 데이터 전처리 및 팩터 생성
│   ├── backtest/        # 백테스팅 엔진 및 시각화
│   ├── pipelines/       # 실행 파이프라인 (Train -> Backtest)
│   └── utils/           # 유틸리티 함수
├── main.py              # 실행 진입점 (Entry Point)
└── README.md            # (This File)
```

---

## 🏗️ 아키텍처 (Architecture)

### 1. 전처리 (Preprocessing)
*   **Input**: `yfinance` 주가 데이터 + Fama-French 5 Factor
*   **Process**: 기술적 지표(RSI, MACD 등) 생성 $\rightarrow$ 결측치 처리 $\rightarrow$ 정규화
*   **Output**: `processed_daily_5factor_model.csv`

### 2. 모델 (Models)
*   **TGNN (Temporal Graph Neural Net)**:
    *   입력: $[N, T, F]$ (Nodes, Time-steps, Features)
    *   구조: GAT (Graph Attention) + GRU/LSTM
    *   출력: Multi-horizon Momentum Score (1M, 3M, 6M, 12M)
*   **DDPG (Deep Deterministic Policy Gradient)**:
    *   입력: TGNN이 추출한 Latent Embedding + Market State
    *   출력: 포트폴리오 비중 Vector $[w_1, w_2, ..., w_N]$ ($\sum w_i = 1$)

### 3. 백테스팅 (Backtesting)
*   **Logic**: `Standard` (Buy & Hold) vs `Model` (TGNN/DDPG)
*   **Rebalancing**:
    *   **Monthly** (21일): 1개월 예측 헤드 사용
    *   **Quarterly** (63일): 3개월 예측 헤드 사용
    *   **Semiannual** (126일): 6개월 예측 헤드 사용
    *   **Annual** (252일): 12개월 예측 헤드 사용

---

## 🚀 시작하기 (Getting Started)

### 1. 환경 설정 (Configuration)
`config/config.yaml` 에서 실험 파라미터를 조정합니다.

```yaml
project:
  selected_model: "tgnn"  # "tgnn", "ddpg", "hybrid"

data:
  stock_universes: ["AAPL", "MSFT", "GOOGL"] # 비워두면 전체 사용
  window_size: 12

training:
  episodes: 800
  learning_rate: 0.001
```

### 2. 실행 (Execution)
`main.py`를 통해 모든 파이프라인을 실행할 수 있습니다.

#### 데이터 전처리
```bash
python main.py --mode preprocess
```

#### 모델 학습 (Train)
```bash
python main.py --mode train
```

#### 백테스팅 (Backtest)
```bash
python main.py --mode backtest
```

---

## 📊 결과 확인 (Results)

실행 결과는 `results/{model_name}/` 디렉토리에 저장됩니다.

*   `logs/trade_logs.csv`: 일자별 포트폴리오 비중 및 리밸런싱 내역
*   `plots/comparison.png`: 벤치마크 대비 누적 수익률, MDD, CAGR 그래프
*   `checkpoints/`: 학습된 모델 가중치 (.pth)

---

## 🧪 연구 재현성 (Reproducibility)
*   **Seed 고정**: `config.yaml`의 `seed` 값(Default: 42)을 통해 모든 난수 발생을 통제합니다.
*   **Lookahead Bias 제거**: 백테스팅 시 `current_weights` 업데이트에 `Target Date`의 수익률(`t`)이 아닌, `Target Date + 1`(`t+1`)의 수익률을 적용하여 미래 정보 참조를 원천 차단했습니다.

---

## 📚 추가 문서 (Research Documentation)
이 프로젝트의 실험 설계 및 연구 논리에 대한 상세 문서는 `docs/` 디렉토리에 있습니다.

*   📄 **[Research Notes (연구 노트)](docs/research_notes.md)**: 모델 설계 철학(Why Softmax?), Full Universe 전략의 정당성 등 핵심 연구 가설을 설명합니다.
*   📊 **[Data Description (데이터 기술서)](docs/data_description.md)**: Fama-French 5 Factor, 기술적 지표 등의 변수 구성과 전처리 상세를 기술하여 **논문의 'Data' 섹션** 작성에 활용합니다.

> **💡 Note for Researchers**: 논문 작성 시, 이 문서들에 기술된 수식과 방법론을 인용(Cite)하여 실험의 신뢰성을 높이세요.

