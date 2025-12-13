# DDPG (Deep Deterministic Policy Gradient)

## 📖 개요
이 디렉토리는 **DDPG (Deep Deterministic Policy Gradient)** 강화학습 모델을 사용하여 포트폴리오 비중을 동적으로 최적화하고, 이를 기반으로 백테스팅을 수행하는 코드를 포함합니다.

DDPG는 연속적인 행동 공간에서 최적의 정책을 학습하는 **Actor-Critic** 알고리즘으로, 시장 상태에 따른 **전체 10개 종목의 최적 투자 비중**을 결정합니다.

---

## 🔬 학습 및 검증 방법: 3년 학습 / 8년 실전 투자

본 프로젝트는 **가장 학술적으로 공정하고 현실적인** 검증 방식을 채택합니다.

### 📅 타임라인
| 구분 | 기간 | 역할 | 비고 |
|------|------|------|------|
| **학습 (Train)** | **2015 ~ 2017** (3년) | AI 모델 학습 | 이 기간의 성과는 백테스팅에서 제외 |
| **백테스팅 (Test)** | **2018 ~ 2025** (8년) | **실전 투자 평가** | 오직 이 기간의 수익률만 비교 |

### 이 방식의 장점
1. **과적합(Overfitting) 원천 차단**: 2018년 이후의 데이터는 학습 단계에서 절대 보지 않습니다.
2. **공정한 비교**: TGNN과 DDPG, 그리고 Buy & Hold 모두 2018년 1월 1일에 투자를 시작한다고 가정합니다.
3. **현실성**: 실제 펀드 운용과 동일하게, 과거 데이터로 충분히 학습한 후 실전에 투입되는 시나리오입니다.

---

## 🧠 모델 구조 (Architecture)
DDPG 모델은 네 가지 핵심 모듈로 구성됩니다 (`model.py`).

1. **Actor Network** :
   - 현재 시장 상태(State)를 입력으로 받아 각 종목의 투자 비중(Action)을 출력합니다.
   - `Softmax` 활성화 함수를 사용하여 비중의 합이 항상 1.0이 되도록 보장합니다.

2. **Critic Network** :
   - State와 Action을 함께 받아 Q-Value(기대 누적 수익)를 추정합니다.
   - Actor의 행동이 얼마나 좋은지 평가하여 학습을 유도합니다.

3. **Target Networks** :
   - Actor/Critic 각각의 Target Network를 유지하여 학습 안정성을 확보합니다.
   - Soft Update (`τ=0.001`)를 통해 점진적으로 업데이트합니다.

4. **Replay Buffer** :
   - 경험 (State, Action, Reward, Next State)을 저장하고 랜덤 샘플링하여 학습합니다.

---

## 📊 데이터 및 전처리
- **입력 데이터** : 10개 주요 기술주 (AAPL, MSFT, NVDA, TSLA, GOOGL, AMZN, META, UNH, PLTR, IONQ)
- **Features (11개)** :
  - 기본 지표: Beta, MarketCap, Momentum(1M, 6M), Volatility, RSI
  - **5-Factor** : Beta_F, Value_F, Size_F, Momentum_F, Volatility_F
- **State Dimension** : 11 Features × 10 Stocks = **110차원**
- **Action Dimension** : 10 (각 종목의 투자 비중)

---

## 📐 보상 함수 설계 (Reward Function Engineering)

본 연구는 단순한 수익률 최대화가 아닌, **위험 회피적(Risk-Averse)이고 비용 효율적인** 투자를 유도하기 위해 금융 경제학 이론에 기반한 보상 함수를 사용합니다.

### 1. CRRA (Constant Relative Risk Aversion) 효용 함수
투자자의 리스크 회피 성향을 수학적으로 모델링하기 위해 **CRRA 효용 함수**를 도입했습니다. 이는 "돈을 잃는 고통"이 "돈을 버는 기쁨"보다 훨씬 크다(Loss Aversion)는 경제학적 대전제를 반영합니다.

$$
U(R) = \frac{(1 + R_{net})^{1 - \gamma}}{1 - \gamma}
$$

- $R_{net}$: 거래비용을 차감한 순수익률
- $\gamma$ (Gamma): **위험 회피 계수 (Risk Aversion Coefficient)**.
  - 본 모델에서는 **$\gamma=2.0$**을 사용하여 적절한 위험 회피와 적극적인 리밸런싱의 균형을 맞춥니다.
  - 초기 시도 ($\gamma=4.0$)에서의 '유동성 함정(Liquidity Trap)' 문제를 해결하고, 확실한 상승 기회(예: 2020년 TSLA)에는 과감하게 비용을 지불하고 자산을 교체하는 능동적 투자 성향을 유도했습니다.

### 2. 거래비용 모델 (Transaction Cost Model)
잦은 매매(Churning)로 인한 수익 잠식을 방지하기 위해 **턴오버(Turnover)에 비례한 페널티**를 부여합니다.

$$
Cost_t = \text{Turnover}_t \times \text{Fee Rate}
$$
$$
R_{net, t} = R_{portfolio, t} - Cost_t
$$

- 이는 모델이 "확실한 기회"가 아니면 포트폴리오를 굳이 건드리지 않도록(Inaction) 유도하여, 불필요한 거래 비용을 절감하고 장기 투자를 장려합니다.

---

## 🧪 백테스팅 전략

1. **투자 방식** :
   - 모든 **10개 종목에 투자** (TGNN의 Top-5와 다름)
   - Actor가 출력한 비중(Softmax)으로 자산 배분

2. **리밸런싱 빈도** (`run_comparison.py`):
   - 월간 / 분기 / 반기 / 연간 비교
   - **벤치마크** : 1/N Buy & Hold 전략 (2018~2025)

3. **평가 지표** :
   - **CAGR** : 연평균 복리 수익률
   - **MDD** : 최대 낙폭 (Maximum Drawdown)
   - **누적 수익률** : 2018년 이후의 총 수익률

---

## 📂 파일 구조

```
models/DDPG/
├── model.py                 # DDPG 모델 정의 (Actor, Critic, ReplayBuffer, Agent)
├── run_comparison.py        # 학습(Train) 및 백테스팅(Compare) 실행 스크립트
├── visualization.py         # 결과 시각화 클래스
├── best_ddpg.pth            # (생성됨) 학습된 Actor 모델 가중치 파일
└── README.md                # 현재 파일
```

---

## 🚀 실행 방법

### 1. 모델 학습 (Train)
2015~2017년 데이터로 DDPG 에이전트를 학습시킵니다.

```bash
python run_comparison.py train
```

- 학습이 완료되면 `best_ddpg.pth` 파일이 생성됩니다.

### 2. 백테스팅 비교 (Compare)
2018~2025년 데이터에 대해 실전 투자를 시뮬레이션합니다.

```bash
python run_comparison.py compare
```

- **결과물** :
  - `results/02_DDPG_Only/rebalancing_comparison.png`: 2018년 이후 수익률 그래프
  - `results/02_DDPG_Only/comparison_results.csv`: 전략별 성과 요약 CSV

---

## 🔬 핵심 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|:--------|:---|:----|
| `lr_actor` | 1e-4 | Actor 학습률 |
| `lr_critic` | 1e-3 | Critic 학습률 |
| `gamma` | 0.99 | 할인율 (Discount Factor) |
| `tau` | 0.001 | Target Network Soft Update 계수 |
| `batch_size` | 64 | 학습 배치 크기 |
| `buffer_size` | 10,000 | Replay Buffer 크기 |

---

## 📚 참고문헌

- Lillicrap, T. P., et al. (2015). *Continuous control with deep reinforcement learning.* ICLR.
- Silver, D., et al. (2014). *Deterministic Policy Gradient Algorithms.* ICML.
