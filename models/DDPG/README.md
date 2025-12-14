# DDPG (Deep Deterministic Policy Gradient)

## 📖 개요
이 디렉토리는 **DDPG (Deep Deterministic Policy Gradient)** 강화학습 모델을 사용하여 포트폴리오 비중을 동적으로 최적화하고, 이를 기반으로 백테스팅을 수행하는 코드를 포함합니다.

DDPG는 연속적인 행동 공간에서 최적의 정책을 학습하는 **Actor-Critic** 알고리즘으로, 시장 상태에 따른 **전체 10개 종목의 최적 투자 비중**을 결정합니다.

### 🆕 2025-12-14 업데이트
- **Factor-Aware 아키텍처**: SharedFactorEncoder를 통한 종목 간 공통 팩터 학습
- **비중 제약**: 2% ~ 30% 범위 제한으로 극단적 집중 투자 방지
- **엔트로피 정규화**: 분산투자 유도 및 과적합 방지
- **8가지 핵심 요소**가 협력하여 안정적이고 고성능 투자 전략 학습

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

## 🧠 모델 구조 (Factor-Aware Architecture)

DDPG 모델은 **Factor-Aware 구조**를 채택하여 종목 간 공통 팩터를 학습하고, 이를 통해 더 효율적이고 일반화된 투자 전략을 수립합니다.

### 1. SharedFactorEncoder (핵심 혁신!)
```python
class SharedFactorEncoder(nn.Module):
    # 입력: (Batch, 10 stocks, 11 features)
    # 출력: (Batch, 10 stocks, 64 hidden)
    Linear(11→64) → LayerNorm → ReLU → Linear(64→64) → LayerNorm → ReLU
```

**특징**:
- 모든 종목이 **동일한 가중치를 공유**하여 팩터 분석
- 11개 Features (기본 6개 + Fama-French 5-Factor) 모두 처리
- **전이 학습 효과**: "NVDA의 성장 패턴"을 학습하여 "TSLA" 등 다른 종목에도 적용
- 파라미터 효율성: 10개 종목 × 별도 네트워크 ❌ → 1개 공유 네트워크 ✅

### 2. Actor Network (포트폴리오 결정)
```python
class Actor(nn.Module):
    # State (110,) → Reshape (10, 11)
    # → SharedFactorEncoder (10, 64)
    # → Flatten (640,)
    # → Global Network (640→128→128→10)
    # → Softmax + Weight Constraints + Entropy
```

**주요 기능**:
- 시장 상태를 분석하여 각 종목의 최적 비중 결정
- **Softmax**: 비중 합=1 자동 정규화
- **비중 제약**: 2% ~ 30% 범위 강제 (극단적 집중 방지)
- **엔트로피 계산**: 분산투자 정도 측정

### 3. Critic Network (가치 평가)
```python
class Critic(nn.Module):
    # State → SharedFactorEncoder (640,)
    # → Concat [State Features, Action] (650,)
    # → MLP (650→128→128→1)
    # → Q-Value
```

**역할**:
- State와 Action을 입력받아 Q-Value(기대 누적 보상) 추정
- Actor의 행동이 얼마나 좋은지 평가
- Bellman Equation으로 최적 정책 학습 유도

### 4. Target Networks (학습 안정화)
- Actor Target, Critic Target 각각 유지
- **Soft Update** (τ=0.001): 0.1%씩 천천히 업데이트
- Moving Target 문제 해결로 학습 안정성 확보

### 5. Replay Buffer (경험 재사용)
- Buffer Size: 10,000
- Random Sampling으로 상관관계 제거
- Off-Policy 학습으로 데이터 효율성 향상

---

## 🎯 8가지 핵심 요소

본 모델은 다음 8가지 요소가 협력하여 안정적이고 고성능 투자 전략을 학습합니다:

### 1. CRRA 효용함수 (주 보상)
```python
Reward = ((1 + net_return)^(1-γ)) / (1-γ)  # γ=2.0
       = -1 / (1 + net_return)
```
- **역할**: 위험회피 투자자의 효용 반영
- **효과**: 손실에 더 큰 페널티, Loss Aversion 모델링
- **예시**: +10% 수익 = -0.91점, -10% 손실 = -1.11점

### 2. 거래비용 (Transaction Cost)
```python
turnover = sum(|new_weight - old_weight|)
cost = turnover × 0.05% (5bps)
net_return = portfolio_return - cost
```
- **역할**: 과도한 리밸런싱 억제
- **효과**: 불필요한 거래 비용 절감, 장기 투자 유도

### 3. Softmax 정규화
```python
weights = F.softmax(scores, dim=-1)  # 합=1.0
```
- **역할**: 비중을 자연스러운 확률 분포로 변환
- **효과**: 자동 정규화, 수치 안정성

### 4. Exploration Noise (탐색 노이즈)
```python
noise_std = max(0.01, 0.2 - episode * 0.002)
# 초반 20% → 점차 감소 → 최종 1%
```
- **역할**: 탐색-활용 균형
- **효과**: 다양한 전략 탐색, 로컬 최적해 탈출

### 5. Target Network Soft Update
```python
θ_target ← 0.001 × θ_main + 0.999 × θ_target
```
- **역할**: 학습 안정화
- **효과**: Moving Target 문제 해결, 과적합 방지

### 6. LayerNorm (레이어 정규화)
```python
nn.Linear(11, 64) → nn.LayerNorm(64) → nn.ReLU()
```
- **역할**: 각 레이어 출력 정규화 (평균 0, 분산 1)
- **효과**: Gradient 안정성, 학습 속도 향상, 깊은 네트워크 학습 가능

### 🆕 7. 비중 제약 (Weight Constraints)
```python
weights = torch.clamp(weights, min=0.02, max=0.30)
weights = weights / weights.sum()  # 다시 정규화
```
- **역할**: 극단적 집중/분산 방지
- **효과**: NVDA 과집중 방지 (최대 30%), 최소 투자 보장 (최소 2%)

### 🆕 8. 엔트로피 정규화 (Entropy Regularization)
```python
entropy = -sum(weights × log(weights))
entropy_bonus = -0.01 × entropy.mean()
total_loss = actor_loss + entropy_bonus
```
- **역할**: 분산투자 유도
- **효과**: 과적합 방지, 탐색 능력 향상, 로컬 최적해 탈출
- **예시**: 균등 분산(0.1씩) = entropy 2.30 (보너스 ↑), 집중(0.9+0.1×9) = entropy 0.73 (보너스 ↓)

---

## 📊 데이터 및 전처리

- **입력 데이터**: 10개 주요 기술주 (AAPL, MSFT, NVDA, TSLA, GOOGL, AMZN, META, UNH, PLTR, IONQ)
- **Features (11개)**:
  - **기본 지표 (6개)**: Beta, MarketCap, Momentum1M, Momentum6M, Volatility, RSI
  - **Fama-French 5-Factor (5개)**: Beta_Factor, Value_Factor, Size_Factor, Momentum_Factor, Volatility_Factor
- **State Dimension**: 11 Features × 10 Stocks = **110차원**
- **Action Dimension**: 10 (각 종목의 투자 비중, 합=1)
- **전처리**: StandardScaler 정규화 (학습 데이터 기준)

---

## 📐 보상 함수 설계 (Reward Function Engineering)

본 연구는 단순한 수익률 최대화가 아닌, **위험 회피적(Risk-Averse)이고 비용 효율적인** 투자를 유도하기 위해 금융 경제학 이론에 기반한 보상 함수를 사용합니다.

### CRRA (Constant Relative Risk Aversion) 효용 함수
투자자의 리스크 회피 성향을 수학적으로 모델링하기 위해 **CRRA 효용 함수**를 도입했습니다.

$$
U(R) = \frac{(1 + R_{net})^{1 - \gamma}}{1 - \gamma}
$$

- $R_{net}$: 거래비용을 차감한 순수익률
- $\gamma$ (Gamma): **위험 회피 계수 (Risk Aversion Coefficient)**
  - 본 모델: **γ=2.0** (적절한 위험 회피 + 적극적 리밸런싱 균형)

**핵심 특징**:
- "돈을 잃는 고통" > "돈을 버는 기쁨" (Loss Aversion)
- 손실에 더 큰 페널티 부여
- 안정적 포트폴리오 유도

### 거래비용 모델 (Transaction Cost Model)
```python
Cost_t = Turnover_t × 0.05%
R_net = R_portfolio - Cost_t
```
- 잦은 매매(Churning) 방지
- "확실한 기회"가 아니면 거래하지 않도록 유도
- 장기 투자 장려

---

## 🔄 전체 작동 흐름

```
Step 1: Actor → 포트폴리오 비중 제안
        ↓ (Softmax 정규화)
        ↓ (비중 제약: 2% ~ 30%)
        ↓ (엔트로피 계산)

Step 2: 시장 → 수익률 계산
        portfolio_return = Σ(weight_i × return_i)

Step 3: 거래비용 차감
        cost = turnover × 0.05%
        net_return = portfolio_return - cost

Step 4: CRRA 보상 계산
        reward = -1 / (1 + net_return)

Step 5: Critic → Q-Value 학습
        target_q = reward + γ × next_q
        (LayerNorm 적용)

Step 6: Actor → Q-Value 최대화
        (+ Exploration Noise)
        (+ 엔트로피 보너스)

Step 7: Target Network 업데이트
        (Soft Update: τ=0.001)
```

---

## 🧪 백테스팅 전략

### 1. 투자 방식
- 모든 **10개 종목에 투자** (TGNN의 Top-5와 다름)
- Actor가 출력한 비중으로 자산 배분
- 비중 제약: 각 종목 2% ~ 30%

### 2. 리밸런싱 빈도 (`run_comparison.py`)
- **월간** / 분기 / 반기 / 연간 비교
- **벤치마크**: 1/N Buy & Hold 전략 (2018~2025)

### 3. 평가 지표
- **CAGR**: 연평균 복리 수익률
- **MDD**: 최대 낙폭 (Maximum Drawdown)
- **Sharpe Ratio**: 위험 대비 수익률
- **Sortino Ratio**: 하방 위험 대비 수익률
- **Turnover**: 평균 연간 회전율

### 4. 백테스팅 결과 (2018-2025)

| 전략 | CAGR | MDD | Sharpe | 최종 자산 |
|------|------|-----|--------|----------|
| **DDPG (월간)** | **43.2%** | **-38.5%** | **1.34** | **$16.2M** |
| DDPG (분기) | 40.6% | -44.7% | 1.26 | $14.0M |
| 1/N Buy & Hold | 35.4% | -44.5% | 1.21 | $10.4M |

**핵심 발견**:
- 월간 리밸런싱이 최고 성과 (CAGR 43.2%)
- Buy & Hold 대비 CAGR +7.8%p, MDD -6.0%p 개선
- NVDA 집중 전략 (평균 비중 30-60%), 하지만 비중 제약으로 리스크 관리

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
python models/DDPG/run_comparison.py train
```

- 학습이 완료되면 `best_ddpg.pth` 파일이 생성됩니다.
- 100 에피소드 학습 (각 에피소드 = 36개월 거래)

### 2. 백테스팅 비교 (Compare)
2018~2025년 데이터에 대해 실전 투자를 시뮬레이션합니다.

```bash
python models/DDPG/run_comparison.py compare
```

**결과물**:
- `results/02_DDPG_Only/rebalancing_comparison.png`: 수익률 그래프
- `results/02_DDPG_Only/results_summary.json`: 전략별 성과 요약
- `results/02_DDPG_Only/results_ddpg_timeseries.csv`: 시계열 데이터
- `results/02_DDPG_Only/trade_logs.csv`: 거래 로그

---

## 🔬 핵심 하이퍼파라미터

### 학습 관련
| 파라미터 | 값 | 설명 |
|:--------|:---|:----|
| `lr_actor` | 1e-4 | Actor 학습률 |
| `lr_critic` | 1e-3 | Critic 학습률 |
| `gamma` | 0.99 | 할인율 (Discount Factor) |
| `tau` | 0.001 | Target Network Soft Update 계수 |
| `batch_size` | 64 | 학습 배치 크기 |
| `buffer_size` | 10,000 | Replay Buffer 크기 |
| `episodes` | 100 | 학습 에피소드 수 |

### 보상 관련
| 파라미터 | 값 | 설명 |
|:--------|:---|:----|
| `gamma_CRRA` | 2.0 | 위험 회피 계수 |
| `transaction_cost` | 5bps | 거래비용 (0.05%) |
| `entropy_coef` | 0.01 | 엔트로피 정규화 계수 |

### 모델 관련
| 파라미터 | 값 | 설명 |
|:--------|:---|:----|
| `hidden_dim_encoder` | 64 | SharedFactorEncoder 은닉 차원 |
| `hidden_dim_actor` | 128 | Actor/Critic 은닉 차원 |
| `min_weight` | 0.02 | 최소 종목 비중 (2%) |
| `max_weight` | 0.30 | 최대 종목 비중 (30%) |

---

## 💡 핵심 혁신 요약

### 1. Factor-Aware Architecture
- SharedFactorEncoder를 통한 종목 간 공통 팩터 학습
- 전이 학습 효과로 일반화 능력 향상
- 파라미터 효율성 (10x 감소)

### 2. 3중 분산투자 안전장치
1. **Softmax**: 자연스러운 확률 분포
2. **비중 제약**: 강제 범위 제한 (2-30%)
3. **엔트로피**: 보상으로 분산 유도

### 3. 금융 경제학 기반 보상 함수
- CRRA 효용함수 (위험회피)
- 거래비용 모델 (비용 효율성)
- Loss Aversion 반영

### 4. 8가지 요소의 협력
- CRRA, 거래비용, Softmax, Exploration Noise
- Target Network, LayerNorm, 비중 제약, 엔트로피
- → 안정적이고 고성능 투자 전략 학습

---

## 📚 참고문헌

- Lillicrap, T. P., et al. (2015). *Continuous control with deep reinforcement learning.* ICLR.
- Silver, D., et al. (2014). *Deterministic Policy Gradient Algorithms.* ICML.
- Fama, E. F., & French, K. R. (2015). *A five-factor asset pricing model.* Journal of Financial Economics.
- Merton, R. C. (1969). *Lifetime portfolio selection under uncertainty: The continuous-time case.* The Review of Economics and Statistics.

---

## 📝 업데이트 이력

- **2025-12-14**: Factor-Aware 구조, 비중 제약, 엔트로피 정규화 추가, 8가지 핵심 요소 문서화
- **2025-12-13**: CRRA 보상 함수 적용, 백테스팅 결과 추가
- **2025-11-28**: 초기 README 작성
