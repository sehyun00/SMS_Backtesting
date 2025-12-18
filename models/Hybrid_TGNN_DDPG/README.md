# Hybrid TGNN-DDPG Model

## 📖 개요
본 모델은 **TGNN (Temporal Graph Neural Network)**의 시공간적 패턴 인식 능력과 **DDPG (Deep Deterministic Policy Gradient)**의 연속적 포트폴리오 제어 능력을 결합한 하이브리드 강화학습 알고리즘입니다.

- **TGNN (State Encoder)**: 종목 간의 관계(상관관계, 산업군)와 시계열 트렌드를 분석하여 고차원 Node Embedding을 생성
- **DDPG (RL Agent)**: 생성된 임베딩을 기반으로 효용함수를 최대화하는 최적의 자산 배분 비중을 결정
- **Ensemble Alpha**: TGNN 예측과 균등 분산(1/N) 전략을 동적으로 결합

---

## 🔬 학습 및 검증 전략

| 구분 | 기간 | 역할 | 데이터 |
|------|------|------|--------|
| **Train** | 2006 ~ 2020 (15년) | Hybrid 에이전트 학습 | train_data.csv |
| **Test** | 2021 ~ 2025 (5년) | Out-of-Sample 백테스팅 | test_data.csv |

### 주요 학습 이벤트
- **2008년 금융위기**: MDD -37% ~ -56% (리스크 관리 학습)
- **2009-2019 장기 상승장**: 사상 최장 Bull Market (수익 극대화 학습)
- **2020년 COVID-19**: 극단적 변동성 (회복력 학습)

### 전이 학습 (Transfer Learning)
- Train 종목 수와 Test 종목 수가 다를 경우 자동으로 TGNN Encoder 레이어만 로드
- 종목 수에 의존하지 않는 시계열 패턴 학습을 재활용

---

## 🧠 모델 아키텍처

### 1. Input (State Space)
- **Node Features**: `(Batch, N_stocks, Window=12, N_features=20)`
  - 기술적 지표: Close, Volume, RSI, MACD, Momentum (1M/3M/6M/12M), Volatility
  - Fama-French 5 Factors: Mkt-RF, SMB, HML, RMW, CMA
- **Adjacency Matrix**: `(Batch, N_stocks, N_stocks)` - 종목 간 수익률 상관관계
- **Labels**: `Momentum1M` (다음 달 수익률, -95% ~ +200%)

### 2. Network Flow
```
Input Features → TGNN Encoder → Node Embeddings → TGNN Weights (Softmax)
                                       ↓
                     Alpha Network (MDD 기반 동적 가중치)
                                       ↓
              α * TGNN + (1-α) * 1/N = Final Portfolio
                                       ↓
                            Actor Network (DDPG)
                                       ↓
                          Critic Network (Q-Value)
```

**핵심 구성 요소:**
1. **Graph Convolution Layer**: 종목 간 정보 전파 (이웃 노드 집계)
2. **Temporal Attention**: 과거 12개월 중 중요 시점 자동 선택
3. **Alpha Network**: 현재 MDD 기반 동적 앙상블 비율 결정
   - MDD 높음 → α 감소 (보수적, 1/N 선호)
   - MDD 낮음 → α 증가 (공격적, TGNN 신뢰)
4. **Actor Network**: Softmax 출력으로 포트폴리오 비중 생성
5. **Critic Network**: 포트폴리오 가치 평가 (Q-Value)

### 3. Output (Action Space)
- **Portfolio Weights**: `(N_stocks,)` - 각 종목 투자 비중 (합=1, 범위=[0,1])
- **Alpha Value**: `[0, 1]` - TGNN 신뢰도

---

## 📂 디렉토리 구조

```
models/Hybrid_TGNN_DDPG/
│
├── 📁 agent/                      # 강화학습 에이전트
│   ├── hybrid_agent.py           # HybridAgent (DDPG 학습 로직)
│   └── replay_buffer.py          # 경험 리플레이 버퍼
│
├── 📁 environment/                # 포트폴리오 환경
│   ├── dataset.py                # HybridDataset (데이터 로드/전처리)
│   └── portfolio_env.py          # HybridPortfolioEnv (RL 환경)
│
├── 📁 networks/                   # 신경망 아키텍처
│   ├── actor.py                  # HybridActor (정책 네트워크)
│   ├── critic.py                 # HybridCritic (가치 네트워크)
│   ├── tgnn_encoder.py           # TGNN 시계열 그래프 인코더
│   └── graph_layers.py           # GraphConvLayer, TemporalAttention
│
├── 📁 utils/                      # 유틸리티
│   ├── metrics.py                # 성과 지표 계산 (CAGR, MDD, Sharpe)
│   └── constraints.py            # 포트폴리오 제약 조건
│
├── 📄 run_comparison.py           # 메인 실행 스크립트
├── 📄 training_monitor.py         # 학습 진행 모니터링
├── 📄 visualization.py            # 백테스팅 결과 시각화
├── 💾 best_hybrid.pth             # 학습된 모델 가중치
└── 📘 README.md                   # 현재 문서
```

---

## 🚀 실행 방법

### 사전 준비
```bash
# 프로젝트 루트에서 전처리 실행 (최초 1회)
python -m preprocessing.pipeline
```
- `data/train_data.csv` (2006-2020) 생성
- `data/test_data.csv` (2021-2025) 생성

**데이터 특징:**
- 38개 Tech 종목 (AAPL, MSFT, GOOGL, AMZN, NVDA 등)
- 20개 Features (기술적 지표 + Fama-French Factors)
- 상장폐지 종목 포함 (생존 편향 제거)

### 1. 모델 학습 (Train Mode)
```bash
cd models/Hybrid_TGNN_DDPG
python run_comparison.py train
```

**학습 과정:**
- 총 800 에피소드 (Early Stopping: 50 에피소드)
- 약 2-3시간 소요 (GPU 권장)
- 학습 완료 시 `best_hybrid.pth` 가중치 저장

**모니터링 지표:**
```
 10/800 | Reward:  537.12 | Return:  1.78% | MDD: 33.29% | Sharpe: 0.368 | Alpha: 0.500
```
- **Reward**: 에피소드별 누적 보상 (높을수록 좋음)
- **Return**: 월평균 수익률 (1.5~2.0% 목표)
- **MDD**: 최대 낙폭 (20~35% 예상, 2008 금융위기 포함)
- **Sharpe Ratio**: 위험 대비 수익 (0.3~0.5 목표)
- **Alpha**: TGNN 가중치 (0.3~0.7 동적 조정)

**디버깅 로그 (초반 3 에피소드):**
```
[DEBUG] Episode 1:
  Values count: 168
  First 5 values: [989751, 1000524, 1020870, 1055177, 1071396]
  Last 5 values: [6626167, 6287831, 6017500, 7028379, 7081968]
  MDD: 37.25%
```

### 2. 백테스팅 (Compare Mode)
```bash
python run_comparison.py compare
```

**비교 전략:**
- **1/N Buy & Hold**: 벤치마크 (균등 가중 매수 후 보유)
- **Hybrid_monthly**: 월간 리밸런싱
- **Hybrid_quarterly**: 분기별 리밸런싱
- **Hybrid_semiannual**: 반기별 리밸런싱
- **Hybrid_annual**: 연간 리밸런싱

**출력 결과:**
```
results/03_Hybrid_TGNN_DDPG/
├── training_logs/                # 학습 진행 차트
│   ├── training_progress_ep50.jpg
│   ├── training_progress_ep100.jpg
│   └── ...
├── hybrid_comparison.jpg         # 전략별 수익률 비교
├── summary_metrics.csv           # 성과 지표 요약
└── hybrid_trade_logs.csv         # 리밸런싱 내역
```

---

## 📊 성과 지표

### 평가 메트릭
- **CAGR (연평균 성장률)**: 복리 수익률
- **MDD (최대 낙폭)**: 최고점 대비 최대 하락률
- **Sharpe Ratio**: 위험 조정 수익률
- **Final Portfolio Value**: 최종 자산 가치

### 예시 결과 (2006-2020 Train Period)
| Metric | Episode 1 | Episode 100 | Episode 800 |
|--------|-----------|-------------|-------------|
| Final Value | $7.1M | $10M | $13M |
| CAGR | 15.2% | 18.5% | 20.4% |
| MDD | 37.3% | 29.1% | 25.2% |
| Sharpe | 0.355 | 0.42 | 0.51 |

### 예시 결과 (2021-2025 Test Period)
| Strategy | CAGR | MDD | Sharpe | Final Value |
|----------|------|-----|--------|-------------|
| Buy & Hold | 24.2% | 23.1% | 0.014 | $2,179,295 |
| Hybrid_monthly | 25.2% | 24.2% | 0.013 | $2,208,945 |

---

## 🔧 하이퍼파라미터

### 학습 설정
```python
# 학습
num_episodes = 800           # 최대 학습 에피소드
early_stopping_patience = 50 # Early Stopping 인내심
batch_size = 64              # 미니배치 크기
replay_buffer_size = 10000   # 경험 버퍼 용량

# 옵티마이저
learning_rate_actor = 1e-4   # Actor 학습률
learning_rate_critic = 1e-3  # Critic 학습률
learning_rate_alpha = 1e-5   # Alpha Network 학습률 (10배 느림)

# DDPG
gamma = 0.99                 # 할인율
tau = 0.001                  # Target 네트워크 업데이트 비율
noise_std = 0.1              # 탐색 노이즈 표준편차

# 정규화
entropy_coef = 0.01          # Entropy 정규화 계수
```

### 모델 구조
```python
# 입력
window_size = 12             # 입력 시계열 길이 (월)
num_features = 20            # 종목별 feature 개수
num_stocks = 38              # 종목 수 (Train)

# TGNN
hidden_dim = 64              # TGNN hidden 차원
num_heads = 4                # Attention head 개수
num_graph_layers = 2         # Graph Convolution 레이어 수

# Actor/Critic
actor_hidden = [128, 64]     # Actor hidden layers
critic_hidden = [128, 64]    # Critic hidden layers
```

### 환경 설정
```python
# 포트폴리오
initial_cash = 1_000_000     # 초기 자본
cost_bps = 0.0005            # 거래 비용 (0.05%)

# 리워드
gamma_crra = 2.0             # CRRA 위험 회피 계수
mdd_penalty = 0.20           # MDD 페널티 시작점 (20%)
concentration_limit = 0.15   # 집중도 제한 (15%)
```

---

## 🔍 주요 특징

### 1. 동적 Alpha 조정
- **MDD 기반 앙상블 가중치**
  - MDD 높을 때 → Alpha 감소 (보수적, 균등 분산 선호)
  - MDD 낮을 때 → Alpha 증가 (공격적, TGNN 신뢰)
- **학습 가능한 Alpha Network**
  - MDD를 입력으로 받아 최적 Alpha 출력
  - 10배 느린 학습률로 안정적 학습

### 2. 그래프 기반 학습
- **Adjacency Matrix 활용**
  - 종목 간 수익률 상관관계 모델링
  - 산업 섹터 정보 반영
- **Graph Convolution**
  - 이웃 종목 정보 집계
  - 시장 전체 트렌드 파악

### 3. 연속 행동 공간
- **DDPG 알고리즘**
  - 이산적 매수/매도가 아닌 연속적 비중 조절
  - Softmax로 정규화된 포트폴리오 가중치 출력
- **탐색 노이즈**
  - Dirichlet 분포 기반 노이즈 (정규화 유지)
  - 초기: 높은 탐색 (noise_std=0.1)
  - 후기: 낮은 탐색 (noise_std=0.01)

### 4. 리밸런싱 유연성
- **다양한 리밸런싱 주기**
  - 월간: 높은 거래 비용, 빠른 반응
  - 분기: 균형잡힌 Trade-off
  - 반기/연간: 낮은 비용, 느린 반응
- **거래 비용 고려**
  - Turnover 페널티 포함
  - 실전 투자 환경 반영

### 5. 생존 편향 제거
- **상장폐지 종목 포함**
  - Momentum1M = nan → 0% 처리
  - TGNN이 위험 패턴 학습
  - 사전 회피 전략 가능

### 6. 안정적 학습
- **Gradient Clipping** (max_norm=1.0)
- **Target Network** (Polyak Averaging)
- **NaN/Inf 체크 및 복구**
- **Replay Buffer** (경험 재사용)

---

## 🐛 알려진 이슈 및 해결

### Issue 1: 학습 중 MDD 0% 표시
**원인**: 최근 12개월만 계산  
**해결**: 전체 에피소드 MDD 계산으로 수정

### Issue 2: Last 5 values [nan, nan, ...]
**원인**: Returns에 nan 포함 (상장폐지)  
**해결**: nan → 0% 처리 (패턴 학습 가능)

### Issue 3: 노이즈 추가 후 재정규화
**원인**: Gaussian 노이즈 추가 후 정규화로 분포 변형  
**해결**: Dirichlet 노이즈 사용 (정규화 유지)

### Issue 4: Entropy 미사용
**원인**: entropy_coef 저장만 하고 사용 안 함  
**해결**: Actor loss에 entropy 페널티 추가

---

## 🛠️ 개선 계획

- [x] MDD 계산 전체 에피소드로 변경
- [x] NaN 처리 로직 개선 (상장폐지 = 0%)
- [x] 디버깅 로그 추가 (초반 3 에피소드)
- [ ] Entropy 정규화 활성화
- [ ] Dirichlet 노이즈로 변경
- [ ] Stop-Loss 기반 긴급 리밸런싱 로직
- [ ] 동적 Alpha 조정 정책 고도화
- [ ] 학습 에피소드 증가 (800 → 1500)
- [ ] 포트폴리오 제약 조건 강화 (최대 비중 15% → 10%)

---

## 📚 참고 문헌

- **TGNN**: [Temporal Graph Networks for Deep Learning on Dynamic Graphs](https://arxiv.org/abs/2006.10637)
- **DDPG**: [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971) (Lillicrap et al., 2015)
- **Portfolio Management**: [Deep Reinforcement Learning for Automated Stock Trading](https://arxiv.org/abs/2011.09607)
- **Ensemble Methods**: [Dynamic Portfolio Optimization with Deep Reinforcement Learning](https://arxiv.org/abs/2004.06626)

---

## 📧 Contact
문의사항은 프로젝트 Repository Issues에 등록해 주세요.

---

## 🔄 Version History

### v1.2.0 (2025-12-19)
- 학습 안정성 개선 (nan 처리, MDD 계산 수정)
- 디버깅 로그 추가
- README 업데이트

### v1.1.0 (2024-12-XX)
- Alpha Network 학습률 분리
- Early Stopping 추가

### v1.0.0 (2024-12-XX)
- 초기 Hybrid TGNN-DDPG 구현
