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
| **Fine-tune** | 2021 ~ 2025 초반 | 출력 레이어 적응 | test_data.csv (50 에피소드) |
| **Test** | 2021 ~ 2025 (5년) | Out-of-Sample 백테스팅 | test_data.csv |

### 전이 학습 (Transfer Learning)
- Train 종목 수와 Test 종목 수가 다를 경우 **TGNN Encoder 레이어만 로드**
- 종목 수에 의존하지 않는 시계열 패턴 학습을 재활용
- Fine-tuning 단계에서 출력 레이어(Actor, Alpha Network)만 테스트 데이터로 재학습

---

## 🧠 모델 아키텍처

### 1. Input (State Space)
- **Node Features**: `(Batch, N_stocks, Window=12, N_features=20+섹터)`
  - 기술적 지표: Close, Volume, RSI, MACD, Momentum (1M/3M/6M/12M), Volatility
  - 팩터: Value, Beta, Momentum, Volatility
  - Fama-French 5 Factors: Mkt-RF, SMB, HML, RMW, CMA
  - Sector One-Hot Encoding (산업군별 특성 반영)
- **Adjacency Matrix**: `(Batch, N_stocks, N_stocks)` - 종목 간 수익률 상관관계

### 2. Network Flow
```
Input Features → TGNN Encoder → Node Embeddings
                                       ↓
                            Alpha Network (α 계산)
                                       ↓
        TGNN Weights ←─ α ─→ 1/N Weights
                       ↓
              Ensemble Portfolio (Action)
```

**핵심 구성 요소:**
1. **Graph Convolution Layer**: 종목 간 정보 전파 (이웃 노드 집계)
2. **Temporal Attention**: 과거 12개월 중 중요 시점 자동 선택
3. **Alpha Network**: 현재 MDD 기반 동적 앙상블 비율 결정
4. **Actor Network**: Softmax 출력으로 포트폴리오 비중 생성 (합=1)

### 3. Output (Action Space)
- **Portfolio Weights**: `(N_stocks,)` - 각 종목 투자 비중 (합=1, 최대 25%)
- **Alpha Value**: `[0, 1]` - TGNN 신뢰도 (0=보수적, 1=공격적)

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
├── 💾 best_hybrid.pth             # 학습된 모델 가중치 (Train)
├── 💾 best_hybrid_finetuned.pth  # Fine-tuned 모델 가중치 (Test)
└── 📘 README.md                   # 현재 문서
```

---

## 🚀 실행 방법

### 사전 준비
```bash
# 프로젝트 루트에서 전처리 실행 (최초 1회)
python -m preprocessing.pipeline
```
- `data/train_data.csv` (2006-2020, 25개 종목)
- `data/test_data.csv` (2021-2025, 7개 종목) 생성

### 1. 모델 학습 (Train Mode)
```bash
cd models/Hybrid_TGNN_DDPG
python run_comparison.py train
```

**학습 과정:**
- 최대 800 에피소드 (Early Stopping: 100 에피소드 patience)
- 배치 크기: 64
- 학습 시간: 약 1-2시간 (GPU 권장)
- 50 에피소드마다 `training_progress_ep{N}.jpg` 자동 저장
- 최고 성과 달성 시 `best_hybrid.pth` 가중치 저장

**Early Stopping:**
- 100 에피소드 동안 성능 개선 없으면 자동 종료
- Best Reward 기록 및 해당 에피소드 번호 출력

**모니터링 지표:**
- Episode Reward: 에피소드별 누적 보상
- Average Return (%): 평균 수익률
- Maximum Drawdown (%): 최대 낙폭
- Sharpe Ratio: 위험 대비 수익
- Ensemble Alpha: TGNN 가중치
- No Improve Count: Early Stopping 카운터

### 2. 백테스팅 (Compare Mode)
```bash
python run_comparison.py compare
```

**실행 단계:**
1. **전이 학습**: Train 모델의 TGNN Encoder만 로드
2. **Fine-tuning**: Test 데이터로 출력 레이어 50 에피소드 재학습
3. **백테스트**: 5가지 리밸런싱 전략 비교

**비교 전략:**
- **1/N Buy & Hold** (벤치마크): 초기 균등 분산 후 홀딩
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
- **Sharpe Ratio**: 위험 조정 수익률 (초과 수익 / 변동성)
- **Final Portfolio Value**: 최종 자산 가치

### 예시 결과 (2021-2025 Test Period)
| Strategy | CAGR | MDD | Sharpe | Final Value |
|----------|------|-----|--------|-------------|
| Buy & Hold | 24.2% | 23.1% | 0.014 | $2,179,295 |
| Hybrid_monthly | 25.2% | 24.2% | 0.013 | $2,208,945 |
| Hybrid_quarterly | 24.8% | 23.5% | 0.012 | $2,195,120 |

---

## 🔧 하이퍼파라미터

### 학습 설정
```python
# 에피소드 및 Early Stopping
MAX_EPISODES = 800                    # 최대 학습 에피소드
EARLY_STOPPING_PATIENCE = 100         # Early Stopping patience
BATCH_SIZE = 64                       # 미니배치 크기
REPLAY_BUFFER_SIZE = 10000            # 경험 버퍼 용량
MIN_BUFFER_SIZE = 512                 # 학습 시작 최소 버퍼 크기

# 탐색 노이즈 (동적 감소)
INITIAL_NOISE_STD = 0.3               # 초기 탐색 노이즈
MIN_NOISE_STD = 0.05                  # 최소 탐색 노이즈
NOISE_DECAY_RATE = (0.3 - 0.05) / 800 # 에피소드별 감소율

# 옵티마이저 학습률
LEARNING_RATE_ACTOR = 1e-4            # Actor 학습률
LEARNING_RATE_CRITIC = 1e-3           # Critic 학습률
LEARNING_RATE_ALPHA = 5e-5            # Alpha Network 학습률

# DDPG 하이퍼파라미터
GAMMA = 0.99                          # 할인율
TAU = 0.001                           # Target 네트워크 Soft Update 비율
ENTROPY_COEF = 0.01                   # Entropy 정규화 계수
```

### Fine-tuning 설정
```python
FINETUNE_EPISODES = 50                # Fine-tuning 에피소드
FINETUNE_BATCH_SIZE = 32              # Fine-tuning 배치 크기
FINETUNE_BUFFER_SIZE = 128            # Fine-tuning 버퍼 크기
FINETUNE_NOISE_STD = 0.1              # Fine-tuning 탐색 노이즈
```

### 모델 구조
```python
WINDOW_SIZE = 12                      # 입력 시계열 길이 (월)
NUM_FEATURES = 20 + N_sectors         # 종목별 feature 개수 + 섹터
HIDDEN_DIM = 64                       # TGNN hidden 차원
NUM_HEADS = 4                         # Attention head 개수
```

### 포트폴리오 제약
```python
INITIAL_CAPITAL = 1,000,000           # 초기 자본
MAX_CONCENTRATION = 0.25              # 개별 종목 최대 비중 (25%)
```

---

## 🔍 주요 특징

### 1. 동적 Alpha 조정
- **MDD 높을 때** → Alpha 감소 (보수적, 균등 분산 선호)
- **MDD 낮을 때** → Alpha 증가 (공격적, TGNN 신뢰)
- Alpha Network가 시장 상황에 따라 자동으로 앙상블 비율 조정

### 2. 그래프 기반 학습
- 단순 시계열이 아닌 **종목 간 관계**를 모델링
- 산업 섹터, 수익률 상관관계를 adjacency matrix로 표현
- Graph Convolution으로 이웃 종목 정보 전파

### 3. 연속 행동 공간
- 이산적 매수/매도가 아닌 **연속적 비중 조절**
- DDPG 알고리즘으로 실전 투자에 가까운 포트폴리오 관리
- Softmax 출력으로 비중 합=1 보장

### 4. 리밸런싱 유연성
- 월간/분기/반기/연간 리밸런싱 빈도 선택 가능
- 거래 비용 vs 수익 최적화 trade-off 분석
- 각 빈도별 성과 지표 비교 제공

### 5. 전이 학습 (Transfer Learning)
- Train 종목(25개)과 Test 종목(7개) 수가 달라도 학습 전이 가능
- **Encoder 레이어만 로드**: 시계열 패턴 학습 재활용
- **Fine-tuning**: 출력 레이어를 새로운 종목에 적응

### 6. Early Stopping
- 과적합 방지를 위한 조기 종료 메커니즘
- 100 에피소드 동안 개선 없으면 자동 중단
- 최고 성과 모델 자동 저장

### 7. Sector-aware Features
- 산업군별 One-Hot Encoding 추가
- 섹터 특성을 반영한 종목 간 관계 학습
- 섹터 로테이션 효과 포착

---

## 📚 참고 문헌

- **TGNN**: Temporal Graph Networks for Deep Learning on Dynamic Graphs
- **DDPG**: Continuous Control with Deep Reinforcement Learning (Lillicrap et al., 2015)
- **Portfolio Management**: Deep Reinforcement Learning for Automated Stock Trading
- **Transfer Learning**: How transferable are features in deep neural networks? (Yosinski et al., 2014)

---

## 🛠️ 개선 계획

- [x] Early Stopping 구현 (100 에피소드 patience)
- [x] Fine-tuning 파이프라인 구축
- [x] Sector One-Hot Encoding 추가
- [x] Dynamic Noise Decay 구현
- [ ] MDD 페널티를 Reward Function에 직접 추가
- [ ] Stop-Loss 기반 긴급 리밸런싱 로직
- [ ] 학습 에피소드 증가 (800 → 1000)
- [ ] 포트폴리오 제약 조건 강화 (최대 비중 25% → 20%)
- [ ] Multi-GPU 분산 학습 지원

---

## 🐛 문제 해결

### Q: "No such file or directory: best_hybrid.pth"
**A:** 먼저 `python run_comparison.py train` 명령으로 모델을 학습해야 합니다.

### Q: GPU 메모리 부족 에러
**A:** `Config` 클래스에서 `BATCH_SIZE`를 64 → 32로 줄이거나, `REPLAY_BUFFER_SIZE`를 10000 → 5000으로 조정하세요.

### Q: Fine-tuning 없이 바로 테스트하고 싶어요
**A:** `run_comparison.py`의 `run_comparison_mode()` 메서드에서 `self.trainer.finetune()` 줄을 주석 처리하세요.

### Q: Train/Test 종목 수가 다른데 괜찮나요?
**A:** 네, 전이 학습 메커니즘이 자동으로 처리합니다. Encoder만 로드되고 출력 레이어는 새로 초기화됩니다.

---

## 📧 Contact
문의사항은 [GitHub Repository Issues](https://github.com/sehyun00/SMS_Backtesting/issues)에 등록해 주세요.
