# Hybrid TGNN-DDPG Model

## 📖 개요
본 모델은 **TGNN (Temporal Graph Neural Network)**의 시공간적 패턴 인식 능력과 **DDPG (Deep Deterministic Policy Gradient)**의 연속적 포트폴리오 제어 능력을 결합한 하이브리드 알고리즘입니다.

- **TGNN (State Encoder)**: 종목 간의 관계(Correlation, Industry)와 시계열 트렌드를 분석하여 고차원 `Node Embedding`을 생성합니다.
- **DDPG (RL Agent)**: 생성된 임베딩을 기반으로 **CRRA 효용 함수**를 최대화하는 최적의 자산 배분 비중을 결정합니다.

---

## 🔬 학습 및 검증 방법: 3년 학습 / 8년 실전 투자
본 프로젝트의 엄격한 검증 표준을 따릅니다.

| 구분 | 기간 | 역할 |
|------|------|------|
| **학습 (Train)** | **2015 ~ 2017** (3년) | 하이브리드 에이전트(TGNN+DDPG) 학습 |
| **테스트 (Test)** | **2018 ~ 2025** (8년) | 실전 투자 시뮬레이션 (Out-of-Sample) |

---

## 🧠 모델 구조 (Architecture)

### 1. State Space (입력)
단순한 팩터 나열이 아닌, **그래프 구조**를 입력받습니다.
- **Features**: `(Batch, 10 Stocks, 12 Months, 11 Features)`
- **Adjacency Matrix**: `(Batch, 10 Stocks, 10 Stocks)` (종목 간 상관관계 그래프)

### 2. Hybrid Network Flow
1. **Graph Convolution**: 인접 행렬을 활용해 종목 간 정보 교환
2. **Temporal Attention**: 과거 12개월 중 중요한 시점에 가중치 부여
3. **Node Embeddings**: 각 종목의 "시장 내 위치"와 "상태"를 함축한 벡터 생성
4. **Policy Network (Actor)**: 임베딩을 종합하여 최종 포트폴리오 비중(`Softmax`) 출력

---

## 📂 파일 구조
```
models/Hybrid_TGNN_DDPG/
├── model.py                 # HybridActor, HybridCritic, TGNNEncoder 정의
├── run_comparison.py        # 학습 및 백테스팅 실행 스크립트
├── visualization.py         # 결과 시각화 모듈
└── README.md                # 현재 파일
```

## 🚀 실행 방법

### 1. 모델 학습 (Train)
2015~2017년 데이터로 Hybrid 에이전트를 학습시킵니다.

```bash
python models/Hybrid_TGNN_DDPG/run_comparison.py train
```

- 학습이 완료되면 `best_hybrid.pth` 파일이 생성됩니다.

### 2. 백테스팅 비교 (Compare)
2018~2025년 데이터에 대해 실전 투자를 시뮬레이션합니다.

```bash
python models/Hybrid_TGNN_DDPG/run_comparison.py compare
```

- **결과물**: `results/03_Hybrid_TGNN_DDPG/` 디렉토리에 저장됩니다.
  - `hybrid_comparison.png`: 리밸런싱 빈도별 수익률 그래프
  - `summary_metrics.csv`: CAGR, MDD, Sharpe 등 상세 지표
