# Training Module (모델 학습)

`src/training`은 데이터셋 생성, 학습 루프 실행, 그리고 강화학습(RL) 환경을 제공하는 모듈입니다.
학술 연구의 표준을 따르기 위해 **지도 학습(Supervised)**과 **강화 학습(RL)** 파이프라인이 분리되어 있으며, **전이 학습(Transfer Learning)**을 위한 전용 메서드를 지원합니다.

## 📦 모듈 구조

| 모듈 | 역할 |
|---|---|
| `dataset.py` | 시계열 데이터를 슬라이딩 윈도우 방식으로 변환하여 PyTorch Dataset 생성 |
| `trainer.py` | TGNN 및 지도 학습 모델을 위한 표준 학습 루프 (Train & Finetune) |
| `rl_trainer.py` | DDPG 등 강화학습 에이전트를 위한 전용 학습 루프 |
| `environment.py` | 강화학습을 위한 포트폴리오 관점의 Trading Environment (OpenAI Gym 스타일 인터페이스) |
| `replay_buffer.py` | 강화학습 경험 리플레이 메모리 (Experience Replay Buffer) |

## 🏗️ 데이터 파이프라인

### 1. FinancialDataset (`dataset.py`)
시계열 데이터를 신경망(TGNN/LSTM)이 이해할 수 있는 텐서 형태로 변환합니다.

*   **입력**: 전처리된 Pandas DataFrame (OHLCV + Technical Indicators + Factors)
*   **출력 (Item Shape)**:
    *   `features`: $[N, T, F]$
        *   $N$: 종목 수 (Nodes)
        *   $T$: 윈도우 크기 (Time Steps, default: 12)
        *   $F$: 특징 개수 (Features)
    *   `adj_matrix`: $[N, N]$ (섹터 기반 인접 행렬)
    *   `labels`: $[N, 4]$ (예측 대상: Momentum 1M, 3M, 6M, 12M)

### 2. Trainer (`trainer.py`)
지도 학습 모델(TGNN, Generic)의 수명 주기를 관리합니다.

*   **주요 메서드**:
    *   `train()`: 전체 데이터셋에 대해 학습을 수행하고 Checkpoint를 저장합니다.
    *   `finetune(epochs, lr_factor)`: **전이 학습(Transfer Learning)**을 위해, 사전 학습된 모델을 새로운 데이터셋(예: 다른 종목 유니버스)에 맞춰 미세 조정합니다. 중복 로직 방지를 위해 `_run_epoch()`를 공유합니다.

### 3. RL Trainer (`rl_trainer.py`)
강화학습 에이전트(DDPG, Hybrid)를 학습합니다.
*   **Process**: Environment $\leftrightarrow$ Agent 상호작용을 통해 Replay Buffer에 경험을 쌓고, 배치 단위로 학습합니다.

## 🚀 사용법 (Usage)

### Trainer 초기화 및 실행
```python
from src.training.dataset import FinancialDataset
from src.training.trainer import Trainer

# 1. 데이터셋 생성
dataset = FinancialDataset(config, train_df, mode="train")

# 2. 트레이너 초기화
trainer = Trainer(config, model, dataset)

# 3. 학습 시작
trainer.train()

# (옵션) 전이 학습 (Fine-tuning)
# 새로운 데이터셋으로 미세 조정
trainer.finetune(epochs=50, lr_factor=0.1)
```

## ⚙️ 설정 의존성 (`config.yaml`)
이 모듈은 `config.yaml`의 다음 항목들을 참조합니다:
- `data.window_size`: 입력 시퀀스 길이 ($T$)
- `data.stock_universes`: 학습에 사용할 종목 리스트 ($N$)
- `training.batch_size`: 배치 크기
- `training.learning_rate`: 학습률
- `training.episodes`: 총 에폭(Epoch) 수
