# SMS Backtesting Framework (Research Edition)

이 프로젝트는 **Temporal Graph Neural Networks (TGNN)** 및 **Deep Reinforcement Learning (DDPG)** 을 활용한 금융 시계열 예측 프레임워크입니다.
학술 연구의 **재현성(Reproducibility)**과 **확장성(Extensibility)**을 보장하기 위해 모듈화된 설계를 따릅니다.

## 🌟 주요 기능 (Key Features)

*   **Clean Architecture**: 데이터 수집, 전처리, 모델링, 학습, 백테스팅이 명확히 분리된 모듈 구조.
*   **Advanced AI Models**:
    *   **TGNN**: 시계열(Temporal)과 종목 간 관계(Graph)를 동시에 학습.
    *   **DDPG**: 연속적인 포트폴리오 비중 조절을 위한 강화학습 에이전트.
    *   **Hybrid**: 지도학습 Encoder와 RL Agent의 결합.
*   **Smart Transfer Learning**: 대규모 종목(55개)으로 학습된 모델을 소수 종목(10개)에 적용 시 자동으로 감지하여 **부분 가중치 로드(Partial Load)** 및 **미세 조정(Fine-tuning)**을 수행합니다.
*   **Realistic Backtesting**: 거래 비용을 고려한 시뮬레이션 및 월간/분기별 리밸런싱 전략 지원.

## 📂 디렉토리 구조 (Directory Structure)

각 디렉토리의 상세 문서는 해당 폴더의 `README.md`를 참조하세요.

| 경로 | 설명 |
|---|---|
| [`src/preprocessing/`](src/preprocessing/README.md) | 데이터 수집(Crawling), 보조지표 계산, Train/Test 분할 |
| [`src/models/`](src/models/README.md) | TGNN, DDPG, Hybrid 모델 아키텍처 정의 |
| [`src/training/`](src/training/README.md) | Dataset 생성, Trainer(지도학습), RL Trainer(강화학습) |
| [`src/backtest/`](src/backtest/README.md) | 전략 핸들러, 백테스트 엔진, 성과 지표 계산, 시각화 |
| [`src/pipelines/`](src/pipelines/README.md) | 학습(`run_train`) 및 백테스트(`run_backtest`) 통합 워크플로우 |
| `config/` | 하이퍼파라미터 및 데이터 설정을 관리하는 `config.yaml` |

## 🚀 시작하기 (Getting Started)

### 1. 전제 조건 (Prerequisites)
```bash
pip install -r requirements.txt
```

### 2. 실행 (Execution)

모든 작업은 `main.py`를 통해 통합 실행됩니다.

#### 1단계: 데이터 전처리
S&P 500 / NASDAQ 데이터를 수집하고 전처리하여 `data/`에 저장합니다.
```bash
python new/backtesting/main.py --mode preprocess
```

#### 2단계: 모델 학습
`config.yaml`에 설정된 모델(기본값: TGNN/DDPG)을 학습합니다.
```bash
python new/backtesting/main.py --mode train
```

#### 3단계: 백테스트 및 검증
학습된 모델을 로드하여 Test Set(2021~)에 대해 검증합니다.
*   **Tip**: `config.yaml`에서 `stock_universes`를 변경하면, 자동으로 전이 학습 모드로 진입합니다.
```bash
python new/backtesting/main.py --mode backtest
```

## ⚙️ 설정 (Configuration)

`config/config.yaml` 파일에서 주요 실험 변수를 제어합니다.

```yaml
project:
  selected_model: "ddpg"  # tgnn, ddpg, hybrid

data:
  window_size: 12        # 입력 시퀀스 길이
  stock_universes: []    # 비워두면 학습된 전체 유니버스 사용, 입력 시 해당 종목만 테스트

training:
  episodes: 50           # 학습 에폭 수
  learning_rate: 0.001
```

## 📊 결과 확인

실행 결과는 `results/{model_type}/` 디렉토리에 저장됩니다.
*   `best_model_*.pth`: 학습된 모델 가중치
*   `trade_logs.csv`: 날짜별 거래 내역
*   `comparison.png`: 벤치마크 대비 누적 수익률 그래프
*   `trained_universe.json`: 학습에 사용된 종목 리스트
