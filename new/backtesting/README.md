# SMS Backtesting Framework (Research Edition)

> 🔬 **학술 논문용**으로 설계된 하이브리드(TGNN + DDPG) 주가 예측 및 포트폴리오 최적화 프레임워크

## 🌟 핵심 기능

| 기능 | 설명 |
|------|------|
| **Hybrid Composition** | TGNN + DDPG 인스턴스를 조합한 앙상블 (코드 중복 제거) |
| **Horizon Matching** | 예측 주기(1M/3M/6M/12M)와 리밸런싱 주기 일치 |
| **날짜 기반 리밸런싱** | 월초/분기초 등 캘린더 기반 리밸런싱 |
| **Fama-French 5-Factor** | Mkt_RF, SMB, HML, RMW, CMA |

---

## 📦 디렉토리 구조

```
backtesting/
├── config/config.yaml   # 실험 설정
├── src/
│   ├── models/
│   │   ├── tgnn/        # 시계열 그래프 신경망
│   │   ├── ddpg/        # 강화학습 Actor-Critic
│   │   └── hybrid/      # TGNN + DDPG Composition
│   ├── backtest/        # 백테스팅 엔진
│   └── pipelines/       # 학습/백테스팅 파이프라인
├── main.py              # 진입점
└── README.md
```

---

## 🚀 실행 방법

### 1. 전체 실행 (Train + Backtest)
```bash
python main.py --mode full
```

### 2. 개별 실행
```bash
# 전처리
python main.py --mode preprocess

# 학습만
python main.py --mode train

# 백테스팅만
python main.py --mode backtest

# 비교 차트 생성
python main.py --mode compare
```

### 3. 모델 선택
`config/config.yaml`에서 설정:
```yaml
project:
  selected_model: "ALL"  # "tgnn", "ddpg", "hybrid", "ALL"
```

---

## ⚙️ 주요 설정

```yaml
model:
  softmax_temperature: 10.0    # 포트폴리오 분산도
  hybrid_alpha_min: 0.2        # TGNN 비중 최소
  hybrid_alpha_max: 0.8        # TGNN 비중 최대

training:
  episodes: 800
  buffer_size: 10000

backtest:
  risk_free_rate: 0.02         # Sharpe Ratio용
  transaction_cost: 0.001      # 0.1%
```

---

## 📊 결과

### 모델별 결과 (`results/{model_name}/`)
- `plots/comparison.png`: 누적 수익률 비교
- `logs/trade_logs.csv`: 거래 내역
- `logs/backtest_metrics.csv`: 성과 지표 (CAGR, Sharpe, MDD)
- `checkpoints/`: 모델 가중치

### 비교 차트 (`results/comparison/`)
- `cagr_comparison_all.png`: CAGR 비교 차트
- `sharpe_comparison_all.png`: Sharpe Ratio 비교
- `risk_return_scatter_all.png`: 리스크-수익 산점도
- `all_strategies_comparison.csv`: 전체 전략 요약

---

## 🧪 연구 무결성

- **Seed 고정**: `config.yaml` → `seed: 42`
- **Look-ahead Bias 제거**: 테스트 데이터로 파인튜닝 금지
- **공정한 벤치마크**: 1/N Buy&Hold와 동일 조건 비교
