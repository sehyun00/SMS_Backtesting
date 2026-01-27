# 🛠️ SMS Backtesting - Data Preprocessing Module

이 모듈은 **TGNN-DDPG 하이브리드 포트폴리오 최적화** 연구를 위한 데이터 전처리 파이프라인입니다.

S&P 500 종목의 과거 데이터를 수집하고, 기술적 지표 및 팩터 점수를 계산하며, **생존 편향(Survivorship Bias)**을 고려한 섹터별 Train/Test 데이터셋 분할을 수행합니다.

---

## 📂 디렉토리 구조

```
preprocessing/
├── pipeline.py              # 전체 파이프라인 통합 실행 (Main Entry Point)
├── data_collector.py        # Wikipedia 크롤링 + yfinance 데이터 수집
├── data_splitter.py         # 섹터 기반 Train/Test 데이터 분할
├── data_processor.py        # 지표 및 팩터 계산 통합
├── indicators.py            # 기술적 지표 (RSI, MACD, Momentum, Volatility)
├── factors.py               # 팩터 점수 및 매수/매도 시그널 생성
├── fama_french_loader.py    # Kenneth French Library 5-Factor 로드
└── README.md                # 본 문서
```

---

## 🚀 주요 전처리 전략

### 1. 생존 종목 필터링 (Survivor Filtering)

**TGNN(시계열 그래프 신경망)**의 구조적 안정성을 위해 **노드(종목)의 개수를 고정**해야 합니다.

**전략:**
- Wikipedia에서 S&P 500 구성 종목 자동 크롤링
- **2006년부터 현재까지 전 기간 상장 유지된 종목**만 필터링
- 각 섹터에서 생존한 종목들을 Train/Test 그룹으로 분리

**이유:**
- 그래프 인접 행렬(Adjacency Matrix)의 크기 N×N를 고정하여 학습 안정성 확보
- 중간에 노드가 사라지거나 생길 경우 발생하는 Temporal Attention 연산 오류 방지

> ⚠️ 생존 편향(Survivorship Bias)이 존재하지만, 그래프 신경망 구조 구현을 위한 현실적인 타협점입니다.

### 2. 시간 및 섹터 기반 데이터 분할

**기간 분할:**
| 기간 | 용도 | 연도 |
|------|------|------|
| Train | 모델 학습 | 2006-2020 (약 15년) |
| Test | 백테스트 검증 | 2021-현재 (약 5년) |

**섹터별 종목 선택:**
```python
# Train: 섹터당 최대 5개 종목 선택
train_per_sector = 5

# Test: 총 7개 종목 선택 (섹터 균형 고려)
test_total = 7
```

**데이터 품질 필터:**
- Train 종목: 최소 1,000행 이상 (약 4년치 데이터)
- Test 종목: 최소 600행 이상 (약 2.5년치 데이터)

### 3. 기술적 지표 및 팩터

**Technical Indicators** (`indicators.py`):
| 지표 | 설명 |
|------|------|
| Momentum 1M/3M/6M/12M | 기간별 수익률 |
| Volatility | 표준편차 기반 변동성 |
| RSI | Relative Strength Index |
| MACD | Moving Average Convergence Divergence |

**Factor Scoring** (`factors.py`):
| 팩터 | 설명 |
|------|------|
| Value Factor | 가치 평가 점수 |
| Beta Factor | 시장 민감도 |
| Momentum Factor | 모멘텀 점수 |
| Volatility Factor | 변동성 점수 |
| Weighted Score | 가중 합계 |
| Signal | 매수/매도 시그널 |

**Market Factors** (`fama_french_loader.py`):
- Kenneth French Data Library의 **Fama-French 5 Factors** 병합
  - Mkt-RF: 시장 리스크 프리미엄
  - SMB: Small Minus Big
  - HML: High Minus Low
  - RMW: Robust Minus Weak
  - CMA: Conservative Minus Aggressive

---

## 💻 사용 방법

### 1. 자동 실행 (S&P 500 크롤링)

```bash
cd c:\Project\SMS_Backtesting
python -m new.backtesting.src.preprocessing.pipeline
```

### 2. CSV 파일 지정 실행

```bash
python -m new.backtesting.src.preprocessing.pipeline --csv data/stock_list.csv
```

### 3. 출력 디렉토리 지정

```bash
python -m new.backtesting.src.preprocessing.pipeline --output results/
```

---

## 📊 실행 과정

```
🚀 전처리 파이프라인 시작
============================================================

📡 S&P 500 종목 자동 크롤링 모드
✅ 503개 종목 크롤링 완료

🔍 생존 종목 필터링 중 (2006-2025)...
  (1/503) ✅ AAPL: Train=3754, Test=1008
  (2/503) ✅ MSFT: Train=3754, Test=1008
  ...
   ✅ Train 후보: 285개
   ✅ Test 후보: 312개

📊 Train 데이터 수집 중 (285개 종목, 2006-2020)...
📊 Test 데이터 수집 중 (312개 종목, 2021-현재)...

✂️ 섹터별 종목 선택 중...

[Train] 섹터당 최대 5개
   [Communication Services         ]  5개
   [Consumer Discretionary         ]  5개
   [Consumer Staples               ]  5개
   [Energy                         ]  5개
   [Financials                     ]  5개
   [Health Care                    ]  5개
   [Industrials                    ]  5개
   [Information Technology         ]  5개
   [Materials                      ]  5개
   [Real Estate                    ]  5개
   [Utilities                      ]  5개

✅ 최종 선택:
   Train: 55개 종목, 206,234행
   Test:  7개 종목, 8,764행

💾 저장 완료: data/train_data.csv (206,234행)
💾 저장 완료: data/test_data.csv (8,764행)

============================================================
✅ 파이프라인 완료!
```

---

## 📁 출력 파일 구조

| 파일 | 기간 | 용도 |
|------|------|------|
| `data/train_data.csv` | 2006-2020 | 모델 학습 |
| `data/test_data.csv` | 2021-현재 | 백테스트 검증 |

**컬럼 구조:**
```
Date, Symbol, Sector, Industry,
Open, High, Low, Close, Volume,
Momentum1M, Momentum3M, Momentum6M, Momentum12M,
volatility, rsi, macd, macd_signal,
value_factor, beta_factor, momentum_factor, volatility_factor,
weighted_score, signal,
Mkt-RF, SMB, HML, RMW, CMA, RF
```

---

## 🔧 커스터마이징

### 섹터당 종목 수 조정

`pipeline.py`의 `split_by_sector()` 호출 수정:
```python
final_train_df, final_test_df, train_symbols, test_symbols = (
    splitter.split_by_sector(
        train_per_sector=10,  # 섹터당 10개로 증가
        test_total=15,        # 테스트 15개로 증가
    )
)
```

### 기간 조정

`Pipeline` 클래스 초기화 시 인자 변경:
```python
pipeline = Pipeline(
    start_year=2010,  # 2010년부터
    end_year=2025
)
```

### 타겟 지수 변경 (NASDAQ 100)

`pipeline.py`에서 수정:
```python
self.collector.load_stocks_auto(target="nasdaq100")
```

---

## 📦 의존성

```
pandas>=2.0.0
numpy>=1.24.0
yfinance>=0.2.28
requests
beautifulsoup4
lxml
pandas-datareader>=0.10.0
```

---

## 🔍 문제 해결

### Q: "종목 테이블을 찾을 수 없습니다" 에러

**A:** Wikipedia 페이지 구조가 변경되었을 수 있습니다. `data_collector.py`의 테이블 파싱 로직을 확인하세요.

### Q: yfinance 다운로드가 느립니다

**A:** S&P 500은 500개 종목이므로 30-60분 소요됩니다. 인터넷 연결 상태를 확인하세요.

### Q: Fama-French 다운로드가 실패합니다

**A:** Kenneth French Library 서버 상태를 확인하세요. 실패해도 파이프라인은 계속 진행됩니다.

---

## 📚 참고 자료

- [Yahoo Finance API (yfinance)](https://github.com/ranaroussi/yfinance)
- [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
- [Wikipedia S&P 500](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)

---

## ⚠️ 주의사항

- 이 전처리 파이프라인은 **연구 목적**으로 설계되었습니다.
- **생존 편향(Survivorship Bias)**이 포함되어 있어 실제 투자 성과와 다를 수 있습니다.
- 과거 데이터 기반 백테스팅은 미래 수익을 보장하지 않습니다.
