# 🛠️ SMS Backtesting - Data Preprocessing Module

이 모듈은 **TGNN-DDPG 하이브리드 포트폴리오 최적화** 연구를 위한 데이터 전처리 파이프라인입니다.

나스닥 100(NASDAQ 100) 종목의 과거 데이터를 수집하고, 기술적 지표 및 팩터 점수를 계산하며, **생존 편향(Survivorship Bias)**을 고려한 섹터별 Train/Test 데이터셋 분할을 수행합니다.

---

## 📂 디렉토리 구조

```
preprocessing/
├── __init__.py                  # 패키지 초기화
├── pipeline.py                  # 전체 파이프라인 통합 실행 (Main Entry Point)
├── data_collector.py            # yfinance를 통한 주가 데이터 수집 및 생존 종목 필터링
├── technical_indicators.py      # 기술적 지표 계산 (RSI, MACD, Momentum, Volatility 등)
├── factor_calculator.py         # 팩터 점수 산출 및 매수/매도 시그널 생성
├── fama_french_loader.py        # Kenneth French Library에서 5-Factor 데이터 다운로드 및 병합
├── data_splitter.py             # 섹터 기반 Train/Test 데이터 분할 (핵심)
└── requirements.txt             # 필요 라이브러리 목록
```

---

## 🚀 주요 전처리 전략

### 1. 생존 종목 필터링 (Survivor Filtering)

**TGNN(시계열 그래프 신경망)**의 구조적 안정성을 위해 **노드(종목)의 개수를 고정**해야 합니다.

**전략:**
- 2025년 현재 나스닥 100 리스트 중, **2006년부터 2025년까지 전 기간 상장 유지된 종목**만 필터링
- 각 섹터에서 생존한 종목들을 Train/Test 그룹으로 분리

**이유:**
- 그래프 인접 행렬(Adjacency Matrix)의 크기 \(N \times N\)를 고정하여 학습 안정성 확보
- 중간에 노드가 사라지거나 생길 경우 발생하는 Temporal Attention 연산 오류 방지
- _Note: 생존 편향(Survivorship Bias)이 존재하지만, 그래프 신경망 구조 구현을 위한 현실적인 타협점입니다._

### 2. 시간 및 섹터 기반 데이터 분할

**기간 분할:**
- **Train Period:** 2006-01-01 ~ 2020-12-31 (약 15년)
- **Test Period:** 2021-01-01 ~ 2025-12-31 (약 5년)

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

**목적:**
- 모델이 특정 종목의 패턴만 외우는 것을 방지
- 섹터의 일반적인 시장 역학을 학습했는지 검증
- Out-of-sample 테스트를 통한 일반화 성능 평가

### 3. 기술적 지표 및 팩터

**Technical Indicators** (`technical_indicators.py`):
- Momentum: 1개월, 3개월, 6개월, 12개월 수익률
- Volatility: 표준편차 기반 변동성
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)

**Factor Scoring** (`factor_calculator.py`):
- Value, Beta, Momentum, Volatility 기반 가중치 점수
- 매수/매도 시그널 생성 (Strong Buy ~ Strong Sell)

**Market Factors** (`fama_french_loader.py`):
- Kenneth French Data Library의 **Fama-French 5 Factors** 병합
  - Mkt-RF (Market Risk Premium)
  - SMB (Small Minus Big)
  - HML (High Minus Low)
  - RMW (Robust Minus Weak)
  - CMA (Conservative Minus Aggressive)

---

## 💻 사용 방법

### 1. 사전 준비

프로젝트 루트 디렉토리의 `data/` 폴더에 나스닥 100 종목 리스트 파일이 있어야 합니다:

```
SMS_Backtesting/
└── data/
    └── nasdaq100_stock_list.csv
```

**필수 컬럼:**
- `Symbol`: 종목 티커 (예: AAPL, MSFT)
- `Sector`: 산업 섹터 (예: Technology, Healthcare)
- `Industry`: (선택) 세부 산업

### 2. 라이브러리 설치

```bash
cd preprocessing
pip install -r requirements.txt
```

### 3. 파이프라인 실행

프로젝트 루트 디렉토리에서 모듈로 실행:

```bash
# 기본 실행 (data/nasdaq100_stock_list.csv 사용)
python -m preprocessing.pipeline

# 커스텀 경로 지정
python -m preprocessing.pipeline --csv data/custom_list.csv --output results/
```

**실행 과정:**
1. 생존 종목 필터링 (2006-2025 전기간 상장 유지)
2. Train 데이터 수집 (2006-2020)
3. Test 데이터 수집 (2021-2025)
4. 기술적 지표 및 팩터 계산
5. Fama-French 5 Factor 데이터 병합
6. 섹터별 종목 선택 및 분할
7. CSV 파일 저장

### 4. 결과 확인

실행 완료 후 `data/` 디렉토리에 생성되는 파일:

- **`train_data.csv`**: 2006-2020년 학습용 데이터
  - 섹터당 최대 5개 종목 × 약 15년 = 수만 행
- **`test_data.csv`**: 2021-2025년 테스트용 데이터
  - 총 7개 종목 × 약 5년 = 수천 행

**데이터 컬럼 구조:**
```
Date, Symbol, Sector, Industry, 
Open, High, Low, Close, Volume,
momentum_1m, momentum_3m, momentum_6m, momentum_12m,
volatility, rsi, macd, macd_signal,
value_factor, beta_factor, momentum_factor, volatility_factor,
weighted_score, signal,
Mkt-RF, SMB, HML, RMW, CMA, RF
```

---

## 📊 출력 예시

```
🚀 Starting Preprocessing Pipeline...
📊 Processing 45 TRAIN stocks (2006-2020)...
📊 Processing 38 TEST stocks (2021-2025)...

📊 Train Total: 157,843 rows
📊 Test Total: 28,492 rows

✂️ Selecting stocks by sector...

[Train Selection] 섹터당 최대 5개
   [Communication Services         ]  5개 선택
   [Consumer Discretionary         ]  5개 선택
   [Consumer Staples               ]  5개 선택
   [Healthcare                     ]  5개 선택
   [Technology                     ]  5개 선택

[Test Selection] 총 7개 (섹터당 약 1개)
   [Communication Services         ]  2개 선택
   [Consumer Discretionary         ]  2개 선택
   [Healthcare                     ]  2개 선택
   [Technology                     ]  1개 선택

✅ Final Selection:
   Train: 25개 종목, 112,450행
   Test:  7개 종목, 8,764행

💾 Saved: data/train_data.csv (112,450 rows)
💾 Saved: data/test_data.csv (8,764 rows)

✅ Pipeline Completed Successfully.
   Train Stocks (25): ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', ...]
   Test Stocks (7): ['NVDA', 'TSLA', 'NFLX', 'PYPL', 'ABNB', 'BKR', 'CCEP']
```

---

## 🔧 커스터마이징

### 종목 선택 개수 조정

`pipeline.py`의 `split_by_sector()` 호출 부분을 수정:

```python
final_train_df, final_test_df, train_symbols, test_symbols = splitter.split_by_sector(
    train_per_sector=10,  # 섹터당 10개로 증가
    test_total=15,        # 테스트 15개로 증가
)
```

### 기간 조정

`pipeline.py`의 `__init__()` 메서드 수정:

```python
self.start_year = 2010  # 2010년부터 시작
self.end_year = 2025
```

### 데이터 품질 기준 조정

`data_splitter.py`의 필터링 조건 수정:

```python
if count >= 2000:  # Train 최소 2000행으로 상향
    valid_symbols.append(sym)
```

---

## 📦 의존성

```
pandas>=2.0.0
numpy>=1.24.0
yfinance>=0.2.28
pandas-datareader>=0.10.0
```

설치:
```bash
pip install -r requirements.txt
```

---

## 🔍 문제 해결

### Q: "No train stocks found" 에러가 발생합니다.

**A:** `nasdaq100_stock_list.csv` 파일의 경로와 컬럼 형식을 확인하세요.
- 필수 컬럼: `Symbol`, `Sector`
- 파일 위치: `data/nasdaq100_stock_list.csv`

### Q: yfinance 다운로드가 실패합니다.

**A:** 인터넷 연결과 Yahoo Finance API 상태를 확인하세요. 일부 종목은 데이터가 없을 수 있으며, 파이프라인은 자동으로 해당 종목을 건너뜁니다.

### Q: Fama-French 데이터 다운로드가 느립니다.

**A:** 정상입니다. Kenneth French Library는 첫 실행 시 전체 기간 데이터를 다운로드하므로 시간이 소요됩니다.

---

## 📚 참고 자료

- [Yahoo Finance API (yfinance)](https://github.com/ranaroussi/yfinance)
- [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
- [SMS 연구 논문](https://www.notion.so/Temporal-Graph-Neural-Network-2a082e91118d809aa283fb97ed6c4ac9)

---

**⚠️ 주의사항**
- 이 전처리 파이프라인은 연구 목적으로 설계되었습니다.
- 생존 편향(Survivorship Bias)이 포함되어 있어 실제 투자 성과와 다를 수 있습니다.
- 과거 데이터 기반 백테스팅은 미래 수익을 보장하지 않습니다.