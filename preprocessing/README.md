# 🛠️ SMS Backtesting - Data Preprocessing Module

이 모듈은 **SMS(Stock Movement System)** 논문 구현 및 **TGNN(Temporal Graph Neural Network)** 모델 학습을 위한 데이터 전처리 파이프라인입니다.

나스닥 100(NASDAQ 100) 종목의 과거 데이터를 수집하고, 기술적 지표 및 팩터 점수를 계산하며, 모델 검증을 위해 **생존 편향(Survivorship Bias)**을 고려한 정교한 Train/Test 데이터셋 분할을 수행합니다.

---

## 📂 디렉토리 구조 (Directory Structure)

preprocessing/
├── init.py # 패키지 초기화
├── data_collector.py # 데이터 수집 (yfinance) 및 생존 종목 필터링
├── technical_indicators.py # 기술적 지표 (RSI, MACD, Volatility 등) 계산
├── factor_calculator.py # 팩터 점수 산출 및 매수/매도 시그널 생성
├── fama_french_loader.py # Fama-French 5 Factor 데이터 다운로드 및 병합
├── data_splitter.py # 섹터 기반 학습/테스트 데이터 분할 로직 (핵심)
└── pipeline.py # 전체 프로세스 통합 실행 (Main Entry Point)

## 🚀 주요 로직 및 전략 (Key Logic)

### 1. 생존 종목 필터링 (Survivor Filtering)

**TGNN(시계열 그래프 신경망)**의 구조적 안정성을 위해 **노드(종목)의 개수를 고정**해야 합니다.

- **전략:** 2025년 현재 나스닥 100 리스트 중, **2006년부터 2025년까지 전 기간 상장 유지된 종목**만 필터링합니다.
- **이유:**
  - 그래프 인접 행렬(Adjacency Matrix)의 크기($N \times N$)를 고정하여 학습 안정성 확보
  - 중간에 노드가 사라지거나 생길 경우 발생하는 Temporal Attention 연산 오류 방지
  - _Note: 생존 편향(Survivorship Bias)이 존재하지만, 모델 구조 구현을 위한 현실적인 타협점입니다._

### 2. 섹터 기반 데이터 분할 (Sector-based Split)

단순 시계열 분할이 아닌, **섹터별 대표성**을 유지하며 데이터를 나눕니다.

- **분할 규칙:** 각 섹터(Sector) 내에서 생존한 종목들을 2개 그룹으로 나눕니다.
  - **Train Set (학습):** 각 섹터의 첫 번째 종목 (기간: `2006` ~ `2020`)
  - **Test Set (테스트):** 각 섹터의 두 번째 종목 (기간: `2021` ~ `2025`)
- **목적:** 모델이 특정 종목(예: 애플)의 패턴만 외우는 것을 방지하고, 해당 섹터(예: 기술주)의 일반적인 움직임을 학습했는지 검증합니다.

### 3. 지표 및 팩터 (Indicators & Factors)

- **Technical Indicators:** Momentum (1/3/6/12M), Volatility, RSI, MACD
- **Factor Scoring:** Value, Beta, Momentum, Volatility 기반 가중치 점수 및 시그널(Strong Buy ~ Strong Sell)
- **Market Factors:** Kenneth French Data Library의 **Fama-French 5 Factors** (Mkt-RF, SMB, HML, RMW, CMA) 병합

---

## 💻 사용 방법 (Usage)

### 1. 사전 준비

프로젝트 루트 디렉토리에 나스닥 100 종목 리스트 파일(`nasdaq100_stock_list.csv`)이 있어야 합니다.

> **필수 컬럼:** `Symbol` (티커), `Sector` (섹터)

### 2. 파이프라인 실행

터미널에서 프로젝트 루트(`SMS_Back_Testing`)로 이동 후 아래 명령어를 실행하세요.

모듈로 실행
python -m preprocessing.pipeline

### 3. 결과 확인

실행이 완료되면 지정된 출력 디렉토리(기본: `data/`)에 두 개의 파일이 생성됩니다.

- **`train_data.csv`**: 2006~2020년 데이터 (학습용 종목군)
- **`test_data.csv`**: 2021~2025년 데이터 (테스트용 종목군 - 학습에 사용되지 않은 종목)

---

## 📦 요구 사항 (Requirements)

이 모듈을 실행하기 위해 다음 라이브러리가 필요합니다.

pandas
numpy
yfinance
pandas_datareader
scikit-learn (Optional, for future scaler)

설치 명령어:

pip install -r requirements.txt
