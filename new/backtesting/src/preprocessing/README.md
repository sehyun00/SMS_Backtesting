# Preprocessing Module (데이터 전처리)

`src/preprocessing`은 원본 금융 데이터 수집, 기술적 지표 계산, 팩터 병합, 그리고 Train/Test 분할을 담당하는 모듈입니다. 연구의 **재현성**을 보장하기 위해 모든 데이터 변환 과정이 결정론적(Deterministic)으로 수행됩니다.

## 📦 모듈 구조

| 모듈 | 역할 |
|---|---|
| `pipeline.py` | 전체 전처리 과정(수집 $\rightarrow$ 가공 $\rightarrow$ 저장)을 관장하는 컨트롤러 |
| `data_collector.py` | `yfinance` 및 NASDAQ 크롤링을 통해 원본 OHLCV 데이터 수집 |
| `data_processor.py` | 기술적 지표(RSI, MACD 등) 계산 및 이상치 처리 |
| `fama_french_loader.py` | `pandas-datareader`를 사용하여 Fama-French 5 Factor 데이터 수집 및 병합 |
| `data_splitter.py` | 섹터별 균형을 고려한 Train/Test 데이터셋 분할 |
| `indicators.py` | 보조지표 계산 로직 모음 |
| `factors.py` | 팩터 계산 로직 모음 |

## 🏗️ 아키텍처 및 데이터 흐름

```mermaid
flowchart LR
    A[Raw Data Source] -->|yfinance| B(DataCollector)
    B --> C(DataProcessor)
    C -->|Add Indicators| D(Merged DataFrame)
    E[Fama-French Source] -->|pandas-datareader| F(FamaFrenchLoader)
    F -->|Merge Factors| D
    D --> G(DataSplitter)
    G -->|Split by Sector| H[Train Data (2006-2020)]
    G -->|Split by Sector| I[Test Data (2021-2025)]
```

## 🔧 주요 기능 상세

### 1. 데이터 수집 (`DataCollector`)
- **자동 크롤링**: `Wikipedia`에서 S\&P 500 또는 NASDAQ 100 종목 리스트를 자동으로 가져옵니다.
- **기간 필터링**: 생존 편향(Survivorship Bias)을 없애기 위해 Start-End 기간 동안 상장 유지된 종목만 선별합니다.

### 2. 파생변수 생성 (`DataProcessor`)
- **기술적 지표**: Momentum(1M, 3M, 6M, 12M), RSI, MACD, Volatility 등 파생 변수를 생성합니다.
- **가중치 점수**: `config.yaml`에 정의된 팩터 가중치(Value, Momentum, Volatility 등)를 기반으로 `weighted_score`를 계산합니다.

### 3. 데이터 분할 (`DataSplitter`)
- **섹터 밸런싱**: Train 및 Test 세트가 특정 산업군(Technology 등)에 편중되지 않도록 섹터별로 균등하게 종목을 분배합니다.
- **기간 분리**: 
    - **Train**: 2006년 ~ 2020년
    - **Test**: 2021년 ~ 2025년

## 🚀 사용법 (Usage)

### CLI 실행
```bash
# NASDAQ 100 / S&P 500 자동 크롤링 모드
python -m new.backtesting.src.preprocessing.pipeline

# CSV 파일 지정 모드
python -m new.backtesting.src.preprocessing.pipeline --csv data/stock_list.csv
```

### Config 설정 (`config/config.yaml`)
데이터 처리에 영향을 주는 주요 설정들은 `data` 섹션에서 관리됩니다.
```yaml
data:
  features: ["Open", "High", "Low", "Close", "Volume"]
  factors:
    weights:
      value: 0.3
      momentum: 0.3
      volatility: 0.2
      beta: 0.2
```
