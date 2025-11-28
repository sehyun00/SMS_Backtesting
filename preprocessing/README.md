# Data Preprocessing & Factor Engineering

## 1. 개요 (Overview)
본 모듈(`DataPreprocessing.py`)은 Yahoo Finance API를 통해 수집된 원시(Raw) 시장 데이터(OHLCV)를 강화학습(DDPG) 및 그래프 신경망(TGNN) 모델이 학습 가능한 형태의 **구조화된 5-Factor 데이터셋**으로 변환합니다.

본 연구의 핵심은 **정적인 재무제표 데이터(Static Financial Data)의 한계를 극복**하고, 고빈도 거래 환경에서 리밸런싱 에이전트가 시장의 미세한 변화를 포착할 수 있도록 **동적 기술적 팩터(Dynamic Technical Factors)**를 설계하는 데 있습니다.

---

## 2. 데이터 구축 전략 (Design Philosophy)

전통적인 Fama-French 5-Factor 모델은 분기별 재무제표(Book Value, Operating Profit 등)에 의존합니다. 그러나 이는 다음과 같은 한계를 가집니다:
1. **데이터 지연(Lag)**: 재무제표는 발표 시점과 실제 시장 반영 시점 간의 괴리가 큼.
2. **정적 특성(Static)**: 일별(Daily) 리밸런싱을 수행하는 강화학습 에이전트에게는 상태(State) 변화가 너무 적음.

따라서 본 연구에서는 **"기술적 대용 지표(Technical Proxies)"** 접근법을 채택하여, 가격(Price)과 거래량(Volume) 정보만으로 전통적 팩터의 금융공학적 의미를 모사(Mimic)하였습니다.

---

## 3. 팩터 상세 정의 및 산출 수식 (Factor Specifications)

모든 팩터는 **일별 횡단면 분석(Daily Cross-Sectional Analysis)**을 기반으로 산출되며, 0과 1 사이의 값으로 정규화(Normalize)됩니다.

### 3.1. Value Factor (가치 팩터) → Technical Reversion
전통적 **PBR(주가순자산비율)**을 대체합니다. "내재 가치 대비 싼 주식"이라는 개념을 **"최근 고점 대비 과매도된 주식(Mean Reversion)"**으로 재해석하였습니다.

*   **금융공학적 의미**: 단기/중기 낙폭 과대 종목의 평균 회귀(Reversion) 기대 수익 포착.
*   **산출 수식**:
    $$ \text{Score}_{\text{Value}} = 1 - \text{Rank}\left( \frac{P_t}{\max(P_{t-252:t})} \right) $$
    *   $P_t$: 현재 주가
    *   $\max(P_{t-252:t})$: 최근 52주(1년) 최고가
    *   설명: 고점 대비 하락률이 클수록(비율이 낮을수록) 높은 점수(Rank 1)를 부여.

### 3.2. Size Factor (규모 팩터) → Liquidity Premium
전통적 **시가총액(Market Cap)**을 대체합니다. 시가총액과 높은 상관관계를 가지는 **거래대금(Dollar Volume)**을 사용합니다.

*   **금융공학적 의미**: 유동성이 낮은 소외주(Illiquid Stocks)가 장기적으로 더 높은 수익률을 보상한다는 소형주 효과(Small-cap Effect) 반영.
*   **산출 수식**:
    $$ \text{Score}_{\text{Size}} = 1 - \text{Rank}\left( \ln(P_t \times V_t) \right) $$
    *   $V_t$: 일일 거래량
    *   설명: 로그 거래대금이 작을수록 높은 점수 부여.

### 3.3. Momentum Factor (모멘텀 팩터)
전통적 정의를 그대로 따릅니다.

*   **금융공학적 의미**: 최근 승자(Winner) 주식이 계속 상승하는 추세 추종(Trend Following) 효과.
*   **산출 수식**:
    $$ \text{Score}_{\text{Mom}} = \text{Rank}\left( \frac{P_t}{P_{t-252}} - 1 \right) $$
    *   설명: 지난 12개월(252일) 수익률이 높을수록 높은 점수 부여.

### 3.4. Volatility Factor (변동성 팩터) → Low Volatility
*   **금융공학적 의미**: 저변동성 주식이 고변동성 주식보다 위험 조정 수익률(Risk-adjusted Return)이 우수하다는 '저변동성 이상현상(Low Volatility Anomaly)' 활용.
*   **산출 수식**:
    $$ \text{Score}_{\text{Vol}} = 1 - \text{Rank}\left( \sigma(R_{t-30:t}) \right) $$
    *   $\sigma$: 최근 30일 일일 수익률의 표준편차
    *   설명: 변동성이 낮을수록 높은 점수 부여.

### 3.5. Beta Factor (베타 팩터) → Defensive Quality
*   **금융공학적 의미**: 시장 민감도(Beta)가 낮은 방어주(Defensive Stocks)를 선호하여 하락장 리스크 관리.
*   **산출 수식**:
    $$ \text{Score}_{\text{Beta}} = 1 - \text{Rank}\left( \frac{\text{Cov}(R_i, R_m)}{\text{Var}(R_m)} \right) $$
    *   Window: 최근 1년(252일) Rolling Window 사용
    *   설명: 시장 베타가 낮을수록 높은 점수 부여.

---

## 4. 데이터 처리 파이프라인 (Pipeline Steps)

1.  **Data Fetching**: `yfinance`를 통해 최근 10년치 OHLCV 데이터 및 S&P500(^GSPC) 벤치마크 지수 수집.
2.  **Feature Engineering**:
    *   이동 평균(Rolling Mean/Max), 변동성, 베타 등 파생 변수 계산.
    *   결측치(NaN) 처리: Forward Fill 적용.
3.  **Cross-Sectional Ranking (핵심)**:
    *   매일(Daily) 존재하는 종목들 간의 **상대 순위(Percentile Rank)** 계산.
    *   **Normalization**: 모든 팩터 값은 $0.0 \sim 1.0$ 범위로 정규화됨.
    *   이 과정은 TGNN이 특정 종목의 절대적 가격이 아닌 **종목 간의 상대적 우위 관계(Graph Relation)**를 학습하는 데 필수적임.
4.  **Export**: 최종 결과물은 `processed_daily_5factor_model.csv`로 저장됨.

---

## 5. 출력 데이터 명세 (Output Schema)

| Column Name | Description |
| :--- | :--- |
| `Date` | 거래일 (YYYY-MM-DD) |
| `ticker` | 종목 코드 (e.g., AAPL) |
| `Close` / `Volume` | 수정 주가 및 거래량 (Raw Data) |
| `Value_Factor` | 0~1 정규화된 가치 점수 (높을수록 저평가) |
| `Size_Factor` | 0~1 정규화된 규모 점수 (높을수록 소형/저유동성) |
| `Momentum_Factor` | 0~1 정규화된 모멘텀 점수 (높을수록 강한 추세) |
| `Volatility_Factor` | 0~1 정규화된 변동성 점수 (높을수록 안정적) |
| `Beta_Factor` | 0~1 정규화된 베타 점수 (높을수록 시장 비상관) |
| `smart_signal` | 팩터 종합 점수에 기반한 매매 시그널 (BUY/SELL/NEUTRAL) |
