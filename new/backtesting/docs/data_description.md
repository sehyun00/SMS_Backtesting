# 📊 데이터셋 상세 기술서 (Data Description)

본 문서는 연구에 사용된 금융 시계열 데이터(Financial Time-series Data)의 구성, 전처리 과정, 및 학술적 정합성을 기술합니다.

## 1. 데이터 개요 (Dataset Overview)
*   **유니버스 (Universe)**: S&P 500 및 NASDAQ 상장 주요 기술주 및 대형주 (총 55개 종목).
*   **기간 (Period)**: 2010.01.01 ~ 2025.12.31 (Training & Testing).
*   **주기 (Frequency)**: 일별 데이터 (Daily).
*   **원천 (Source)**: Yahoo Finance (OHLCV), Kenneth R. French Data Library (Fama-French Factors).

## 2. 변수 구성 (Feature Engineering)
본 연구는 단순히 주가 변동성뿐만 아니라, **Asset Pricing Model(자산 가격 결정 모형)**에 입각한 재무적 요인들을 입력 변수(Features)로 사용하였습니다.

### 2.1. 시장 및 재무 팩터 (Fama-French 5 Factors + Momentum)
[Fama & French (2015)](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)의 5요인 모형을 적용하여 모델이 시장 초과 수익(Alpha)을 학습할 수 있도록 설계하였습니다.

| 변수명 (Variable) | 설명 (Description) | 학술적 의의 (Rationale) |
| :--- | :--- | :--- |
| **Mkt-RF** | 시장 초과 수익률 (Market Excess Return) | 시장 전반의 리스크 프리미엄 통제. CAPM의 Beta에 해당. |
| **SMB** | 소형주 효과 (Small Minus Big) | 시가총액(Size)에 따른 리스크 요인 반영. |
| **HML** | 가치주 효과 (High Minus Low) | 가치주(Value) vs 성장주(Growth) 성향 반영. |
| **RMW** | 수익성 효과 (Robust Minus Weak) | 기업의 영업이익력(Profitability) 요인 반영. |
| **CMA** | 투자 패턴 (Conservative Minus Aggressive) | 기업의 자산 재투자 성향 반영. |
| **RF** | 무위험 이자율 (Risk-Free Rate) | 거시경제 금리 환경(Monetary Policy) 대리 변수. |

### 2.2. 기술적 지표 (Technical Indicators)
주가의 단기 및 중장기 추세를 포착하기 위해 다중 시계열(Multi-horizon) 모멘텀을 사용하였습니다.

*   **Momentum (1M, 3M, 6M, 12M)**: 1개월~12개월 누적 수익률. (Jegadeesh & Titman, 1993의 모멘텀 효과 반영)
*   **RSI (Relative Strength Index)**: 과매수/과매도 구간 식별 (Range: 0~100).
*   **MACD & Signal**: 단기/장기 이동평균 수렴확산 지표. 추세 반전 포착.
*   **Volatility (Hist. Vol)**: 과거 변동성. 리스크(Risk) 측정 지표.

## 3. 데이터 전처리 (Preprocessing Methodology)
데이터의 **정상성(Stationarity)** 확보와 모델 학습 안정성을 위해 엄격한 전처리를 수행하였습니다.

1.  **정규화 (Normalization)**:
    *   가격 데이터(Price)가 아닌 **수익률(Return)** 데이터를 사용하여 Non-stationary 문제 해결.
    *   주요 팩터(Momentum, Volatility 등)는 **Z-Score Normalization** ($\frac{x - \mu}{\sigma}$) 또는 MinMax Scaling을 적용하여 이상치(Outlier) 영향을 최소화.
    
2.  **결측치 처리 (Missing Values)**:
    *   Rolling Window 연산(예: 12개월 모멘텀)으로 인해 발생하는 초기 결측치는 학습에서 배제(Drop)하거나, Backward Fill 방식으로 보정.

3.  **라벨링 (Labeling)**:
    *   **Weighted Score**: 미래 수익률과 샤프 지수 기여도를 결합한 복합 점수(Composite Score). 지도학습(TGNN)의 Target으로 사용.
    *   **Smart Signal**: 팩터 기반 퀀트 전략(Quant Rule)에 의해 생성된 매수/매도 시그널 (Auxiliary Loss로 활용 가능).

## 4. 데이터셋 품질 평가 (Quality Assessment)
*   **통계적 유의성**: Fama-French 팩터의 일별 변동폭은 1% 내외($\pm 0.01$)로 정상 범위에 분포함.
*   **노이즈 레벨**: RSI 및 MACD 지표는 발산(Divergence) 없이 유의미한 시그널을 제공하고 있음이 확인됨.
*   **활용 가능성**: 본 데이터셋은 **Factor Investing**과 **Deep Learning**을 결합한 하이브리드 모델 연구에 최적화되어 있음.
