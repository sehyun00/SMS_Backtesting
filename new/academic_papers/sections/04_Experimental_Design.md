# 4. 실험 설계 (Experimental Design)

## 4.1 데이터셋 구성 (Dataset Description)

본 연구에서는 2006년 1월부터 2025년까지의 글로벌 주식 시장 데이터를 활용하였다. 학습 기간(Train)은 2006–2020년, 테스트 기간(Test)은 2021–2025년으로 설정하였다. 포트폴리오 구성 종목은 S&P 500 지수의 구성 종목 중 GICS(Global Industry Classification Standard) 섹터 다각화 원칙에 따라 체계적으로 선정하였다.

### 4.1.1 종목 선정 방법론 (Stock Selection Methodology)

포트폴리오 종목 선정은 다음의 절차를 통해 수행되었다.

**(1) S&P 500 종목 자동 수집**

Wikipedia의 S&P 500 구성 종목 리스트를 자동 크롤링하여 전체 후보군을 확보하였다. 각 종목의 GICS 섹터 및 세부 산업 정보는 크롤링 데이터와 yfinance API를 통해 교차 검증하였다.

**(2) Survivorship Bias 방지 필터링**

실험의 타당성을 확보하기 위해, 전체 실험 기간(2006–2025) 동안 상장이 유지된 종목만을 대상으로 하였다. 구체적으로, Train 기간(2006–2020)에 최소 1,000 거래일, Test 기간(2021–2025)에 최소 600 거래일의 데이터를 보유한 종목만 후보로 선정하였다.

**(3) 섹터별 균등 추출**

포트폴리오의 위험 분산 효과를 극대화하기 위해, GICS 10개 섹터에서 각 1개 대표 종목을 균등하게 선정하였다. 이는 Moskowitz & Grinblatt(1999)의 산업군 기반 포트폴리오 전략과 Choueifaty & Coignard의 최대 다각화 이론에 기반한다. 선정 기준은 다음과 같다:

- 섹터 완전 다각화: GICS 10개 섹터(Industrials, Health Care, IT, Financials, Materials, Real Estate, Communication Services, Consumer Staples, Consumer Discretionary, Energy) 전체를 포함
- 데이터 품질: 섹터 내 후보 중 데이터 행 수가 가장 풍부한 종목을 우선 선정
- Disjoint Split: Train/Test 간 종목 중복을 원천 차단하여 정보 누출 방지

**(4) 최종 종목 리스트**

위 절차를 통해 선정된 10개 Test 종목은 Table 2와 같다.

| 티커  | 종목명                   | 섹터                   |
| ----- | ------------------------ | ---------------------- |
| MMM   | 3M Company               | Industrials            |
| ABT   | Abbott Laboratories      | Health Care            |
| ACN   | Accenture plc            | Information Technology |
| AFL   | Aflac Inc                | Financials             |
| APD   | Air Products & Chemicals | Materials              |
| ARE   | Alexandria Real Estate   | Real Estate            |
| GOOGL | Alphabet Inc             | Communication Services |
| MO    | Altria Group             | Consumer Staples       |
| AMZN  | Amazon.com Inc           | Consumer Discretionary |
| APA   | APA Corporation          | Energy                 |

**Table 2.** S&P 500에서 GICS 섹터별 균등 추출된 10개 Test 종목

이러한 종목 구성은 GICS 10개 섹터를 빠짐없이 포함하여 최대 수준의 섹터 다각화를 달성하였으며, Evans & Archer(1968)가 제시한 최적 분산투자 종목 수(10~15개)의 범위 내에 있다. 또한 Survivorship Bias 필터링을 통해 전체 실험 기간 동안 안정적으로 거래된 종목만을 포함하여 실험의 신뢰성을 확보하였다.

### 4.1.2 기타 데이터 구성 (Other Data Components)

선정된 10개 종목에 대해 다음의 데이터를 수집하였다:

- **Yahoo Finance API (`yfinance`)**: 일별 주가(OHLCV — Open, High, Low, Close, Volume) 데이터
- **Kenneth French Data Library (`pandas-datareader`)**: Fama & French(2015)의 5요인(Mkt-RF, SMB, HML, RMW, CMA) 일별 팩터 수익률

모델의 입력 특징은 두 가지 경로(Dual-Path)로 분리하여 구성하였다:

| 경로           | 특징                           | 개수     |
| -------------- | ------------------------------ | -------- |
| Local (Price)  | Open, High, Low, Close, Volume | 5개      |
| Global (Macro) | Mkt_RF, SMB, HML, RMW, CMA     | 5개      |
| **합계**       |                                | **10개** |

**Table 3.** Dual-Path 입력 특징 구성

이러한 Dual-Path 구조는 종목별 가격 신호(Local)와 시장 전체 거시경제 팩터(Global)의 신호 희석(Signal Dilution)을 방지하기 위해 설계되었다. 또한 기술적 지표(Momentum 1M/3M/6M/12M)는 TGNN의 예측 대상(Label)으로 활용되었다.

## 4.2 데이터 전처리 (Data Preprocessing)

데이터 품질을 보장하기 위해 다음의 전처리 절차를 수행하였다:

**(1) 기술적 지표 생성 (Technical Indicator Engineering):**

원시 OHLCV 데이터로부터 다음의 파생변수를 생성하여 TGNN의 예측 레이블(Label)로 활용하였다:

| 지표                       | 산출 방법                                   | 목적                                     |
| -------------------------- | ------------------------------------------- | ---------------------------------------- |
| Momentum (1M, 3M, 6M, 12M) | 각각 20, 60, 120, 252 거래일 수익률         | 다중 시간 스케일의 추세 포착 (예측 대상) |
| Volatility                 | 20일 수익률 표준편차 ×$\sqrt{252}$ (연율화) | 리스크 수준 정량화                       |
| RSI (14)                   | 14일 상대강도지수                           | 과매수/과매도 판단                       |
| MACD (12, 26, 9)           | 12일·26일 EMA 차이 및 9일 시그널            | 단기 모멘텀 전환 감지                    |

**Table 4.** TGNN 입력 기술적 지표 및 산출 방법

**(2) Fama-French 팩터 병합:**

Kenneth French Data Library에서 일별 Fama-French 5-Factor 데이터를 다운로드하여, 날짜 기준 Left Join으로 주가 데이터와 병합하였다. 공휴일 등으로 인한 결측값은 Forward Fill로 보완하였다.

**(3) 결측치 처리 (Missing Value Handling):**

슬라이딩 윈도우 생성 시, 특정 종목의 데이터가 윈도우 길이보다 부족한 경우 Zero Padding을 적용하였다. 또한 각 윈도우의 마지막 날짜에 거래 데이터가 존재하지 않는 종목은 Active Mask를 통해 학습 시 제외하여, 비활성 종목이 Ranking Loss에 영향을 미치지 않도록 처리하였다.

**(4) 정규화 (Normalization):**

각 슬라이딩 윈도우 단위로 Robust Z-Score 정규화를 적용하였다. 구체적으로, 윈도우 내 전체 종목·시점의 평균($\mu$)과 표준편차($\sigma$)를 계산하여 $x' = (x - \mu) / (\sigma + \epsilon)$ ($\epsilon = 10^{-8}$) 변환을 수행하였다. 이 방식은 가격(Local)과 매크로(Global) 특징을 상대적으로 스케일링하여 신호 간 균형을 유지한다.

**(5) 그래프 구축 (Graph Construction):**

종목 간 관계를 표현하기 위해 GICS 섹터 기반 인접 행렬(Adjacency Matrix)을 생성하였다. 동일 섹터에 속하는 종목 쌍에는 가중치 $w = 1.0$, 이종 섹터 간에는 약한 연결($w = 0.5$)을 부여하며, 자기 연결(Self-loop, $w = 1.0$)을 포함하여 그래프가 완전 연결(Fully Connected)되도록 구성하였다. 이 그래프는 각 시점(Snapshot)별로 생성되어 TGNN의 공간적 관계 학습에 활용된다.

**(6) 섹터 밸런싱 기반 데이터 분할 (Sector-Balanced Disjoint Split):**

모델의 일반화 성능을 정확히 평가하기 위해, Train/Test 데이터셋을 **Disjoint Split** 전략으로 분리하였다. 이 전략은 두 단계로 구성된다:

1. **Test Set 우선 선정**: 각 섹터에서 데이터 품질이 충분한(≥600 거래일) 대표 종목을 균등하게 선정하여 Test Set(10개 종목)을 먼저 확정한다.
2. **Train Set 배타적 선정**: Test Set에 포함된 종목을 완전히 배제(Disjoint Condition)한 후, 나머지 종목 중 충분한 데이터(≥1,000 거래일)를 보유한 종목을 섹터당 균등하게 배분하여 Train Set을 구성한다.

이 접근법은 (a) 특정 산업군에 대한 편중을 방지하고, (b) Train/Test 간 종목 중복으로 인한 정보 누출(Data Leakage)을 원천적으로 차단하며, (c) 각 섹터의 시장 특성이 테스트 환경에 고르게 반영되도록 보장한다.

## 4.3 실험 환경 및 하이퍼파라미터 (Experimental Environment & Hyperparameters)

| 구성 요소                    | 설정값     |
| ---------------------------- | ---------- |
| TGNN 히든 레이어             | 64         |
| TGNN Attention Head          | 4          |
| TGNN Dropout                 | 0.1        |
| DDPG Actor LR                | 1e–4       |
| DDPG Critic LR               | 1e–3       |
| 할인계수 ($\gamma$)          | 0.99       |
| Soft Update ($\tau$)         | 0.001      |
| 배치크기                     | 64         |
| Replay Buffer 크기           | 10,000     |
| Softmax Temperature ($\tau$) | 3.0        |
| Hybrid Alpha ($\alpha$)      | 0.5 (고정) |
| Epoch 수                     | 800        |
| Optimizer                    | Adam       |

**Table 5.** 모델 하이퍼파라미터 설정

환경: Python 3.12 / PyTorch 2.1 / CUDA 12.9
하드웨어: NVIDIA GeForce RTX 2060 (VRAM 6GB)
운영체제: Windows 11

### 4.3.1 재현성 프로토콜 (Reproducibility Protocol)

본 연구는 학술적 신뢰성과 실험 결과의 완전한 재현성을 보장하기 위해 두 단계의 재현성 프로토콜을 수립하고 준수하였다.

**[1단계] 결정론적 실험 환경 구성**

1. **Seed Fixing**: Python, NumPy, PyTorch, CUDA 환경의 난수 시드(Seed)를 통일하여 모든 실험의 초기화 상태를 고정하였다.
2. **Deterministic Algorithms**: PyTorch 백엔드 설정에서 `cudnn.deterministic = True` 및 `cudnn.benchmark = False`를 적용하여, GPU 연산의 비결정적 요소(Non-deterministic behavior)를 제거하였다.
3. **Hardware Consistency**: 하드웨어별 부동소수점 연산 차이를 최소화하기 위해 모든 실험은 단일 고정 환경(NVIDIA GeForce RTX 2060, Windows 11, CUDA 12.9)에서 수행되었다.

**[2단계] 다중 시드 강건성 검증 (Multi-Seed Robustness Validation)**

단일 시드 고정만으로는 실험 결과가 특정 초기화 상태에 의존하는 **Lucky Seed 문제**를 배제할 수 없다. 이에 본 연구는 5개의 독립적인 시드($S = \{0, 42, 123, 456, 789\}$)를 사용하여 모든 모델(TGNN, DDPG, Hybrid)과 전체 리밸런싱 주기(Monthly, Quarterly, Semiannual, Annual)에 대해 실험을 반복 수행하였다.

최종 보고 지표는 5회 실험의 평균(Mean)과 표준편차(Std)로 제시하여, 결과의 통계적 안정성을 검증하였다:

$$
\bar{m} \pm \sigma = \frac{1}{|S|}\sum_{s \in S} m_s \pm \sqrt{\frac{1}{|S|}\sum_{s \in S}(m_s - \bar{m})^2}
$$

여기서 $m_s$는 시드 $s$에서의 성과 지표(CAGR, Sharpe Ratio 등)이다. 이 프로토콜은 단일 시드 결과의 과적합(Overfitting to a lucky seed) 위험을 차단하며, 제안 모델의 성과가 특정 초기화에 의존하지 않는 **구조적 우위(Structural Advantage)**임을 통계적으로 입증한다.

이러한 2단계 프로토콜을 통해, 본 연구의 실험 결과는 단순한 우연의 산물이 아닌, 검증 가능하고 재현 가능한 논리적 결과임을 보장한다.

## 4.4 평가 지표 (Evaluation Metrics)

제안된 모델의 성능은 세 가지 범주에서 평가하였다.

| 범주                                        | 지표                                     | 목적                              |
| ------------------------------------------- | ---------------------------------------- | --------------------------------- |
| 예측 정확도(Prediction Accuracy)            | MSE, RMSE, R²                            | TGNN 예측 및 임베딩 품질 평가     |
| 리스크 조정 성과(Risk-adjusted Performance) | Sharpe Ratio, Sortino Ratio, CVaR, Omega | DDPG 정책의 리스크-보상 균형 평가 |

**Table 6.** 성능 평가 지표 범주 및 목적

**Sharpe Ratio**와 **Sortino Ratio**는 다음과 같이 정의된다:

$$
\text{Sharpe Ratio} = \frac{R_p - R_f}{\sigma_p}
$$

$$
\text{Sortino Ratio} = \frac{R_p - R_f}{\sigma_d}
$$

- **CVaR(Conditional Value-at-Risk)**은 손실 분포 하위 5%의 평균 손실로 측정하였다.

## 4.5 비교 모델 (Baseline Models)

제안된 하이브리드 AI DSS의 우수성을 검증하기 위해, 다음의 비교 대상을 설정하였다:

| 모델                        | 설명                           | 특성                                    |
| --------------------------- | ------------------------------ | --------------------------------------- |
| Benchmark (Buy & Hold)      | 동일 비중 매입 후 보유 전략    | 패시브 전략 기준선                      |
| TGNN                        | 그래프 기반 관계 예측 모델     | 시장 구조 반영 가능, 정책 최적화 미포함 |
| DDPG                        | 강화학습 기반 정책 최적화 모델 | 정책 학습 가능, 종목 간 관계 인식 불가  |
| TGNN+DDPG Hybrid (Proposed) | 하이브리드 DSS 모델            | 관계·정책·앙상블 최적화 통합 구조       |

**Table 7.** 비교 대상 모델(Baseline) 구성

각 모델은 Monthly, Quarterly, Semiannual, Annual의 4가지 리밸런싱 주기에서 독립적으로 평가하여, 주기별 성과 특성을 분석하였다.

## 4.6 검증 절차 및 강건성 평가 (Validation and Reliability Check)

모델의 일반화 성능과 신뢰성을 검증하기 위해, 시간적 순서를 엄격히 준수하는 데이터 분할을 수행하였다.

- **학습(Train)**: 2006–2020년 데이터 (약 3,700 거래일)
- **테스트(Test)**: 2021–2025년 데이터 (약 1,000 거래일)
- **종목 분리**: Train/Test 간 Disjoint Split 적용 (섹터당 균등 배분, 종목 중복 없음)

Train/Test 간 종목이 완전히 분리(Disjoint)되어 있어, 특정 종목에 대한 과적합(Overfitting)이 아닌 시장 구조 자체를 학습했는지를 평가할 수 있다.

또한, 테스트 기간에는 코로나19 팬데믹 이후 회복기(2021), 러시아-우크라이나 전쟁(2022), 금리 인상기(2023) 등 다양한 시장 국면이 포함되어 있어, 모델의 강건성을 자연스럽게 검증할 수 있는 환경을 제공한다.

## 4.7 DSS 통합 및 피드백 구조 (Integration into DSS)

AI 엔진은 Flask 기반 API 서버에서 구동되며, Spring Boot 백엔드와 React 프런트엔드를 통해 DSS 인터페이스와 연동된다 (Park & Han, 2024).

모델 출력(추천 비중, 리스크 경고, 거래 제안 등)은 RESTful API를 통해 대시보드에 실시간 전송되고, 사용자 피드백은 데이터베이스에 저장되어 지속 학습(Continual Learning)에 활용된다. 또한 Explainable AI(XAI) 모듈을 통합하여 TGNN Multi-Head Temporal Self-Attention 가중치를 추출하고, Temporal Attention Map(T×T)과 Stock Attention Map(N×T)을 시각화함으로써 사용자가 AI의 의사결정 근거(reasoning path)를 직관적으로 이해할 수 있도록 하였다.

## 4.8 실험 설계 요약 (Summary of Experimental Design)

본 실험 설계는 다음의 세 가지 목표를 달성하도록 구성되었다:

① 관계 기반 예측 구조의 정확성 검증: TGNN의 시공간 관계 학습 능력을 평가

② 정책 최적화 및 신뢰성 검증: Deterministic 환경하에서 DDPG의 리스크 조정 수익률 및 재현성 입증

③ 실시간 DSS 통합 타당성 검증: Flask–Spring–React 환경에서 실시간 응답성과 피드백 효율성 평가
