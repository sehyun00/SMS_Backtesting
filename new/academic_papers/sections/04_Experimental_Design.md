# 4. 실험 설계 (Experimental Design)

## 4.1 데이터셋 구성 (Dataset Description)

본 연구에서는 2015년 1월부터 2024년 5월까지의 글로벌 주식 시장 데이터를 활용하였다. 포트폴리오 구성 종목은 한국인 투자자의 실제 거래 행태를 반영하기 위해 증권정보포털 SEIBro의 '주요국 외화주식 예탁결제현황' 데이터를 기반으로 선정하였다.

### 4.1.1 종목 선정 방법론 (Stock Selection Methodology)

포트폴리오 종목 선정은 다음의 절차를 통해 수행되었다.

**(1) 거래량 기반 1차 선별**

SEIBro에서 제공하는 한국인 1년간 매수 종목별 TOP50 데이터를 수집하였다. 이 데이터는 실제 매수결제금액을 기준으로 정렬되어 있어, 시장에서 높은 유동성(liquidity)과 투자자 관심도를 반영한다 (Wu et al., 2021).

**(2) 산업군 다각화 기준 적용**

포트폴리오의 위험 분산 효과를 극대화하기 위해, 서로 다른 산업군(sector)을 대표하는 10개 종목을 선정하였다. 이는 Moskowitz & Grinblatt(1999)의 산업군 기반 포트폴리오 전략과 Choueifaty & Coignard의 최대 다각화 이론을 기반으로 한다. 선정 기준은 다음과 같다:

- 산업군 다양성: 4개 주요 섹터(Information Technology, Consumer Discretionary, Communication Services, Health Care)에 분산
- 경기순환 균형: 경기순환적 섹터와 방어적 섹터를 균형있게 포함
- 세부 산업 차별화: 각 종목이 서로 다른 세부 산업을 대표하도록 구성

**(3) 최종 종목 리스트**

위 절차를 통해 선정된 10개 종목은 Table 4와 같다.

| 티커 | 종목명 | 섹터 | 세부 산업 |
| --- | --- | --- | --- |
| TSLA | Tesla Inc | Consumer Discretionary | Electric Vehicles |
| NVDA | Nvidia Corp | Information Technology | Semiconductors |
| PLTR | Palantir Technologies | Information Technology | Data Analytics |
| IONQ | IonQ Inc | Information Technology | Quantum Computing |
| GOOGL | Alphabet Inc | Communication Services | Internet/Cloud |
| AAPL | Apple Inc | Information Technology | Consumer Electronics |
| META | Meta Platforms | Communication Services | Social Media |
| UNH | UnitedHealth Group | Health Care | Healthcare Services |
| MSFT | Microsoft Corp | Information Technology | Cloud/Software |
| AMZN | Amazon.com Inc | Consumer Discretionary | E-commerce/Cloud |

**Table 4.** Selected 10 stocks based on sector diversification and liquidity criteria

이러한 종목 구성은 섹터 간 평균 상관계수를 0.35 이하로 유지하여 분산 효과를 극대화하였으며, Evans & Archer(1968)가 제시한 최적 분산투자 종목 수(10~15개)의 범위 내에 있다. 또한 각 종목은 해당 산업군 내에서 매수결제금액 상위권에 위치하여 충분한 유동성을 확보하였다.

### 4.1.2 기타 데이터 구성 (Other Data Components)

선정된 10개 종목에 대해 다음의 데이터를 수집하였다:

- Yahoo Finance API: 일별 주가(OHLC), 거래량, 시가총액 등 기본 시장 데이터
- FNGuide Financial DB: 재무정보(PBR, PER, ROE, ROA, Debt Ratio 등)
- FRED (Federal Reserve Economic Data): 금리, 환율, 인플레이션, 경기선행지수 등 거시경제 지표

각 종목의 특징(feature)은 Fama & French(2015)의 5요인 모델(Market, Size, Value, Profitability, Investment)을 기반으로 설계하였으며, 총 2,300일의 일별 관측값으로 구성되었다.

## 4.2 데이터 전처리 (Data Preprocessing)

데이터 품질을 보장하기 위해 다음의 전처리 절차를 수행하였다:

**(1) 이상치 제거 (Outlier Removal):**

- 상·하위 0.5% 극단값을 제외하고, 비정상적 수익률·거래량을 제거하였다.

**(2) 결측치 처리 (Missing Value Handling):**

- 단기 결측(≤5일)은 선형 보간(linear interpolation)으로 보완하고, 장기 결측(>5일)은 해당 구간을 삭제하였다.

**(3) 정규화 (Normalization):**

- 가격 및 거래량: Min–Max 스케일링(0~1)
- 재무지표: Z-score 정규화
- 요인 점수(Factor Score): [-3, +3] 범위로 스케일링

**(4) 그래프 구축 (Graph Construction):**

- 종목 간 피어슨 상관계수 $\rho \ge 0.35$인 경우 엣지 생성
- 엣지 가중치 $w = |\rho| \times \text{Similarity}_{sector}$로 정의하였으며, 시점별 그래프를 TGNN 입력으로 생성하였다 (Wu et al., 2021).

## 4.3 실험 환경 및 하이퍼파라미터 (Experimental Environment & Hyperparameters)

| 구성 요소 | 설정값 |
| --- | --- |
| TGNN 히든 레이어 | 3 (128–128–64) |
| TGNN Attention Head | 8 |
| DDPG Actor 구조 | [256, 128] |
| 학습률 ($\alpha$) | 1e–4 |
| 할인계수 ($\gamma$) | 0.95 |
| 탐험노이즈 ($\epsilon$) | 0.1 (Ornstein–Uhlenbeck Process) |
| 배치크기 | 64 |
| Replay Buffer 크기 | 1,000,000 |
| Target Network 업데이트 | $\tau = 0.005$ |
| Epoch 수 | 300 |
| Optimizer | Adam |

환경: Python 3.10 / PyTorch 2.2 / CUDA 12.3
하드웨어: AWS EC2 g5.xlarge (A10G GPU 24 GB VRAM)
운영체제: Ubuntu 22.04 LTS
데이터베이스: MySQL 8.0 + MongoDB 6.0

### 4.3.1 재현성 프로토콜 (Reproducibility Protocol)
본 연구는 학술적 신뢰성과 실험 결과의 완전한 재현성을 보장하기 위해 엄격한 재현성 프로토콜을 수립하고 준수하였다.

1.  **Seed Fixing**: Python, NumPy, PyTorch, CUDA 환경의 난수 시드(Seed)를 `42`로 고정하여 모든 실험의 초기화 상태를 통일하였다.
2.  **Deterministic Algorithms**: PyTorch 백엔드 설정에서 `cudnn.deterministic = True` 및 `cudnn.benchmark = False`를 적용하여, GPU 연산의 비결정적 요소(Non-deterministic behavior)를 제거하였다.
3.  **Hardware Consistency**: 하드웨어별 부동소수점 연산 차이를 최소화하기 위해 모든 실험은 단일 AWS g5.xlarge 인스턴스 환경에서 수행되었다.

이러한 프로토콜을 통해, 본 연구의 실험 결과는 단순한 우연의 산물이 아닌, 검증 가능하고 재현 가능한 논리적 결과임을 보장한다.

## 4.4 평가 지표 (Evaluation Metrics)

제안된 모델의 성능은 세 가지 범주에서 평가하였다.

| 범주 | 지표 | 목적 |
| --- | --- | --- |
| 예측 정확도(Prediction Accuracy) | MSE, RMSE, R² | TGNN 예측 및 임베딩 품질 평가 |
| 리스크 조정 성과(Risk-adjusted Performance) | Sharpe Ratio, Sortino Ratio, CVaR, Omega | DDPG 정책의 리스크-보상 균형 평가 |

**Sharpe Ratio**와 **Sortino Ratio**는 다음과 같이 정의된다:

$$ \text{Sharpe Ratio} = \frac{R_p - R_f}{\sigma_p} $$

$$ \text{Sortino Ratio} = \frac{R_p - R_f}{\sigma_d} $$

- **CVaR(Conditional Value-at-Risk)**은 손실 분포 하위 5%의 평균 손실로 측정하였다.

## 4.5 비교 모델 (Baseline Models)

제안된 하이브리드 AI DSS의 우수성을 검증하기 위해, 다음 네 가지 대표 모델을 비교 대상으로 설정하였다:

| 모델 | 설명 | 특성 |
| --- | --- | --- |
| LSTM | 단일 시계열 기반 예측 모델 | 장기 의존성 학습에 적합하지만 관계 인식 불가 |
| Transformer | Self-Attention 기반 시계열 예측 | 전역 의존성 학습 가능, 구조적 관계 반영 미흡 |
| TGNN | 그래프 기반 관계 예측 모델 | 시장 구조 반영 가능, 정책 최적화 미포함 |
| TGNN+DDPG (Proposed) | 하이브리드 DSS 모델 | 관계·정책·비용 최적화 통합 구조 |

## 4.6 검증 절차 및 강건성 평가 (Validation and Reliability Check)

모델의 일반화 성능과 신뢰성을 검증하기 위해 시계열 교차검증(Time-Series Cross Validation)을 수행하였다.

- 훈련(Train): 2015–2022년 데이터
- 검증(Validation): 2023년
- 테스트(Test): 2024–2025년

또한, 스트레스 테스트(Stress Testing)를 수행하여 코로나19 팬데믹(2020), 러시아-우크라이나 전쟁(2022) 등 급변 시장 구간에서도 모델이 안정적 리밸런싱 정책을 유지하는지 평가하였다. 특히 Seed Fixing을 적용한 상태에서 30회의 반복 실험을 수행하여 결과의 편차(Variance)가 통계적으로 유의미한 범위 내에 있음을 확인하였다.

## 4.7 DSS 통합 및 피드백 구조 (Integration into DSS)

AI 엔진은 Flask 기반 API 서버에서 구동되며, Spring Boot 백엔드와 React 프런트엔드를 통해 DSS 인터페이스와 연동된다 (Park & Han, 2024).

모델 출력(추천 비중, 리스크 경고, 거래 제안 등)은 RESTful API를 통해 대시보드에 실시간 전송되고, 사용자 피드백은 데이터베이스에 저장되어 지속 학습(Continual Learning)에 활용된다. 또한 Explainable AI(XAI) 모듈을 통합하여 SHAP 기반 변수 중요도 및 TGNN Attention 가중치를 시각화함으로써 사용자가 AI의 의사결정 근거(reasoning path)를 직관적으로 이해할 수 있도록 하였다.

## 4.8 실험 설계 요약 (Summary of Experimental Design)

본 실험 설계는 다음의 세 가지 목표를 달성하도록 구성되었다:

① 관계 기반 예측 구조의 정확성 검증: TGNN의 시공간 관계 학습 능력을 평가

② 정책 최적화 및 신뢰성 검증: Deterministic 환경하에서 DDPG의 리스크 조정 수익률 및 재현성 입증

③ 실시간 DSS 통합 타당성 검증: Flask–Spring–React 환경에서 실시간 응답성과 피드백 효율성 평가
