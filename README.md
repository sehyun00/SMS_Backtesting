# SMS_Backtesting: TGNN & DDPG 하이브리드 포트폴리오 최적화

## 📖 프로젝트 개요
본 저장소는 연구 논문 **"TGNN-DDPG 하이브리드 모델을 활용한 포트폴리오 리밸런싱 전략 강화"**를 위한 구현 및 백테스팅 프레임워크입니다.

제안하는 하이브리드 모델의 성능을 검증하기 위해, 다음 세 가지 모델의 투자 성과를 비교 실험합니다:
1. **TGNN-Only**: 시계열 그래프 신경망(Temporal Graph Neural Network)을 이용한 종목 간 관계 분석 및 예측.
2. **DDPG-Only**: 심층 결정론적 정책 그래디언트(DDPG) 강화학습을 이용한 동적 비중 최적화.
3. **Hybrid (TGNN+DDPG)**: TGNN이 추출한 잠재 특징(Latent Features)을 기반으로 DDPG가 포트폴리오를 최적화하는 결합 모델.

---

## 📂 디렉토리 구조
실험의 재현성(Reproducibility)과 모델 간 명확한 비교를 위해 다음과 같이 구조화되었습니다.

SMS_Backtesting/
│
├── 📁 data/ # [공통] 데이터 저장소
│ ├── stock_list.csv # 대상 종목 리스트 (10개 우량주)
│ └── processed_daily_5factor_model.csv # 전처리 완료된 5팩터 데이터
│
├── 📁 preprocessing/ # [공통] 데이터 엔지니어링 모듈
│ ├── DataPreprocessing.py # 팩터 계산 및 데이터 정제 코드
│ └── README.md # 기술적 팩터 정의 및 산출 방식 상세 명세
│
├── 📁 models/ # 모델 구현체
│ ├── 📁 TGNN/ # 비교군 1: 그래프 신경망 단독 모델
│ ├── 📁 DDPG/ # 비교군 2: 강화학습 단독 모델
│ └── 📁 Hybrid_TGNN_DDPG/ # 제안 모델 (Main Method)
│
├── 📁 results/ # 실험 결과 (로그 및 그래프)
│ ├── 📁 01_TGNN_Only/
│ ├── 📁 02_DDPG_Only/
│ └── 📁 03_Hybrid_Model/
│
└── main.py # 전체 실험 실행 파일

---

## 📊 데이터 전처리 전략 (Data Preprocessing)

본 연구에서는 무료 금융 데이터(Yahoo Finance)의 한계(과거 재무제표 데이터의 부재)를 극복하고, 강화학습(RL) 에이전트에게 **고빈도 시장 변동성(High-frequency Market Dynamics)** 정보를 풍부하게 제공하기 위해 **가격 및 거래량 기반의 기술적 대용 지표(Technical Proxies)**를 5-Factor로 재정의하여 사용하였습니다.

### 🛠️ 5-Factor 정의 (기술적 대용 지표)

| 팩터 명칭 | 전통적 개념 (Fama-French) | **본 연구의 구현 (Technical Proxy)** | AI 학습 관점의 선정 근거 |
| :--- | :--- | :--- | :--- |
| **Value** | PBR (주가순자산비율) | **Technical Reversion (낙폭과대)**<br>수식: `1 - Rank(현재가 / 52주 최고가)` | 고정된 장부가 대신, **평균 회귀(Mean Reversion)** 원리를 이용하여 저가 매수 기회를 포착함. |
| **Size** | Market Cap (시가총액) | **Liquidity (저유동성/소외주)**<br>수식: `1 - Rank(일일 거래대금 로그값)` | 거래대금이 적은 종목의 **소형주 효과(Small-cap Effect)**를 모사하며, 매일 변동하는 유동성 리스크를 반영함. |
| **Momentum** | 12개월 수익률 | **Momentum (12M)**<br>수식: `Rank(12개월 수익률)` | 전통적 정의와 동일. 상승 추세가 강한 종목에 높은 가중치 부여. |
| **Volatility** | 이익 변동성 | **Low Volatility (저변동성)**<br>수식: `1 - Rank(일일 변동성)` | 주가 변동성이 낮아 안정적인 자산을 선호하도록 학습 유도. |
| **Beta** | 시장 베타 | **Low Beta (시장 방어력)**<br>수식: `1 - Rank(1년 롤링 베타)` | 하락장에서 시장 민감도가 낮아 방어주(Defensive Stock) 성격을 띠는 종목 선별. |

> **참고**: 모든 팩터는 `Rank(pct=True)`를 통해 `0.0` ~ `1.0` 사이로 정규화(Normalize)되며, 매일 날짜별(Cross-Sectional) 상대 순위를 사용하여 그래프 신경망의 관계 학습에 최적화되었습니다.

---

## 🧪 실험 환경 및 구성

**기간**: 2015년 ~ 2025년 (10년)
**대상**: NYSE/NASDAQ 상위 10개 기술주 및 우량주 (AAPL, MSFT, NVDA, TSLA 등)

### 1. TGNN-Only 모델
- **목적**: 주식 간의 상관관계를 그래프로 모델링하여 미래 순위 예측.
- **구조**: 노드(종목)와 엣지(상관관계)로 구성된 그래프 네트워크 학습.

### 2. DDPG-Only 모델
- **목적**: 시장 상태(State)에 따라 최적의 포트폴리오 비중(Action) 결정.
- **구조**: Actor-Critic 구조를 활용한 연속적 행동 공간 제어.

### 3. Hybrid (TGNN+DDPG) 모델 **(제안 방법)**
- **목적**: TGNN의 관계 추출 능력과 DDPG의 동적 최적화 능력을 결합.
- **메커니즘**:
    1. **TGNN**이 시장 데이터에서 종목 간 잠재 특징(Latent Features) 추출.
    2. **DDPG**가 추출된 특징을 상태(State)로 받아 최종 투자 비중 결정.

---

## 🚀 실행 방법

### 1. 필요 라이브러리 설치
pip install -r requirements.txt

### 2. 데이터 전처리 실행
`data/` 폴더에 `processed_daily_5factor_model.csv` 파일이 생성됩니다.
python preprocessing/DataPreprocessing.py

### 3. 전체 모델 백테스팅 실행
3가지 모델을 순차적으로 실행하며, 결과는 `results/` 폴더에 저장됩니다.
python main.py

---

## 📈 예상 결과물
실행이 완료되면 `results/` 디렉토리에서 다음을 확인할 수 있습니다:
- **누적 수익률 그래프** (`.png`)
- **성과 지표** (Sharpe Ratio, MDD, CAGR 등)
- **comparison_report.csv**: 모델별 성능 요약 비교표
