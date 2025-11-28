# SMS_Backtesting: TGNN-DDPG 하이브리드 포트폴리오 최적화

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Research-Paper-orange.svg)](https://www.notion.so/Temporal-Graph-Neural-Network-2a082e91118d809aa283fb97ed6c4ac9)

## 📖 프로젝트 개요

본 저장소는 연구 논문 **"TGNN-DDPG 하이브리드 모델을 활용한 포트폴리오 리밸런싱 전략 강화"** 의 구현 및 백테스팅 프레임워크입니다.

### 연구 배경

전통적인 포트폴리오 최적화 방법론은 정적 분석에 의존하며, 시장의 동적 변화를 충분히 반영하지 못합니다. 본 연구는 **시계열 그래프 신경망 (TGNN)** 과 **심층 강화학습 (DDPG)** 을 결합하여 다음을 달성합니다:

- 주식 간 복잡한 시간적 상관관계를 그래프 구조로 학습
- 시장 변화에 따른 동적 포트폴리오 리밸런싱
- 장기적 누적 수익률 극대화 및 리스크 관리

### 비교 실험 모델

제안하는 하이브리드 모델의 성능을 검증하기 위해 세 가지 모델을 비교합니다:

1. **TGNN-Only** : 시계열 그래프 신경망을 이용한 종목 간 관계 분석 및 순위 예측
2. **DDPG-Only** : 심층 결정론적 정책 그래디언트를 이용한 동적 비중 최적화
3. **Hybrid (TGNN+DDPG)** ⭐: TGNN의 잠재 특징을 DDPG가 활용하는 제안 모델

---

## 📂 디렉토리 구조

실험의 재현성 (Reproducibility) 과 모델 간 명확한 비교를 위해 다음과 같이 구조화되었습니다.

```
SMS_Backtesting/
│
├── 📁 data/                              # [공통] 데이터 저장소
│   ├── stock_list.csv                    # 대상 종목 리스트 (10개 우량주)
│   └── processed_daily_5factor_model.csv # 전처리 완료된 5팩터 데이터
│
├── 📁 preprocessing/                     # [공통] 데이터 엔지니어링 모듈
│   ├── DataPreprocessing.py              # 팩터 계산 및 데이터 정제 코드
│   └── README.md                         # 기술적 팩터 정의 및 산출 방식 상세 명세
│
├── 📁 models/                            # 모델 구현체
│   ├── 📁 TGNN/                          # 비교군 1: 그래프 신경망 단독 모델
│   ├── 📁 DDPG/                          # 비교군 2: 강화학습 단독 모델
│   └── 📁 Hybrid_TGNN_DDPG/              # 제안 모델 (Main Method)
│
├── 📁 results/                           # 실험 결과 (로그 및 그래프)
│   ├── 📁 01_TGNN_Only/
│   ├── 📁 02_DDPG_Only/
│   └── 📁 03_Hybrid_Model/
│
├── main.py                               # 전체 실험 실행 파일
├── requirements.txt                      # 필요 라이브러리 목록
└── README.md                             # 프로젝트 문서 (본 파일)
```

---

## 📊 데이터 전처리 전략

### 연구 방법론

무료 금융 데이터 (Yahoo Finance) 의 제약사항을 극복하고 강화학습 에이전트에 **고빈도 시장 역학 (High-frequency Market Dynamics)** 정보를 제공하기 위해, **가격 및 거래량 기반의 기술적 대용 지표 (Technical Proxies)** 를 5-Factor로 재정의하였습니다.

### 🛠️ 5-Factor 정의

전통적인 Fama-French 5-Factor 모델을 시장 데이터로 실시간 계산 가능한 기술적 지표로 변환하였습니다.

| 팩터 명칭 | 전통적 개념 | 본 연구의 구현 (Technical Proxy) | AI 학습 관점의 선정 근거 |
|:---------|:-----------|:------------------------------|:---------------------|
| **Value** | PBR (주가순자산비율) | **Technical Reversion**<br>`1 - Rank(현재가 / 52주 최고가)` | 평균 회귀 원리를 이용한 저가 매수 기회 포착 |
| **Size** | Market Cap (시가총액) | **Liquidity**<br>`1 - Rank(log(일일 거래대금))` | 소형주 효과 모사 및 유동성 리스크 반영 |
| **Momentum** | 12개월 수익률 | **Momentum (12M)**<br>`Rank(12개월 수익률)` | 상승 추세가 강한 종목에 높은 가중치 부여 |
| **Volatility** | 이익 변동성 | **Low Volatility**<br>`1 - Rank(일일 변동성)` | 안정적 자산을 선호하도록 학습 유도 |
| **Beta** | 시장 베타 | **Low Beta**<br>`1 - Rank(1년 롤링 베타)` | 하락장에서 방어주 성격 종목 선별 |

> **정규화 방법** : 모든 팩터는 `Rank(pct=True)`를 통해 0.0 ~ 1.0 사이로 정규화되며, 매일 날짜별 횡단면 (Cross-Sectional) 상대 순위를 사용하여 그래프 신경망의 관계 학습에 최적화되었습니다.

---

## 🧪 실험 설정

### 백테스팅 기간 및 대상

- **기간** : 2015년 1월 ~ 2025년 11월 (약 10년)
- **대상 종목** : NYSE/NASDAQ 상위 10개 기술주 및 우량주
  - AAPL (Apple), MSFT (Microsoft), NVDA (NVIDIA), TSLA (Tesla) 등
- **리밸런싱 주기** : 월간 (Monthly)
- **초기 자본** : $10,000

### 모델별 상세 설명

#### 1. TGNN-Only 모델

**목적** : 주식 간 상관관계를 그래프로 모델링하여 미래 순위 예측

**핵심 메커니즘** :
- 노드 (Node): 개별 종목
- 엣지 (Edge): 종목 간 상관관계 (동적 계산)
- 출력: 다음 기간 종목 순위 예측

#### 2. DDPG-Only 모델

**목적** : 시장 상태에 따른 최적 포트폴리오 비중 결정

**핵심 메커니즘** :
- State: 5-Factor 시계열 데이터
- Action: 각 종목의 투자 비중 (연속 공간)
- Reward: Sharpe Ratio 기반 위험 조정 수익률

#### 3. Hybrid (TGNN+DDPG) 모델 ⭐ **(제안 방법)**

**목적** : TGNN의 관계 추출과 DDPG의 동적 최적화 결합

**핵심 메커니즘** :
1. **Feature Extraction** : TGNN이 시장 데이터에서 종목 간 잠재 특징 (Latent Features) 추출
2. **Policy Optimization** : DDPG가 추출된 특징을 State로 받아 최종 투자 비중 결정
3. **End-to-End Learning** : 두 모델이 통합 손실 함수로 동시 학습

---

## 🚀 실행 방법

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/sehyun00/SMS_Backtesting.git
cd SMS_Backtesting

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필요 라이브러리 설치
pip install -r requirements.txt
```

### 2. 데이터 전처리

```bash
python preprocessing/DataPreprocessing.py
```

실행 후 `data/processed_daily_5factor_model.csv` 파일이 생성됩니다.

### 3. 모델 백테스팅

```bash
# 전체 모델 순차 실행
python main.py

# 개별 모델 실행 (선택 사항)
python models/TGNN/train.py
python models/DDPG/train.py
python models/Hybrid_TGNN_DDPG/train.py
```

---

## 📈 결과 분석

### 출력 파일

실행 완료 시 `results/` 디렉토리에 다음 파일들이 생성됩니다:

- **누적 수익률 그래프** (`cumulative_returns.png`)
- **드로우다운 분석** (`drawdown_analysis.png`)
- **성과 지표 요약** (`performance_metrics.csv`)
- **모델 비교 리포트** (`comparison_report.csv`)

### 평가 지표

- **CAGR** (Compound Annual Growth Rate): 연평균 성장률
- **Sharpe Ratio** : 위험 조정 수익률
- **MDD** (Maximum Drawdown): 최대 낙폭
- **Win Rate** : 수익 발생 거래 비율
- **Volatility** : 포트폴리오 변동성

---

## 🔬 기술 스택

- **Deep Learning** : PyTorch, PyTorch Geometric
- **Reinforcement Learning** : Stable-Baselines3, Gym
- **Data Processing** : Pandas, NumPy
- **Visualization** : Matplotlib, Seaborn
- **Financial Data** : yfinance

---

## 📚 참고문헌

본 연구는 다음 논문들을 기반으로 합니다:

- Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR*.
- Lillicrap, T. P., et al. (2015). Continuous control with deep reinforcement learning. *ICLR*.
- Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*.

---

## 📝 라이선스

본 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

## 👥 기여자

- **연구진** : SMS 연구팀
- **논문 링크** : [Notion 연구 문서](https://www.notion.so/Temporal-Graph-Neural-Network-2a082e91118d809aa283fb97ed6c4ac9)

---

## 📧 문의

프로젝트 관련 문의사항은 [Issues](https://github.com/sehyun00/SMS_Backtesting/issues)를 통해 제출해주세요.

---

## 🔄 업데이트 이력

- **2025.11.28** : README.md 구조화 및 내용 보완
- **2025.11.XX** : 초기 프로젝트 생성

---

**⚠️ 면책 조항** : 본 프로젝트는 학술 연구 목적으로 제작되었으며, 실제 투자 조언이 아닙니다. 투자 결정에 대한 모든 책임은 투자자 본인에게 있습니다.
