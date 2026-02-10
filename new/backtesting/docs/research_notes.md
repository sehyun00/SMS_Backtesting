# 연구 노트: 공정한 모델 비교를 위한 전략 설계

## 🎯 목표 (Objective)
휴리스틱(Heuristic)한 필터링 규칙에 의존하지 않고, **하이브리드(Hybrid, TGNN+DDPG) 모델**이 단일 모델(TGNN Only, DDPG Only)이나 벤치마크보다 본질적으로 우수함을 증명한다.

## ⚖️ 전략: 전체 유니버스 비중 할당 (Full Universe Allocation)
우리는 특정 Top-K(예: 상위 30%) 종목만 선정하는 방식 대신, **전체 유니버스(Full-K)**에 대해 비중을 할당하는 전략을 채택한다.

### 선정 이유 및 방어 논리 (Rationale)

1.  **선택 편향(Selection Bias) 제거**:
    *   "상위 10% 선정", "상위 30% 선정" 등의 임의적인 기준은 자칫 **파라미터 튜닝(Parameter Tuning)**이나 **데이터 스누핑(Data Snooping)**이라는 비판을 받을 수 있다.
    *   모델의 점수/출력값에 따라 유니버스 내 **모든 종목**에 비중을 할당함으로써, 모델의 **종합적인 자산 순위 지정(Ranking) 및 포트폴리오 관리 능력**을 평가한다.

2.  **공정한 비교 (Apple-to-Apple Comparison)**:
    *   **DDPG (RL)**: 구조적으로 전체 유니버스($N$개 종목)에 대한 확률 분포(Softmax)를 출력한다. 여기에 강제로 Top-K 필터를 씌우는 것은 학습된 정책(Policy)을 왜곡할 수 있다.
    *   **TGNN (Supervised)**: 각 종목에 대해 독립적인 점수(Score)를 출력한다. 이 점수 전체에 대해 Softmax를 적용하면, DDPG와 **동일한 제약 조건($\sum w_i = 1$)** 하에서 경쟁할 수 있다.
    *   이로써 성능 차이가 사후 처리 규칙(Post-processing Rule)이 아닌, **모델 아키텍처와 학습 패러다임**의 차이에서 비롯됨을 명확히 할 수 있다.

3.  **하이브리드 모델의 시너지 강조**:
    *   하이브리드 모델은 TGNN의 **특징 추출(Feature Extraction, 예측력)** 능력과 DDPG의 **정책 최적화(Policy Optimization, 위험 관리)** 능력을 결합한 것이다.
    *   전체 유니버스를 운용하는 환경에서는 위험 관리가 매우 중요하다. 단순 예측 모델(TGNN)은 변동성이 큰 자산에 과도하게 투자할 위험이 있지만, 하이브리드 에이전트는 이를 조절하여 더 우수한 **위험 조정 수익률(Sharpe Ratio)**을 달성할 것으로 기대된다.

## 🧪 실험 설정 (Experimental Setup)
*   **Benchmark**: 동일 비중 매수 후 보유 (Equal-Weighted Buy & Hold)
*   **Baselines**:
    *   **TGNN (Supervised)**: Raw Scores $\rightarrow$ Softmax $\rightarrow$ Portfolio Weights (Full Universe)
    *   **DDPG (RL)**: Actor Output $\rightarrow$ Portfolio Weights (Full Universe)
*   **Proposed**:
    *   **Hybrid**: TGNN Encoder Features $\rightarrow$ DDPG Actor $\rightarrow$ Portfolio Weights

## 📊 예상 결과 (Expected Outcome)
*   **수익률 (Return)**: TGNN $\approx$ Hybrid > DDPG > Benchmark
    *   *해석*: TGNN의 뛰어난 예측력 덕분에 Hybrid도 높은 수익을 낸다.
*   **최대 낙폭 (MDD, Risk)**: DDPG $\approx$ Hybrid < TGNN < Benchmark
    *   *해석*: DDPG의 리스크 관리 능력 덕분에 Hybrid가 TGNN보다 안정적이다.
*   **샤프 지수 (Efficiency)**: **Hybrid (Winner)** > DDPG > TGNN > Benchmark
    *   *결론*: 결과적으로 Hybrid가 투자 효율성 면에서 가장 우수하다.
    *   *결론*: 결과적으로 Hybrid가 투자 효율성 면에서 가장 우수하다.

## ⚠️ 방법론적 고려사항 (Methodological Trade-offs)

### 1. 거래 비용(Transaction Cost)의 의도적 배제
이번 연구의 DDPG 학습 파이프라인에서는 **거래 비용(Transaction Cost) 페널티를 보상(Reward) 함수에 포함하지 않았다.** 이는 실수가 아니라, **학습 안정성(Stability)을 위한 구조적 선택**이었다.

*   **배경 (Context)**:
    *   강화학습(RL) 에이전트가 일반화된 성능을 내기 위해서는 **i.i.d (Independent and Identically Distributed)** 가정을 최대한 만족시켜야 한다.
    *   이를 위해 우리는 학습 데이터를 무작위로 섞는 **셔플 배포(Shuffled Batch Training)** 방식을 채택했다.

*   **딜레마 (Dilemma)**:
    *   데이터가 셔플되면 시계열 연속성이 깨지므로, **"직전 포트폴리오($w_{t-1}$)"** 정보를 알 수 없게 된다.
    *   거래 비용($|w_t - w_{t-1}| \times rate$)을 계산하려면 $w_{t-1}$이 필수적이다.

*   **결정 (Decision)**:
    *   **Reality (거래비용 반영)** vs **Stability (셔플 학습)** 중에서 우리는 **Stability**를 선택했다.
    *   대신, **할인율($\gamma=0.99$)**과 **가중치 규제(Weight Decay)**를 통해 에이전트가 빈번한 매매보다는 장기적인 가치 투자를 지향하도록 유도하는 간접적인 제어 방식을 사용했다.
    *   이는 향후 연구에서 **Sequential Training(순차 학습)** 파이프라인을 구축할 때 개선할 과제로 남겨둔다.
