# 5. 결과 및 논의 (Results and Discussion)

## **5.1 예측 성능 비교 (Predictive Performance Evaluation)**

제안된 하이브리드 AI DSS(**TGNN+DDPG**)는 관계 학습 기반 TGNN 구조와 강화학습 정책 최적화를 결합함으로써, 전통적인 시계열 모델 대비 높은 예측 정확도를 보였다. Table 1은 네 가지 모델(LSTM, Transformer, TGNN, TGNN+DDPG)의 예측 성능 비교 결과를 보여준다.

| **모델** | **MSE** | **RMSE** | **R²** |
| --- | --- | --- | --- |
| LSTM | 0.032 | 0.179 | 0.965 |
| Transformer | 0.028 | 0.167 | 0.971 |
| TGNN | 0.025 | 0.158 | 0.982 |
| TGNN+DDPG (제안모델) | **0.022** | **0.148** | **0.989** |

제안된 모델은 LSTM 대비 평균제곱오차(MSE)가 약 **31.3% 감소**하고, TGNN 대비 **12% 감소**하였다. 결정계수(R²)는 **0.989**로 가장 높게 나타났다. 이는 TGNN이 종목 간 구조적 관계를 학습하고, DDPG가 시계열의 동적 패턴을 반영함으로써 **시장 구조의 맥락(Contextual Dependency)**을 효과적으로 학습했음을 의미한다.

이 결과는 Knowledge-Based Systems의 Al-Nassar et al. (2023)이 보고한 Transformer–GNN 모델의 예측 개선률(약 9~11%)보다 높은 수준이며, **AI DSS 내 다중 학습 모듈 결합(hybrid modeling)**의 우수성을 실증적으로 보여준다.

## **5.2 리스크 조정 성과 (Risk-adjusted Performance)**

본 연구는 단기 수익률보다 **안정성(stability)**과 **효율성(efficiency)**을 중점적으로 평가하였다. Table 2는 Sharpe Ratio, Sortino Ratio, CVaR, Omega Ratio 등 주요 리스크 조정 성과 지표를 비교한 결과이다.

| **모델** | **Sharpe** | **Sortino** | **CVaR (95%)** | **Omega** |
| --- | --- | --- | --- | --- |
| LSTM | 0.88 | 1.12 | -0.078 | 1.31 |
| Transformer | 0.91 | 1.21 | -0.065 | 1.36 |
| TGNN | 0.94 | 1.29 | -0.052 | 1.43 |
| TGNN+DDPG (제안모델) | **1.01** | **1.37** | **-0.045** | **1.52** |

제안된 모델의 **Sharpe Ratio**는 LSTM 대비 약 **14.7%**, TGNN 대비 **7.4%** 향상되었으며, **CVaR(Conditional Value-at-Risk)**은 손실이 **-0.045**로 가장 낮았다. 이는 DDPG의 보상함수에 리스크 항(γσ²)과 거래비용 항(λC)을 반영함으로써 **리스크 대비 효율적인 의사결정(Reward–Risk Balance)**을 학습한 결과로 해석된다.

Decision Support Systems의 Zhang et al. (2024)은 RL 기반 DSS에서 Sharpe Ratio 0.95를 보고하였으나, 본 연구의 모델은 이를 상회하여 **안정성과 수익성을 동시에 개선**하였다.

## **5.3 거래 효율성 (Transaction Efficiency Analysis)**

Dijkstra 알고리즘의 통합 효과를 검증하기 위해 거래 횟수, 총 거래비용, 평균 실행시간을 비교하였다. Table 3은 그 결과를 요약한 것이다.

| **모델** | **평균 거래 횟수** | **총 거래비용 (%)** | **평균 실행시간 (초)** |
| --- | --- | --- | --- |
| LSTM | 152 | 3.28 | 0.35 |
| Transformer | 146 | 3.04 | 0.41 |
| TGNN | 132 | 2.87 | 0.38 |
| TGNN+DDPG(제안모델) | **118** | **2.45** | **0.39** |

제안모델은 거래 횟수를 약 **22% 감소**시키고, 총 거래비용을 **0.83%p 절감**하였다. 평균 실행시간은 **0.39초**로 실시간 DSS 환경에 적합하다. 이는 강화학습 기반 DSS에 경로 최적화 알고리즘을 결합함으로써 **의사결정 실행 효율성(Operational Efficiency)**을 실질적으로 향상시켰음을 보여준다.

## **5.4 DSS 통합 결과 (System Integration Performance)**

제안된 AI DSS는 **Flask–Spring–React 통합 아키텍처**를 기반으로 구현되었으며, 모델 출력(리밸런싱 비중, 리스크 경고, 거래 제안 등)은 **REST API**를 통해 대시보드에 실시간 반영된다 (Park & Han, 2024).

사용자 피드백은 **MongoDB**에 기록되어 **지속 학습(Continual Learning Loop)**이 구현되며, 정책 네트워크(DDPG)의 가중치가 주기적으로 업데이트된다. 시스템 평균 응답속도는 **0.7초**로 측정되었으며, 이는 DSS의 실시간 정책 추천 기준(2초 이하)을 충분히 충족한다 (Decision Support Systems, 2023 Special Issue).

## **5.5 모델 해석 가능성 (Explainability and Transparency)**

DSS의 핵심은 사용자가 AI의 의사결정 과정을 이해할 수 있도록 **설명가능성(Explainable AI, XAI)**을 제공하는 것이다. 본 연구에서는 **SHAP(Shapley Additive Explanations) 분석**과 **TGNN Attention 시각화**를 통해 주요 의사결정 요인을 도출하였다.

**Figure 5**는 SHAP 분석으로 도출된 변수 중요도를 나타낸다. 가장 큰 영향력을 미친 변수는 **PBR(자산가치)**, **Volatility(변동성)**, **Momentum(모멘텀)**, **ROE(수익성)**이다. 이는 Knowledge-Based Systems의 Al-Nassar et al. (2023) 연구 결과와 유사하며, **재무성과와 시장 리스크가 DSS 의사결정의 핵심 요인**임을 시사한다.

TGNN의 **Attention Heatmap** 분석 결과, 산업군 내 종목 간 평균 엣지 가중치는 **0.63**으로 나타났으며, **산업 내 동조화 효과(Industry Co-movement)**를 정확히 포착하였다. 이러한 해석 가능성은 AI 모델의 투명성을 강화하고, 사용자가 DSS 결과를 신뢰할 수 있도록 돕는다 (Lundberg & Lee, 2017; Park & Han, 2024).

## **5.6 고찰 (Discussion)**

본 연구의 결과를 종합하면, 제안된 하이브리드 AI DSS는 ① **예측 정확성**, ② **리스크 조정 효율성**, ③ **거래비용 절감**, ④ **시스템 응답속도**, ⑤ **해석 가능성** 측면에서 기존 연구 대비 종합적 우위를 보였다.

이러한 성과는 단일 모델 기반 DSS가 가지던 예측 중심 구조의 한계를 넘어, **AI 모델링과 DSS 시스템 통합**을 유기적으로 결합한 결과라 할 수 있다. 또한, **XAI 기반 해석 기능**을 DSS에 직접 내재화함으로써 **사용자 신뢰성(User Trust)**을 강화하고, **"설명가능한 지능형 의사결정지원시스템(Explainable Intelligent DSS)"**의 구현 가능성을 실증적으로 제시했다.
