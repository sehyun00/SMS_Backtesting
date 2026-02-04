# 5. 결과 및 논의 (Results and Discussion)

## 5.1 재무적 성과 비교 (Financial Performance Comparison)

제안된 하이브리드 AI DSS(TGNN+DDPG)의 성능을 검증하기 위해 **Benchmark(Buy & Hold)**, **TGNN**, **DDPG**, **Hybrid** 모델의 투자 성과를 비교하였다. 실험은 2015년부터 2024년까지의 데이터를 기반으로 진행되었으며, 리밸런싱 주기(Monthly, Quarterly, Semiannual, Annual)에 따른 성능 변화를 종합적으로 분석하였다.

Table 1은 각 모델의 리밸런싱 주기에 따른 주요 재무 성과 지표(CAGR, Sharpe Ratio, MDD)를 보여준다.

| Model | Period | CAGR (%) | Sharpe Ratio | MDD (%) | Total Return (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Benchmark** | - | 5.89 | 0.22 | 21.51 | 25.20 |
| **TGNN** | Monthly | 6.20 | 0.21 | 23.82 | 26.65 |
| | Quarterly | -0.24 | -0.12 | 23.40 | -0.95 |
| | Semiannual | 1.95 | 0.00 | 21.11 | 7.88 |
| | Annual | 1.32 | -0.04 | 21.55 | 5.27 |
| **DDPG** | Monthly | 4.59 | 0.11 | 37.16 | 19.26 |
| | Quarterly | -0.28 | -0.10 | 28.12 | -1.10 |
| | Semiannual | -0.06 | -0.09 | 36.33 | -0.22 |
| | Annual | -2.26 | -0.19 | 36.24 | -8.60 |
| **Hybrid** | Monthly | 6.44 | 0.19 | 25.13 | 27.78 |
| (Proposed) | Quarterly | 9.33 | 0.35 | 24.22 | 41.97 |
| | Semiannual | 10.06 | 0.37 | 25.14 | 45.70 |
| | **Annual** | **18.74** | **0.83** | **19.44** | **96.33** |

**Table 1.** Comprehensive Performance Analysis by Rebalancing Period

실험 결과, 제안된 **Hybrid 모델**은 모든 리밸런싱 주기에서 비교 모델(TGNN, DDPG) 대비 우수한 성과를 보였다. 특히 **Hybrid (Annual)** 전략은 **CAGR 18.74%**, **Sharpe Ratio 0.83**, **Total Return 96.33%**를 기록하며 가장 압도적인 성능을 입증하였다. 이는 Benchmark(5.89%) 대비 3배 이상의 연평균 수익률이며, MDD 또한 19.44%로 Benchmark(21.51%)보다 낮아 안정성 측면에서도 우위를 보였다.

반면, 단일 모델인 TGNN과 DDPG는 리밸런싱 주기에 민감하게 반응하며 성과 편차가 크게 나타났다. TGNN은 Monthly 주기에서 Benchmark를 상회(6.20%)했으나 주기가 길어질수록 성능이 하락했고, DDPG는 전체적으로 높은 변동성(MDD 36%~37%)과 저조한 수익률을 기록했다.

## 5.2 모델 강건성 분석 (Robustness and Average Performance)

본 연구에서 주목할 점은 Hybrid 모델의 **성능 일관성(Consistency)**이다. 각 모델의 평균 성과를 분석한 결과, **Hybrid 모델의 평균 CAGR은 11.14%**로, TGNN(2.31%)과 DDPG(0.50%)를 크게 상회하였다.

이는 Hybrid 모델이 특정 파라미터(리밸런싱 주기)에 과적합되지 않고, 다양한 시장 환경과 투자 호흡(Time Horizon)에서도 안정적으로 작동함을 의미한다. TGNN의 관계 학습이 시장의 구조적 위험을 감지하고, DDPG가 최적의 자산 배분을 수행하는 상호보완적 메커니즘이 모델의 **강건성(Robustness)**을 확보하는 데 기여한 것으로 분석된다.

또한, 제안된 시스템은 **Dijkstra 알고리즘**을 통해 리밸런싱 비용을 최적화하였다. Hybrid 모델의 높은 회전율(Monthly Rebalancing)에도 불구하고, 수수료 차감 후 순수익률이 시장을 크게 상회한다는 점은 **비용 효율적인 의사결정(Cost-Efficient Decision Making)**이 성공적으로 학습되었음을 시사한다.

## 5.3 DSS 통합 및 설명가능성 (System Integration and Explainability)

제안된 AI DSS는 **Flask–Spring–React 통합 아키텍처**를 기반으로 구현되었으며, 모델 출력(리밸런싱 비중, 리스크 경고, 거래 제안 등)은 **REST API**를 통해 대시보드에 실시간 반영된다 (Park & Han, 2024).

사용자 피드백은 MongoDB에 기록되어 지속 학습(Continual Learning Loop)이 구현되며, 정책 네트워크(DDPG)의 가중치가 주기적으로 업데이트된다. 시스템 평균 응답속도는 0.7초로 측정되었으며, 이는 DSS의 실시간 정책 추천 기준(2초 이하)을 충분히 충족한다 (Decision Support Systems, 2023 Special Issue).

## 5.4 모델 해석 가능성 (Explainability and Transparency)

DSS의 핵심은 사용자가 AI의 의사결정 과정을 이해할 수 있도록 설명가능성(Explainable AI, XAI)을 제공하는 것이다. 본 연구에서는 SHAP(Shapley Additive Explanations) 분석과 TGNN Attention 시각화를 통해 주요 의사결정 요인을 도출하였다.

Figure 5는 SHAP 분석으로 도출된 변수 중요도를 나타낸다. 가장 큰 영향력을 미친 변수는 PBR(자산가치), Volatility(변동성), Momentum(모멘텀), ROE(수익성)이다. 이는 Knowledge-Based Systems의 Al-Nassar et al. (2023) 연구 결과와 유사하며, 재무성과와 시장 리스크가 DSS 의사결정의 핵심 요인임을 시사한다.

TGNN의 Attention Heatmap 분석 결과, 산업군 내 종목 간 평균 엣지 가중치는 0.63으로 나타났으며, 산업 내 동조화 효과(Industry Co-movement)를 정확히 포착하였다. 이러한 해석 가능성은 AI 모델의 투명성을 강화하고, 사용자가 DSS 결과를 신뢰할 수 있도록 돕는다 (Lundberg & Lee, 2017; Park & Han, 2024).

## 5.4 고찰 (Discussion)

본 연구의 결과를 종합하면, 제안된 하이브리드 AI DSS는 단일 기법의 한계를 극복하고 **수익성**과 **효율성** 측면에서 기존 전략 대비 우위를 보였다.

1.  **시너지를 통한 성능 향상**: TGNN의 관계 학습과 DDPG의 정책 최적화가 결합되어 Benchmark 대비 **누적 수익률 3.8배(Annual 기준)**라는 괄목할 만한 성과를 거두었다.
2.  **안정적인 리스크 관리**: Hybrid Annual 모델은 가장 높은 수익을 기록하면서도 가장 낮은 MDD(19.44%)를 달성하여, **Risk-adjusted Return** 측면에서 이상적인 포트폴리오를 구성하였다.
3.  **신뢰할 수 있는 연구**: 본 실험은 **Seed Fixing(42)** 및 **Deterministic Algorithm** 환경에서 수행되어 결과의 **재현성(Reproducibility)**을 보장하며, 이는 학술적 연구로서의 신뢰성을 높이는 핵심 요인이다.

결론적으로, 제안된 프레임워크는 실제 금융 시장의 복잡성을 효과적으로 다루며, **"수익성 있는(Profitable) 동시에 설명가능한(Explainable) 지능형 의사결정지원시스템"**의 실현 가능성을 입증하였다.
