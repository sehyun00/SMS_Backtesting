# Temporal Graph Neural Network와 강화학습을 통합한 하이브리드 인공지능 기반 포트폴리오 리밸런싱 의사결정지원 프레임워크

A Hybrid AI-Based Decision Support Framework Integrating Temporal Graph Neural Network and Reinforcement Learning for Portfolio Rebalancing

JoongHyun Park, Sehyun Kim, Junghyun Back, Hamin Kim, Hyun Lee

## Abstract

디지털 전환과 인공지능(AI) 기술의 급속한 발전으로 의사결정 환경의 복잡성과 불확실성이 빠르게 증가하고 있다. 이에 따라 AI 기반 의사결정지원시스템(Decision Support Systems, DSS)의 중요성이 높아지고 있다. 본 연구는 금융 시장과 같이 동적이고 비정형적인 환경에서 지능적 의사결정을 지원하기 위한 신뢰할 수 있는(Trustworthy) 하이브리드 AI 기반 DSS 프레임워크를 제안한다.

제안된 시스템은 Temporal Graph Neural Network (TGNN)와 Deep Deterministic Policy Gradient (DDPG)를 고정 알파(Fixed Alpha, α = 0.5) 균등 앙상블로 통합하여 포트폴리오 리밸런싱(Portfolio Rebalancing) 문제를 해결한다. 특히 본 연구는 단순한 성과 향상을 넘어, AI 의사결정의 재현성(Reproducibility)과 신뢰성(Reliability)을 보장하기 위해 5개 독립 시드를 사용한 **다중 시드 재현성 프로토콜(Multi-Seed Reproducibility Protocol)**을 도입하였다.

S&P 500 구성 종목을 대상으로 약 10년간(2015–2024)의 데이터를 활용한 5-seed 반복 실험 결과, 제안된 Hybrid 모델은 **분기(Quarterly) 리밸런싱 주기에서 CAGR 9.09 ± 2.86%로 Benchmark(6.35%) 대비 +2.74%p의 안정적 초과 수익**을 달성하였다. DDPG 단독 모델이 강화학습 Critic 불안정성으로 인해 전 주기에서 음수 수익을 기록한 반면, Hybrid 모델은 TGNN의 관계 예측 신호가 DDPG의 불안정성을 효과적으로 보완하여 안정적인 초과 성과를 시현하였다. 또한 Flask 기반 AI 서버와 Spring–React 사용자 인터페이스를 통합하여 실시간 DSS 환경을 구현하였으며, TGNN Attention 기반 설명가능성(Explainability) 기법을 통해 모델의 의사결정 과정을 투명하게 검증하였다.

본 연구는 예측 중심의 기존 DSS를 넘어, 관계 인식형(Structure-Aware)·재현 가능한(Reproducible)·설명 가능한(Explainable) 지능형 의사결정지원시스템의 새로운 표준을 제시한다.

**Keywords**: Decision Support Systems (DSS); Graph Neural Network (GNN); Reinforcement Learning (RL); Deep Deterministic Policy Gradient (DDPG); Temporal Graph Neural Network (TGNN); Reproducibility; Explainable Artificial Intelligence (XAI).

**저널**: Decision Support Systems (Elsevier, SCIE, IF≈6.8)

**논문 주제**: Temporal Graph Neural Network와 강화학습을 통합한 하이브리드 인공지능 기반 포트폴리오 리밸런싱 의사결정지원 프레임워크

**초점**: AI 모델링 중심 + DSS 통합 응용

**형식**: IMRaD 구조 (서론–관련연구–방법론–실험–결과–결론)

---

# 1. 서론

AI 기반 디지털 전환(Digital Transformation)이 가속화되는 환경에서, 의사결정의 복잡성과 불확실성이 빠르게 증가하고 있다. 이러한 변화에 따라 인공지능(AI) 기반 의사결정지원시스템(Decision Support Systems, DSS)의 중요성이 한층 높아지고 있다.

특히 금융 시장은 데이터의 복잡성, 비정형성, 그리고 높은 변동성으로 인해 전통적인 통계 기반 의사결정모형의 한계가 명확히 드러나고 있다(Park & Han, 2024; Chen & Goetzmann, 2020). 이에 따라 복합 데이터 구조를 처리할 수 있는 딥러닝(Deep Learning) 및 강화학습(Reinforcement Learning) 기반의 DSS 연구가 활발히 이루어지고 있으며, 최근에는 종목 간 상관관계나 시장 네트워크 구조를 학습할 수 있는 그래프 신경망(Graph Neural Network, GNN)과 정책 최적화를 수행하는 강화학습(Reinforcement Learning, RL)을 결합한 하이브리드 DSS 프레임워크가 새로운 패러다임으로 주목받고 있다 (Al-Nassar et al., 2023; Zhang et al., 2024).

## 1.1 연구 배경 및 필요성

기존의 금융 DSS는 대부분 회귀분석, ARIMA, 혹은 LSTM 기반의 단일 시계열 예측 모델에 의존해왔다. 이러한 접근법은 개별 종목의 수익률 예측에는 효과적이지만, 시장 구조적 상호작용, 다차원적 리스크 요인, 동적 정책 최적화를 반영하기 어렵다는 한계를 지닌다 (Wu et al., 2021).

최근 Decision Support Systems 저널은 AI 기반 DSS의 구조적 진화를 강조하고 있으며, 특히 비정형·네트워크 데이터 기반의 "관계형 DSS(Relational DSS)" 개념이 전통적인 규칙 기반 시스템의 한계를 극복할 새로운 접근으로 제시되고 있다. 또한 단순한 예측 성능뿐만 아니라, AI 모델의 결과를 신뢰할 수 있는지에 대한 "신뢰성(Trustworthiness)" 문제가 중요한 화두로 떠오르고 있다.

## 1.2 최근 연구 동향

AI DSS 관련 최신 연구는 크게 두 가지 방향으로 발전하고 있다.

첫째, AI 모델링 중심 DSS 연구에서는 머신러닝·딥러닝 모델을 DSS 구조에 통합하여 의사결정의 정확성과 적응성을 향상시키는 방향이 주목받고 있다. 예를 들어, Zhang et al. (2024)은 Transformer 기반 DSS를 통해 비정형 금융 텍스트 데이터를 통합하여 의사결정 품질을 높였으며, Lin et al. (2023)은 강화학습을 적용해 리스크 조정 수익률을 향상시키는 정책학습 기반 포트폴리오 DSS를 제안하였다.

둘째, AI DSS 응용 및 통합 연구에서는 AI 모듈을 DSS 환경 내에서 실시간으로 동작시키는 구조적 프레임워크 설계가 활발히 진행되고 있다. Park & Han (2024)은 Flask 기반 API 서버를 활용한 AI-DSS 통합 구조를 제안하여, 사용자 피드백 루프를 포함한 지속 학습(Continual Learning) 프로세스를 구현하였다.

그러나 이러한 연구들은 주로 모델의 수익률(Profitability)이나 예측 정확도(Accuracy) 향상에 집중하는 경향이 있으며, AI 모델이 일관된 결과를 산출하는지(Reproducibility), 다양한 환경에서도 신뢰할 수 있는지(Reliability)에 대한 검증은 상대적으로 부족한 실정이다.

## 1.3 연구 공백 및 문제 제기

이러한 발전에도 불구하고 다음과 같은 한계가 여전히 존재한다.

1. 대부분의 연구가 모델의 성능 극대화에 치중하여, 실험 결과의 재현성(Reproducibility)과 신뢰성 검증을 간과하였다.
2. 기존 DSS는 AI 모델을 예측 모듈로만 활용하여 의사결정의 상호작용 구조(예: 종목 간 관계)를 반영하지 못했다.
3. 강화학습 기반 DSS 연구들은 예측 모듈과 정책 모듈을 독립적으로 운용하여, 시장 구조 학습과 포트폴리오 최적화를 투자 주기(Horizon)에 따라 동적으로 결합하는 앙상블 메커니즘이 부재하였다.

따라서, 관계 학습(Relationship Learning), 정책 학습(Policy Optimization)뿐만 아니라 신뢰할 수 있는 재현성(Reproducibility)을 갖춘 통합 AI DSS 프레임워크의 필요성이 제기된다.

## 1.4 연구 목적 및 기여

본 연구는 위의 한계를 해결하기 위해, Temporal Graph Neural Network (TGNN)와 Deep Deterministic Policy Gradient (DDPG) 알고리즘을 통합하고, 엄격한 재현성 프로토콜을 적용한 신뢰할 수 있는 하이브리드 AI DSS를 제안한다.

이 시스템은 다음 세 단계로 구성된다:

1. TGNN을 통해 시장 내 관계 구조를 학습하고,
2. DDPG를 통해 포트폴리오 비중 조정 정책을 강화학습 기반으로 최적화하며,
3. Deterministic Algorithm 및 Seed Fixing을 통해 실험의 완전한 재현성을 보장한다.

본 연구는 S&P 500 구성 종목 중 GICS 섹터별 대표 종목을 1개씩 추출하여 10개 종목으로 실험을 수행하였으며, 이러한 종목 선정 방식은 연구 목적과 투자 대상의 특성을 반영한 의도적 설계이다.

본 모델의 목적은 소형주 발굴이나 알파 종목 탐색이 아닌, 대형 우량주 중심의 포트폴리오에서 리밸런싱 비율을 안정적으로 최적화하는 데 있으며, 실험 유니버스를 S&P 500 구성 종목으로 한정하는 것은 실제 기관 투자자 및 개인 투자자의 투자 행태와 일치하는 현실적 설정이다.
실험 전체 기간(2006–2025) 동안 상장이 유지된 종목만을 선정한 것은 Survivorship Bias에 대한 우려를 고려한 결과이나, S&P 500은 시가총액, 유동성, 재무건전성 등 엄격한 편입 기준을 충족한 종목만으로 구성되어 해당 종목의 상장폐지 발생 빈도가 전체 시장 대비 현저히 낮다. 실제로 Russell 3000 기준 연간 상장폐지율이 약 3–5% 수준인 데 반해, S&P 500 편입 종목의 연간 상장폐지율은 0.1% 미만으로 보고되며, S&P 500은 구성 종목을 분기별로 재검토하여 재무 기준 미달 종목을 선제적으로 교체하므로 개별 종목의 완전한 상장폐지보다는 지수 제외(index exclusion) 형태로 리스크가 관리된다. 이처럼 S&P 500 유니버스 내에서의 생존편향은 전체 시장 대비 그 영향이 제한적이며, 본 연구의 핵심 평가 지표인 포트폴리오 수익률과 샤프 지수에 미치는 편향 효과는 최소화된다. 나아가 본 연구는 개별 종목의 절대 수익을 예측하는 것이 아니라 종목 간 rebalancing ratio를 학습하는 모델을 평가하므로, 생존편향에 의한 절대 수익 과대추정 문제가 직접적으로 적용되지 않는다. 또한 섹터 다각화를 통해 특정 산업군에 편중되지 않는 일반화 가능한 포트폴리오를 구성하였다.

이를 통해 본 연구는 다음과 같은 기여를 한다:

① **신뢰할 수 있는 AI DSS 표준 제시**: 단순 성능 우위를 넘어, 결정론적(Deterministic) 알고리즘과 시드 고정(Seed Fixing)을 통해 언제나 검증 가능한 재현성(Reproducibility)을 확보한 AI 연구 방법론을 제시한다.

② **AI 모델 통합형 DSS 설계**: 예측(TGNN)과 정책(DDPG)을 Fixed Alpha(α = 0.5) 균등 앙상블로 결합하는 통합형 DSS 아키텍처를 제안한다. 이는 동적 가중치 학습 방식 대비 시드 의존성을 제거하고 결정론적 재현성을 보장하는 설계 선택이다.

③ **AI DSS의 실시간 자동화 구현**: Flask-Spring-React 기반의 시스템 통합을 통해 사용자 피드백이 실시간으로 AI 정책 업데이트에 반영되는 구조를 구현한다.

④ **학문적·실무적 시사점 제시**: 학문적으로는 AI DSS의 구조적 확장을 제시하고, 실무적으로는 금융 의사결정에서 설명가능(Explainable)하고 자동화된 지능형 DSS(Automated Intelligent DSS)의 가능성을 입증한다.
