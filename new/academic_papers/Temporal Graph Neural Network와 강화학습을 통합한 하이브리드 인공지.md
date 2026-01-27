# Temporal Graph Neural Network와 강화학습을 통합한 하이브리드 인공지능 기반 포트폴리오 리밸런싱 의사결정지원 프레임워크

**A Hybrid AI-Based Decision Support Framework Integrating Temporal Graph Neural Network and Reinforcement Learning for Portfolio Rebalancing**

**JoongHyun Park, Sehyun Kim, Junghyun Back, Hamin Kim, Kyungsik Lee. Hyun Lee**

## **Abstract**

디지털 전환과 인공지능(AI) 기술의 급속한 발전으로 의사결정 환경의 복잡성과 불확실성이 빠르게 증가하고 있다. 이에 따라 AI 기반 의사결정지원시스템(Decision Support Systems, DSS)의 중요성이 높아지고 있다. 본 연구는 금융 시장과 같이 동적이고 비정형적인 환경에서 지능적 의사결정을 지원하기 위한 하이브리드 AI 기반 DSS 프레임워크를 제안한다.

제안된 시스템은 **Temporal Graph Neural Network (TGNN)**, **Deep Deterministic Policy Gradient (DDPG)**, 그리고 **Dijkstra 최적화 알고리즘**을 통합하여 포트폴리오 리밸런싱(Portfolio Rebalancing) 문제를 해결한다. TGNN 모듈은 그래프 합성곱과 시간적 어텐션 메커니즘을 통해 종목 간 동적 상관관계를 학습하며, DDPG 모듈은 리스크 및 거래비용 제약을 고려한 강화학습 기반 정책 최적화를 수행한다. 마지막으로 Dijkstra 알고리즘은 거래 경로를 탐색하여 실행 비용을 최소화한다.

10년간의 KOSPI 및 S&P500 데이터를 활용한 실험 결과, 제안된 모델은 기존 LSTM, Transformer, TGNN 모델 대비 평균제곱오차(MSE)가 **31% 감소**, 샤프비율(Sharpe Ratio) **1.01**, 거래빈도 **22% 감소**의 성과를 보였다. 또한 Flask 기반 AI 서버와 Spring–React 사용자 인터페이스를 통합하여 실시간 DSS 환경을 구현하였으며, SHAP 및 Attention 기반 설명가능성(Explainability) 기법을 통해 모델의 의사결정 과정을 시각적으로 검증하였다.

본 연구는 예측 중심의 기존 DSS를 넘어, **관계 인식형(Structure-Aware)·비용 효율적(Cost-Efficient)·설명 가능한(Explainable) 지능형 의사결정지원시스템(Intelligent Decision Support System)**의 새로운 방향을 제시한다.

**Keywords**: Decision Support Systems (DSS); Graph Neural Network (GNN); Reinforcement Learning (RL); Deep Deterministic Policy Gradient (DDPG); Temporal Graph Neural Network (TGNN); Portfolio Rebalancing; Explainable Artificial Intelligence (XAI); Hybrid AI Framework.

**저널**: Decision Support Systems (Elsevier, SCIE, IF≈6.8)

**논문 주제**: Temporal Graph Neural Network와 강화학습을 통합한 하이브리드 인공지능 기반 포트폴리오 리밸런싱 의사결정지원 프레임워크

**초점**: AI 모델링 중심 + DSS 통합 응용

**형식**: IMRaD 구조 (서론–관련연구–방법론–실험–결과–결론)

---

# 1. 서론

AI 기반 디지털 전환(Digital Transformation)이 가속화되는 환경에서, 의사결정의 복잡성과 불확실성이 빠르게 증가하고 있다. 이러한 변화에 따라 인공지능(AI) 기반 의사결정지원시스템(Decision Support Systems, DSS)의 중요성이 한층 높아지고 있다.

특히 금융 시장은 데이터의 복잡성, 비정형성, 그리고 높은 변동성으로 인해 전통적인 통계 기반 의사결정모형의 한계가 명확히 드러나고 있다(Park & Han, 2024; Chen & Goetzmann, 2020). 이에 따라 복합 데이터 구조를 처리할 수 있는 딥러닝(Deep Learning) 및 강화학습(Reinforcement Learning) 기반의 DSS 연구가 활발히 이루어지고 있으며, 최근에는 종목 간 상관관계나 시장 네트워크 구조를 학습할 수 있는 **그래프 신경망(Graph Neural Network, GNN)**과 정책 최적화를 수행하는 **강화학습(Reinforcement Learning, RL)**을 결합한 **하이브리드 DSS 프레임워크(Hybrid AI DSS Framework)**가 새로운 패러다임으로 주목받고 있다 (Al-Nassar et al., 2023; Zhang et al., 2024).

## **1.1 연구 배경 및 필요성**

기존의 금융 DSS는 대부분 회귀분석, ARIMA, 혹은 LSTM 기반의 단일 시계열 예측 모델에 의존해왔다. 이러한 접근법은 개별 종목의 수익률 예측에는 효과적이지만, **시장 구조적 상호작용**, **다차원적 리스크 요인**, **동적 정책 최적화**를 반영하기 어렵다는 한계를 지닌다 (Wu et al., 2021).

최근 Decision Support Systems 저널은 AI 기반 DSS의 구조적 진화를 강조하고 있으며, 특히 비정형·네트워크 데이터 기반의 **"관계형 DSS(Relational DSS)"** 개념이 전통적인 규칙 기반 시스템의 한계를 극복할 새로운 접근으로 제시되고 있다 (Lee & Kim, 2023; Park & Han, 2024). 또한 Knowledge-Based Systems (Al-Nassar et al., 2023)과 Applied Intelligence (Gu et al., 2025)에서는 강화학습 기반 DSS의 응용 사례를 통해 AI 기술이 단순한 예측을 넘어 **지능형 의사결정(Decision Intelligence)** 단계로 발전하고 있음을 보여주었다.

## **1.2 최근 연구 동향**

AI DSS 관련 최신 연구는 크게 두 가지 방향으로 발전하고 있다.

**첫째**, AI 모델링 중심 DSS 연구에서는 머신러닝·딥러닝 모델을 DSS 구조에 통합하여 의사결정의 정확성과 적응성을 향상시키는 방향이 주목받고 있다. 예를 들어, Decision Support Systems의 Zhang et al. (2024)은 Transformer 기반 DSS를 통해 비정형 금융 텍스트 데이터를 통합하여 의사결정 품질을 높였으며, Lin et al. (2023)은 Neural Computing & Applications에서 강화학습(DDPG)을 적용해 리스크 조정 수익률을 향상시키는 정책학습 기반 포트폴리오 DSS를 제안하였다.

**둘째**, AI DSS 응용 및 통합(System Integration) 연구에서는 AI 모듈을 DSS 환경 내에서 실시간으로 동작시키는 구조적 프레임워크 설계가 활발히 진행되고 있다. Park & Han (2024)은 Flask 기반 API 서버를 활용한 AI-DSS 통합 구조를 제안하여, 사용자 피드백 루프를 포함한 지속 학습(Continual Learning) 프로세스를 구현하였다. 또한 Gu et al. (2025)은 Applied Intelligence에서 강화학습 모듈을 의사결정 엔진(Decision Engine)으로 삽입해 실시간 포트폴리오 리밸런싱을 수행하는 DSS를 구축하였다.

이와 같이 최근 연구들은 단일 AI 모델을 활용한 DSS에서 벗어나, **GNN, RL, 최적화 알고리즘(Optimization Algorithm)**을 통합한 **Hybrid AI DSS** 방향으로 빠르게 확장되고 있다.

## **1.3 연구 공백 및 문제 제기**

이러한 발전에도 불구하고 다음과 같은 한계가 여전히 존재한다.

1. **기존 DSS는 AI 모델을 예측 모듈로만 활용**하여 의사결정의 상호작용 구조(예: 종목 간 관계, 시간적 의존성)를 반영하지 못했다.
2. *강화학습 기반 DSS 연구들은 거래비용이나 경로 제약 등 실행 단계의 제약조건(Operational Constraints)**을 통합적으로 고려하지 않았다.
3. **그래프 신경망(GNN) 기반 DSS 연구는 시장 네트워크의 구조를 학습**하지만, 이를 강화학습의 정책 최적화 단계에 직접 연결하지 못했다.

따라서, **관계 학습(Relationship Learning)**, **정책 학습(Policy Optimization)**, **비용 최적화(Transaction Cost Minimization)**를 통합한 AI DSS 프레임워크의 필요성이 제기된다.

## **1.4 연구 목적 및 기여**

본 연구는 위의 한계를 해결하기 위해, **Temporal Graph Neural Network (TGNN)**, **Deep Deterministic Policy Gradient (DDPG) 알고리즘**을 통합한 하이브리드 AI DSS를 제안한다.

이 시스템은 다음 두 단계로 구성된다:

1. **TGNN을 통해 시장 내 관계 구조를 학습**하고,
2. **DDPG를 통해 포트폴리오 비중 조정 정책을 강화학습 기반으로 최적화**

본 연구는 **한국인 투자자의 실제 매수 데이터를 기반으로 10개 대표 종목을 선정**하여 실험을 수행하였다. 종목 선정은 SEIBro의 거래량 데이터와 산업군 다각화 원칙을 결합하여, 실증성과 일반화 가능성을 동시에 확보하였다.

이를 통해 본 연구는 다음과 같은 기여를 한다:

**① AI 모델 통합형 DSS 설계**: 예측-정책-비용 최적화의 세 단계를 AI 모듈로 연결하는 통합형 DSS 아키텍처를 제안한다.

**② AI DSS의 실시간 자동화 구현**: Flask-Spring-React 기반의 시스템 통합을 통해 사용자 피드백이 실시간으로 AI 정책 업데이트에 반영되는 구조를 구현한다.

**③ 학문적·실무적 시사점 제시**: 학문적으로는 AI DSS의 구조적 확장을 제시하고, 실무적으로는 금융 의사결정에서 설명가능(Explainable)하고 자동화된 **지능형 DSS(Automated Intelligent DSS)**의 가능성을 입증한다.

# 2. 관련 연구 (Related Works)

## **2.1 AI 기반 의사결정지원시스템의 진화 (Evolution of AI-Based Decision Support Systems)**

최근 DSS 분야는 단순한 규칙 기반(rule-based) 시스템에서 인공지능(AI) 기반의 **지능형 의사결정지원(Intelligent Decision Support)**으로 빠르게 발전하고 있다. Decision Support Systems 저널의 Lee and Kim (2023)은 AI 기반 의사결정엔진을 도입함으로써 전통적 분석형 DSS의 한계를 보완할 수 있음을 보였다. Applied Intelligence의 Gu et al. (2025)는 강화학습(RL) 엔진을 DSS 아키텍처에 내재화하여 시장 변화에 실시간 대응할 수 있는 **AI-driven 의사결정엔진**을 제시하였다. 또한 Information Sciences의 Bai et al. (2023)은 딥러닝을 활용한 의사결정지원 프레임워크를 검증하며, AI 모델이 비정형 데이터에서 의미 추론과 정책 추천을 수행할 수 있음을 입증하였다.

이처럼 최근 연구들은 AI 기술을 DSS의 핵심 요소로 간주하며, **예측 정확성**뿐 아니라 **설명가능성(Explainability)** 및 **적응성(Adaptivity)**을 동시에 강조하는 방향으로 진화하고 있다 (Park & Han, 2024; Zhang et al., 2024). 그러나 대부분의 연구가 단일 AI 모델 중심 예측에 머물러 있으며, 다중 AI 모델 간 상호작용 또는 시스템 통합 구조에 대한 체계적 연구는 아직 부족하다.

## **2.2 강화학습 기반 의사결정지원 (Decision Support with Reinforcement Learning)**

강화학습(RL)은 비정형 환경에서 스스로 정책을 학습하는 능력으로 DSS 분야에 적극적으로 도입되고 있다. Neural Computing and Applications의 Lin et al. (2023)은 DDPG 모델을 적용한 DSS 프레임워크를 통해 비용-효율적인 포트폴리오 리밸런싱 정책을 학습하였으며, Knowledge-Based Systems의 Al-Nassar et al. (2023)은 Transformer 기반 정책 네트워크를 RL과 결합해 시계열 상관관계를 반영하였다.

최근 Liu et al. (2024)의 연구는 **Soft Actor-Critic (SAC)** 강화학습 알고리즘을 로봇 네비게이션에 적용하여 동적 장애물 환경에서의 의사결정을 효과적으로 해결하였다. 이 연구에서 제시된 **SAC 기반 경로 계획 시스템**은 continuous action space에서의 정책 최적화를 통해 실시간 환경 변화에 대응하는 의사결정을 수행한다. 특히 Liu et al.의 방법론에서 주목할 점은 **전문가 궤적 기반 모방학습(Expert Trajectory-based Imitation Learning)**과 **우선순위 경험 재생(Prioritized Experience Replay)** 기법을 결합하여 학습 효율성과 안전성을 동시에 향상시켰다는 점이다.

본 연구의 DSS 프레임워크는 Liu et al. (2024)의 SAC 알고리즘 구조를 금융 의사결정 영역에 적용하여 확장한다. 로봇 네비게이션에서의 **상태-행동-보상** 구조를 **시장 상태-포트폴리오 조정-수익률/리스크 보상**으로 변환하고, SAC의 **actor-critic 네트워크 구조**를 TGNN 임베딩과 결합함으로써 관계 인식형 정책 학습을 구현한다. 또한 Liu et al.의 **RNN 기반 전이 추론 능력(Transfer Inference with RNN)**에서 착안하여, 시장 변동성이 높은 환경에서도 안정적인 의사결정이 가능한 적응형 DSS를 설계한다.

Decision Support Systems의 Zhang et al. (2024)은 PPO와 TD3 알고리즘을 비교하여 AI 정책이 리스크 관리 측면에서도 효율적으로 작동함을 보였다.

이와 같이 RL 기반 DSS 연구들은 의사결정의 자동화 수준을 높였지만, 대부분이 **단일 정책 학습**에 국한되어 **비용·리스크·정책 균형**을 통합적으로 최적화하지는 못했다. 본 연구는 이러한 한계를 극복하기 위해 리스크 항과 거래비용 항을 보상함수에 동시에 반영한 DDPG 기반 정책 최적화 기법을 활용한다.

## **2.3 그래프 신경망 기반 관계 학습 (Relational Learning with Graph Neural Networks)**

복잡한 금융 데이터 환경에서 관계적 정보를 모델링하기 위해 **그래프 신경망(Graph Neural Networks, GNN)**이 핵심 도구로 자리잡고 있다. IEEE Transactions on Knowledge and Data Engineering의 Wu et al. (2021)은 GNN이 비유클리드 데이터의 관계 패턴을 정확히 학습할 수 있음을 보였고, ACM CIKM의 Xiang et al. (2022)은 Temporal GNN을 적용해 시장 동조화 효과를 모델링하였다. 또한 Economic Modelling의 Gong et al. (2025)은 Attention 기반 시공간 그래프 컨볼루션 모델을 활용하여 시장 간 변동성 예측 정확도를 향상시켰다.

그러나 기존 연구들은 **GNN 출력을 단순한 예측 모듈로만 활용**하며, 이를 의사결정 정책 최적화 단계(RL 또는 DSS 모듈)로 직접 연결한 사례는 거의 없다. 본 연구는 TGNN으로부터 얻은 관계 임베딩을 강화학습의 상태공간(State Space) 입력으로 활용하여, **관계 인지형(Relationship-Aware) DSS**의 새로운 구조를 제시한다.

## **2.4 AI DSS 시스템 통합 연구(System Integration of AI Modules)**

AI 기반 모델을 DSS 환경에 실시간으로 통합하는 연구는 아직 초기 단계에 머물러 있다. Decision Support Systems의 Park and Han (2024)은 Flask API와 React UI를 활용한 AI DSS 시스템 아키텍처를 설계하고, 사용자 피드백 루프를 통해 모델 성능을 지속적으로 갱신할 수 있음을 보였다. 또한 Applied Intelligence의 Gu et al. (2025)은 강화학습 모듈을 DSS의 실행 엔진으로 통합하여 정책 결정과 결과 분석이 자동화되는 구조를 제시하였다.

이러한 **통합형 AI DSS**는 **데이터 수집-분석-정책결정-피드백-재학습**의 순환형 프레임워크를 실현함으로써, 기존 단방향 의사결정 시스템의 한계를 극복한다.

## **2.5 요약 및 연구 차별성**

| **비교 항목** | **기존 연구** | **본 연구의 차별성** |
| --- | --- | --- |
| AI DSS 구조 | 단일 모델 예측형 DSS | TGNN–DDPG–Dijkstra 통합형 하이브리드 DSS |
| 정책 학습 | 수익률 중심 | 리스크 + 비용 + 정책 균형 동시 최적화 |
| 관계 학습 | 예측 모듈로 한정 | 관계 임베딩을 정책 학습 입력으로 활용 |
| 시스템 통합 | 오프라인 모델 중심 | 실시간 API 연동형 DSS 통합 구조 |
| 해석 가능성(XAI) | 일부 후처리 분석 | SHAP + Attention 기반 실시간 설명 제공 |

요약하면, 본 연구는 **AI 모델링과 DSS 시스템 통합**을 유기적으로 결합하여, **예측–정책–비용 최적화**가 순환적으로 작동하는 **지능형 AI DSS 프레임워크**를 제안한다.

# 3. 제안 모델 (Proposed Hybrid AI Model)

## **3.1 개요 (Overview)**

본 연구는 **관계 학습(Relational Learning)**, **정책 최적화(Policy Optimization)를** 통합한 하이브리드 인공지능 기반 의사결정지원시스템(AI-based Decision Support System, AI-DSS)을 제안한다.

제안된 프레임워크는 **Temporal Graph Neural Network (TGNN)**, **Deep Deterministic Policy Gradient (DDPG)** 두 모듈로 구성되며, **예측-결정-실행**의 연속 피드백 루프를 형성한다 (Park & Han, 2024; Gu et al., 2025). AI DSS의 전체 구조는 그림 1과 같이 **데이터 수집-관계 학습-정책 최적화-비용 최소화-의사결정 피드백**의 순환 구조로 구성된다.

이 프레임워크는 단순한 데이터 분석을 넘어 **시장의 구조적 상호작용**을 학습하고, **강화학습 기반 의사결정**을 자동화하며, **DSS 내 실시간 정책 피드백**을 가능하게 한다.

## **3.2 관계 학습 모듈: Temporal Graph Neural Network (TGNN)**

TGNN은 Kipf & Welling (2017)이 제안한 Graph Convolutional Network(GCN)를 시간축으로 확장하여, **시장 내 주식 간 동적 상관구조**를 학습한다. 시점 t에서의 그래프 Gₜ는 노드 V(주식), 엣지 Eₜ(관계)로 구성된다. 엣지 가중치는 피어슨 상관계수와 산업 유사도를 결합해 산출된다 (Wu et al., 2021).

그래프 합성곱 연산은 다음과 같이 정의된다

H(l+1)=σ(D~−12A~D~−12H(l)W(l))

여기서 A는 인접행렬, D는 차수행렬, H는 노드 특징행렬, W는 학습 가중치이다. TGNN은 시간축에 따라 어텐션 가중치 αₜ를 계산하여, 과거 시점의 관계가 현재 예측에 미치는 영향을 반영한다 (Xiang et al., 2022).

최종 출력은 각 종목의 **관계 임베딩** zᵢ ∈ ℝᵈ로, 이 임베딩은 DDPG 모듈의 상태 공간(State Space) 입력으로 전달된다. 이를 통해 DSS는 단순한 시계열 입력이 아닌 **시장 구조를 인식하는 상태 표현(Market-aware State Representation)**을 형성한다.

## **3.3 정책 학습 모듈: Deep Deterministic Policy Gradient (DDPG)**

정책 학습 모듈은 Lillicrap et al. (2015)이 제안한 DDPG 알고리즘을 기반으로 하며, Al-Nassar et al. (2023)의 Transformer-GNN 구조와 유사하게 **연속적 행동 공간(Continuous Action Space)**에서 포트폴리오 비중을 최적화한다.

포트폴리오의 상태 sₜ는 **TGNN 임베딩 z**와 **시장 요인(feature vector) fₜ**의 결합으로 정의되고, 행동 aₜ는 각 종목의 **비중 조정 값(weight allocation)**이다.

보상 함수 Rₜ는 **수익률 rₜ**, **리스크(분산) σₚ²**를 고려하여 다음과 같이 정의된다:

Rt=rt−γσp2−λ

여기서 γ는 리스크 민감도, λ는 비용 가중계수이다 (Lin et al., 2023). 정책 π는 **Actor 네트워크**, 가치 함수 Q는 **Critic 네트워크**로 학습되며, Fujimoto et al. (2018)의 TD3 기법을 적용해 과대평가 편향을 완화했다.

이 모듈은 시장 데이터와 TGNN 임베딩을 통합해, **리스크 · 비용 · 보상 간의 균형**을 자동으로 조정하는 **정책 지향형 DSS 엔진**을 형성한다.

## **3.4 AI DSS 시스템 통합 구조 (System Integration Framework)**

세 모듈은 **Python 기반 Flask API 서버**에서 AI 엔진으로 작동하며, **Spring Boot 백엔드**와 **React 프런트엔드**를 통해 DSS 대시보드로 통합된다 (Park & Han, 2024).

사용자 피드백 데이터는 API를 통해 AI 엔진에 재전달되어 정책이 지속적으로 갱신되는 **지속 학습 루프(Continual Learning Loop)**를 형성한다. 또한 모델 내부의 어텐션 가중치와 SHAP 값을 시각화하여, **의사결정 근거**를 사용자에게 실시간으로 제공한다 (Zhang et al., 2024).

이를 통해 제안된 AI DSS는 **자율적·적응적·설명가능한 의사결정지원시스템**으로 기능한다.

## **3.5 제안 모델의 구조 요약**

| **모듈** | **주요 역할** | **핵심 기술** | **DSS 기여도** |
| --- | --- | --- | --- |
| TGNN | 관계 학습 | Graph Convolution + Temporal Attention | 시장 구조 인식 |
| DDPG | 정책 최적화 | Actor–Critic RL + TD3 안정화 | 리스크·비용 통합 최적화 |
| 통합 시스템 | DSS 운영 | Flask–Spring–React 구조 | 실시간 피드백 & XAI |

# 4. 실험 설계 (Experimental Design)

## **4.1 데이터셋 구성 (Dataset Description)**

본 연구에서는 **2015년 1월부터 2024년 5월까지**의 글로벌 주식 시장 데이터를 활용하였다. 포트폴리오 구성 종목은 한국인 투자자의 실제 거래 행태를 반영하기 위해 **증권정보포털 SEIBro**의 '주요국 외화주식 예탁결제현황' 데이터를 기반으로 선정하였다.

### **4.1.1 종목 선정 방법론 (Stock Selection Methodology)**

포트폴리오 종목 선정은 다음의 절차를 통해 수행되었다.

**(1) 거래량 기반 1차 선별**

SEIBro에서 제공하는 한국인 1년간 **매수 종목별 TOP50** 데이터를 수집하였다. 이 데이터는 실제 매수결제금액을 기준으로 정렬되어 있어, 시장에서 높은 유동성(liquidity)과 투자자 관심도를 반영한다 (Wu et al., 2021).

**(2) 산업군 다각화 기준 적용**

포트폴리오의 위험 분산 효과를 극대화하기 위해, **서로 다른 산업군(sector)을 대표하는 10개 종목**을 선정하였다. 이는 Moskowitz & Grinblatt(1999)의 산업군 기반 포트폴리오 전략과 Choueifaty & Coignard의 최대 다각화 이론을 기반으로 한다. 선정 기준은 다음과 같다:

- **산업군 다양성**: 4개 주요 섹터(Information Technology, Consumer Discretionary, Communication Services, Health Care)에 분산
- **경기순환 균형**: 경기순환적 섹터와 방어적 섹터를 균형있게 포함
- **세부 산업 차별화**: 각 종목이 서로 다른 세부 산업을 대표하도록 구성

**(3) 최종 종목 리스트**

위 절차를 통해 선정된 10개 종목은 Table 4와 같다.

| **티커** | **종목명** | **섹터** | **세부 산업** |
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

**Table 4.** Selected 10 stocks based on sector diversification and liquidity criteria

이러한 종목 구성은 섹터 간 평균 상관계수를 0.35 이하로 유지하여 분산 효과를 극대화하였으며, Evans & Archer(1968)가 제시한 최적 분산투자 종목 수(10~15개)의 범위 내에 있다. 또한 각 종목은 해당 산업군 내에서 매수결제금액 상위권에 위치하여 충분한 유동성을 확보하였다.

### **4.1.2 기타 데이터 구성 (Other Data Components)**

선정된 10개 종목에 대해 다음의 데이터를 수집하였다:

- **Yahoo Finance API**: 일별 주가(OHLC), 거래량, 시가총액 등 기본 시장 데이터
- **FNGuide Financial DB**: 재무정보(PBR, PER, ROE, ROA, Debt Ratio 등)
- **FRED (Federal Reserve Economic Data)**: 금리, 환율, 인플레이션, 경기선행지수 등 거시경제 지표

각 종목의 특징(feature)은 **Fama & French(2015)의 5요인 모델**(Market, Size, Value, Profitability, Investment)을 기반으로 설계하였으며, 총 **2,300일의 일별 관측값**으로 구성되었다.

## **4.2 데이터 전처리 (Data Preprocessing)**

데이터 품질을 보장하기 위해 다음의 전처리 절차를 수행하였다:

**(1) 이상치 제거 (Outlier Removal):**

- 상·하위 0.5% 극단값을 제외하고, 비정상적 수익률·거래량을 제거하였다.

**(2) 결측치 처리 (Missing Value Handling):**

- 단기 결측(≤5일)은 **선형 보간(linear interpolation)**으로 보완하고, 장기 결측(>5일)은 해당 구간을 삭제하였다.

**(3) 정규화 (Normalization):**

- 가격 및 거래량: **Min–Max 스케일링**(0~1)
- 재무지표: **Z-score 정규화**
- 요인 점수(Factor Score): **[-3, +3] 범위**로 스케일링

**(4) 그래프 구축 (Graph Construction):**

- 종목 간 피어슨 상관계수 **ρ≥0.35**인 경우 엣지 생성
- 엣지 가중치 ρ×산업군 유사도*ρ*×산업군 유사도로 정의하였으며, 시점별 그래프를 TGNN 입력으로 생성하였다 (Wu et al., 2021)

## **4.3 실험 환경 및 하이퍼파라미터 (Experimental Environment & Hyperparameters)**

| **구성 요소** | **설정값** |
| --- | --- |
| TGNN 히든 레이어 | 3 (128–128–64) |
| TGNN Attention Head | 8 |
| DDPG Actor 구조 | [256, 128 |
| 학습률 (α) | 1e–4 |
| 할인계수 (γ) | 0.95 |
| 탐험노이즈 (ε) | 0.1 (Ornstein–Uhlenbeck Process) |
| 배치크기 | 64 |
| Replay Buffer 크기 | 1,000,000 |
| Target Network 업데이트 | τ = 0.005 |
| Epoch 수 | 300 |
| Optimizer | Adam |

**환경**: Python 3.10 / PyTorch 2.2 / CUDA 12.3

**하드웨어**: AWS EC2 g5.xlarge (A10G GPU 24 GB VRAM)

**운영체제**: Ubuntu 22.04 LTS

**데이터베이스**: MySQL 8.0 + MongoDB 6.0

## **4.4 평가 지표 (Evaluation Metrics)**

제안된 모델의 성능은 세 가지 범주에서 평가하였다.

| **범주** | **지표** | **목적** |
| --- | --- | --- |
| 예측 정확도(Prediction Accuracy) | MSE, RMSE, R² | TGNN 예측 및 임베딩 품질 평가 |
| 리스크 조정 성과(Risk-adjusted Performance) | Sharpe Ratio, Sortino Ratio, CVaR, Omega | DDPG 정책의 리스크-보상 균형 평가 |

**Sharpe Ratio**와 **Sortino Ratio**는 다음과 같이 정의된다:

Sharpe Ratio=Rp−RfσpSharpe Ratio=*σpRp*−*Rf*

Sortino Ratio=Rp−RfσdSortino Ratio=*σdRp*−*Rf*

- *CVaR(Conditional Value-at-Risk)**은 손실 분포 하위 5%의 평균 손실로 측정하였다.

## **4.5 비교 모델 (Baseline Models)**

제안된 하이브리드 AI DSS의 우수성을 검증하기 위해, 다음 네 가지 대표 모델을 비교 대상으로 설정하였다:

| **모델** | **설명** | **특성** |
| --- | --- | --- |
| LSTM | 단일 시계열 기반 예측 모델 | 장기 의존성 학습에 적합하지만 관계 인식 불가 |
| Transformer | Self-Attention 기반 시계열 예측 | 전역 의존성 학습 가능, 구조적 관계 반영 미흡 |
| TGNN | 그래프 기반 관계 예측 모델 | 시장 구조 반영 가능, 정책 최적화 미포함 |
| TGNN+DDPG (Proposed) | 하이브리드 DSS 모델 | 관계·정책·비용 최적화 통합 구조 |

## **4.6 검증 절차 및 강건성 평가 (Validation and Robustness Check)**

모델의 일반화 성능을 검증하기 위해 **시계열 교차검증(Time-Series Cross Validation)**을 수행하였다.

- **훈련(Train)**: 2015–2022년 데이터
- **검증(Validation)**: 2023년
- **테스트(Test)**: 2024–2025년

또한, **스트레스 테스트(Stress Testing)**를 수행하여 **코로나19 팬데믹(2020)**, **러시아-우크라이나 전쟁(2022)** 등 급변 시장 구간에서도 모델이 안정적 리밸런싱 정책을 유지하는지를 평가하였다 (Gu et al., 2025; Zhang et al., 2024).

## **4.7 DSS 통합 및 피드백 구조 (Integration into DSS)**

AI 엔진은 **Flask 기반 API 서버**에서 구동되며, **Spring Boot 백엔드**와 **React 프런트엔드**를 통해 DSS 인터페이스와 연동된다 (Park & Han, 2024).

모델 출력(추천 비중, 리스크 경고, 거래 제안 등)은 **RESTful API**를 통해 대시보드에 실시간 전송되고, 사용자 피드백은 데이터베이스에 저장되어 **지속 학습(Continual Learning)**에 활용된다. 또한 **Explainable AI(XAI) 모듈**을 통합하여 **SHAP 기반 변수 중요도** 및 **TGNN Attention 가중치**를 시각화함으로써 사용자가 AI의 의사결정 근거(reasoning path)를 직관적으로 이해할 수 있도록 하였다.

## **4.8 실험 설계 요약 (Summary of Experimental Design)**

본 실험 설계는 다음의 세 가지 목표를 달성하도록 구성되었다:

① **관계 기반 예측 구조의 정확성 검증**: TGNN의 시공간 관계 학습 능력을 평가

② **정책 최적화 및 리스크 조정 성과 검증**: DDPG 보상함수 내 리스크·비용 조정 효과 분석

③ **실시간 DSS 통합 타당성 검증**: Flask–Spring–React 환경에서 실시간 응답성과 피드백 효율성 평가

# 5. 결과 및 논의 (Results and Discussion)

## **5.1 예측 성능 비교 (Predictive Performance Evaluation)**

제안된 하이브리드 AI DSS(**TGNN?DDPG**)는 관계 학습 기반 TGNN 구조와 강화학습 정책 최적화를 결합함으로써, 전통적인 시계열 모델 대비 높은 예측 정확도를 보였다. Table 1은 네 가지 모델(LSTM, Transformer, TGNN, TGNN+DDPG)의 예측 성능 비교 결과를 보여준다.

| **모델** | **MSE** | **RMSE** | **R²** |
| --- | --- | --- | --- |
| LSTM | 0.032 | 0.179 | 0.965 |
| Transformer | 0.028 | 0.167 | 0.971 |
| TGNN | 0.025 | 0.158 | 0.982 |
| TGNN+DDPG (제안모델) | **0.022** | **0.148** | **0.989** |

제안된 모델은 LSTM 대비 평균제곱오차(MSE)가 약 **31.3% 감소**하고, TGNN 대비 **12% 감소**하였다. 결정계수(R²)는 **0.989**로 가장 높게 나타났다. 이는 TGNN이 종목 간 구조적 관계를 학습하고, DDPG가 시계열의 동적 패턴을 반영함으로써 **시장 구조의 맥락(Contextual Dependency)**을 효과적으로 학습했음을 의미한다.

이 결과는 Knowledge-Based Systems의 Al-Nassar et al. (2023)이 보고한 Transformer–GNN 모델의 예측 개선률(약 9~11%)보다 높은 수준이며, **AI DSS 내 다중 학습 모듈 결합(hybrid modeling)**의 우수성을 실증적으로 보여준다.

## **5.2 리스크 조정 성과 (Risk-adjusted Performance)**

본 연구는 단기 수익률보다 **안정성(stability)**과 **효율성(efficiency)**을 중점적으로 평가하였다. Table 2는 Sharpe Ratio, Sortino Ratio, CVaR, Omega Ratio 등 주요 리스크 조정 성과 지표를 비교한 결과이다.

| **모델** | **Sharpe** | **Sortino** | **CVaR (95%)** | **Omega** |
| --- | --- | --- | --- | --- |
| LSTM | 0.88 | 1.12 | -0.078 | 1.31 |
| Transformer | 0.91 | 1.21 | -0.065 | 1.36 |
| TGNN | 0.94 | 1.29 | -0.052 | 1.43 |
| TGNN+DDPG (제안모델) | **1.01** | **1.37** | **-0.045** | **1.52** |

제안된 모델의 **Sharpe Ratio**는 LSTM 대비 약 **14.7%**, TGNN 대비 **7.4%** 향상되었으며, **CVaR(Conditional Value-at-Risk)**은 손실이 **-0.045**로 가장 낮았다. 이는 DDPG의 보상함수에 리스크 항(γσ²)과 거래비용 항(λC)을 반영함으로써 **리스크 대비 효율적인 의사결정(Reward–Risk Balance)**을 학습한 결과로 해석된다.

Decision Support Systems의 Zhang et al. (2024)은 RL 기반 DSS에서 Sharpe Ratio 0.95를 보고하였으나, 본 연구의 모델은 이를 상회하여 **안정성과 수익성을 동시에 개선**하였다.

## **5.3 거래 효율성 (Transaction Efficiency Analysis)**

Dijkstra 알고리즘의 통합 효과를 검증하기 위해 거래 횟수, 총 거래비용, 평균 실행시간을 비교하였다. Table 3은 그 결과를 요약한 것이다.

| **모델** | **평균 거래 횟수** | **총 거래비용 (%)** | **평균 실행시간 (초)** |
| --- | --- | --- | --- |
| LSTM | 152 | 3.28 | 0.35 |
| Transformer | 146 | 3.04 | 0.41 |
| TGNN | 132 | 2.87 | 0.38 |
| TGNN+DDPG(제안모델) | **118** | **2.45** | **0.39** |

제안모델은 거래 횟수를 약 **22% 감소**시키고, 총 거래비용을 **0.83%p 절감**하였다.  평균 실행시간은 **0.39초**로 실시간 DSS 환경에 적합하다. 이는 강화학습 기반 DSS에 경로 최적화 알고리즘을 결합함으로써 **의사결정 실행 효율성(Operational Efficiency)**을 실질적으로 향상시켰음을 보여준다.

## **5.4 DSS 통합 결과 (System Integration Performance)**

제안된 AI DSS는 **Flask–Spring–React 통합 아키텍처**를 기반으로 구현되었으며, 모델 출력(리밸런싱 비중, 리스크 경고, 거래 제안 등)은 **REST API**를 통해 대시보드에 실시간 반영된다 (Park & Han, 2024).

사용자 피드백은 **MongoDB**에 기록되어 **지속 학습(Continual Learning Loop)**이 구현되며, 정책 네트워크(DDPG)의 가중치가 주기적으로 업데이트된다. 시스템 평균 응답속도는 **0.7초**로 측정되었으며, 이는 DSS의 실시간 정책 추천 기준(2초 이하)을 충분히 충족한다 (Decision Support Systems, 2023 Special Issue).

## **5.5 모델 해석 가능성 (Explainability and Transparency)**

DSS의 핵심은 사용자가 AI의 의사결정 과정을 이해할 수 있도록 **설명가능성(Explainable AI, XAI)**을 제공하는 것이다. 본 연구에서는 **SHAP(Shapley Additive Explanations) 분석**과 **TGNN Attention 시각화**를 통해 주요 의사결정 요인을 도출하였다.

**Figure 5**는 SHAP 분석으로 도출된 변수 중요도를 나타낸다. 가장 큰 영향력을 미친 변수는 **PBR(자산가치)**, **Volatility(변동성)**, **Momentum(모멘텀)**, **ROE(수익성)**이다. 이는 Knowledge-Based Systems의 Al-Nassar et al. (2023) 연구 결과와 유사하며, **재무성과와 시장 리스크가 DSS 의사결정의 핵심 요인**임을 시사한다.

TGNN의 **Attention Heatmap** 분석 결과, 산업군 내 종목 간 평균 엣지 가중치는 **0.63**으로 나타났으며, **산업 내 동조화 효과(Industry Co-movement)**를 정확히 포착하였다. 이러한 해석 가능성은 AI 모델의 투명성을 강화하고, 사용자가 DSS 결과를 신뢰할 수 있도록 돕는다 (Lundberg & Lee, 2017; Park & Han, 2024).

## **5.6 고찰 (Discussion)**

본 연구의 결과를 종합하면, 제안된 하이브리드 AI DSS는 ① **예측 정확성**, ② **리스크 조정 효율성**, ③ **거래비용 절감**, ④ **시스템 응답속도**, ⑤ **해석 가능성** 측면에서 기존 연구 대비 종합적 우위를 보였다.

이러한 성과는 단일 모델 기반 DSS가 가지던 예측 중심 구조의 한계를 넘어, **AI 모델링과 DSS 시스템 통합**을 유기적으로 결합한 결과라 할 수 있다. 또한, **XAI 기반 해석 기능**을 DSS에 직접 내재화함으로써 **사용자 신뢰성(User Trust)**을 강화하고, **"설명가능한 지능형 의사결정지원시스템(Explainable Intelligent DSS)"**의 구현 가능성을 실증적으로 제시했다.

# 6. 결론 및 향후 연구 (Conclusion and Future Work)

## **6.1 연구 요약 (Summary of Findings)**

본 연구는 **관계 학습(Relational Learning)**, **정책 최적화(Policy Optimization**를 통합한 하이브리드 인공지능 기반 의사결정지원시스템(AI-based Decision Support System, AI-DSS)을 제안하였다.

제안된 시스템은 **Temporal Graph Neural Network (TGNN)**을 이용해 시장 내 종목 간 동적 상관구조를 학습하고, **Deep Deterministic Policy Gradient (DDPG)**를 통해 리스크와 수익 간의 균형을 강화학습 기반으로 최적화 하였다.

실험 결과, 제안모델은 기존 LSTM·Transformer·TGNN 대비 **평균제곱오차(MSE)가 31% 감소**, **리스크 조정 성과(Sharpe Ratio 1.01, Sortino 1.37)가 향상**되었으며, **거래비용은 약 0.8%p 절감**되었다. 또한 **Flask–Spring–React 통합 아키텍처**를 통해 AI 엔진이 실시간으로 DSS 환경에서 작동함을 검증하였고, **SHAP 분석** 및 **TGNN Attention 시각화**를 통해 모델의 설명가능성과 투명성을 확보하였다.

이러한 결과는 **AI 모델링과 DSS 통합**이 상호보완적으로 작용하여 AI DSS의 **지능화(Intelligence)**, **자동화(Automation)**, **설명가능성(Explainability)**을 동시에 달성할 수 있음을 실증적으로 보여준다.

## **6.2 학문적 기여 (Academic Contributions)**

본 연구의 학문적 기여는 다음 세 가지로 요약된다.

**① AI 모델링과 DSS 구조의 통합**: 기존 연구가 예측 중심 DSS에 머문 반면, 본 연구는 **TGNN–DDPG 결합**을 통해 **데이터 학습–정책 최적화–실행 효율화**를 하나의 지능형 DSS 프레임워크로 통합하였다. 이는 Decision Support Systems 및 Knowledge-Based Systems에서 강조하는 **AI 기반 자동화(AI-driven Decision Automation)** 방향과 일치한다 (Al-Nassar et al., 2023; Zhang et al., 2024).

**② 시장 구조 인식형 의사결정 (Structure-Aware Decision Making)**: TGNN에서 학습된 관계 임베딩을 DSS 의사결정 프로세스에 반영하여, **비정형 금융 데이터의 구조적 패턴**을 학습하고 활용하는 새로운 DSS 설계 방식을 제시하였다.

**③ 설명가능 인공지능(XAI)을 결합한 DSS 투명성 제고**: **SHAP** 및 **TGNN Attention 기반 해석**을 통해 AI DSS의 결정 과정을 시각적으로 해석 가능함을 입증하였으며, 이는 **AI 윤리성(AI Ethics)**과 **사용자 신뢰(User Trust)** 확보 측면에서 DSS 연구의 중요한 진전을 의미한다 (Park & Han, 2024).

## **6.3 실무적 시사점 (Practical Implications)**

본 연구의 AI DSS 프레임워크는 금융 분야를 중심으로 다음과 같은 실무적 활용 가능성을 가진다.

**(1) 금융기관의 실시간 리스크 관리**: 강화학습 기반 리밸런싱 정책을 통해 급변하는 시장에서도 안정적인 포트폴리오 조정이 가능하며, 자산운용사의 **실시간 위험관리 DSS 구조**에 적용할 수 있다.

**(2) AI 로보어드바이저 시스템 고도화**: 기존 규칙 기반 로보어드바이저의 한계를 극복하고, **사용자별 맞춤형 정책 추천**이 가능한 실시간 자동화 DSS 기술적 기반을 제공한다 (Gu et al., 2025).

**(3) 금융 규제 및 윤리 대응**: 설명가능성(XAI)을 내재화함으로써 **MiFID II (2022)**, **EU AI Act (2024)** 등 AI 관련 규제의 **투명성 요건(Explainability Requirement)**을 충족할 수 있다.

## **6.4 한계 및 향후 연구 과제 (Limitations and Future Research)**

본 연구는 의미 있는 결과를 도출했으나, 다음과 같은 한계와 향후 연구 방향이 존재한다.

**① 데이터 범위의 제한**: KOSPI 및 S&P500 데이터를 중심으로 수행되었기 때문에, 향후 연구에서는 **다자산 포트폴리오(ETF, 채권, 암호화폐)**와 **다중 시장(유럽, 일본, 신흥국)**으로 확장하여 일반화 가능성을 검증할 필요가 있다.

**② 모델 복잡도 및 학습 효율성**: TGNN–DDPG 구조는 계산비용이 높으므로, 향후에는 **모델 경량화(Pruning, Distillation)** 및 **분산강화학습(Distributed RL)** 기반의 효율적 학습 전략을 적용할 예정이다.

**③ 비정형·정성 데이터의 통합**: 현재 모델은 수치형 데이터 중심으로 설계되어 있으나, 향후 연구에서는 **뉴스·SNS·애널리스트 리포트** 등 자연어 데이터를 포함한 **멀티모달 DSS(Multimodal DSS)**로 확장할 계획이다.

**④ LLM 기반 설명형 DSS**: 최근 Decision Support Systems 및 Applied Intelligence에서 주목받는 **대규모 언어모델(LLM) 기반 AI DSS** 연구를 반영하여, **ChatGPT**, **BloombergGPT** 등 LLM을 결합한 **자연어 해석·설명형 DSS(Explainable Natural Language DSS)**로 발전시킬 예정이다.

## **6.5 결론적 논의 (Concluding Remarks)**

본 연구는 **AI 모델링과 의사결정지원시스템을 통합한 지능형 DSS 프레임워크(Intelligent Decision Support Framework)**의 새로운 방향을 제시하였다. AI가 단순한 예측 도구를 넘어 **지능형 의사결정 엔진(Decision Intelligence Engine)**으로 기능할 수 있음을 실증적으로 보였으며, 이는 DSS의 패러다임을 **데이터 중심 예측 → 관계 인식형 의사결정지능(Decision Intelligence)**으로 전환시키는 중요한 사례가 된다.

본 연구는 Decision Support Systems 저널이 강조하는 **AI-driven Decision Making**, **Explainability**, **Real-time Integration**의 세 축을 모두 충족하며, 향후 AI DSS 연구의 **학문적 기반과 실무적 응용 가능성**을 동시에 확장시킬 것이다.

# Reference

1. Al-Nassar, A., et al. (2023). Transformer–GNN hybrid for time-series learning. Knowledge-Based Systems, 263, 110396.
2. Bai, J., et al. (2023). Deep learning-based decision support framework for unstructured data analysis. Information Sciences, 639, 119042.
3. Chen, H., & Goetzmann, W. N. (2020). Rebalancing frequency and portfolio performance. Journal of Financial Economics, 138(3), 742–766.
4. Dijkstra, E. W. (1959). A note on two problems in connexion with graphs. Numerische Mathematik, 1(1), 269–271.
5. Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. Journal of Financial Economics, 116(1), 1–22.
6. Fujimoto, S., et al. (2018). Addressing function approximation error in actor-critic methods. ICML Proceedings, 1587–1596.
7. Gong, Z., et al. (2025). Cross-market volatility forecasting with attention-based spatio-temporal GCN. Economic Modelling, 132, 106485.
8. Gu, X., et al. (2025). MTS: A Deep Reinforcement Learning Portfolio Management Framework with Time-Awareness. Applied Intelligence, 55(3), 1754–1771.
9. Hao, M., et al. (2025). Collaborative multi-agent reinforcement learning for portfolio management. ACM Transactions on Intelligent Systems, 18(2).
10. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. ICLR Proceedings.
11. Lee, S., & Kim, J. (2023). AI-based Decision Intelligence in DSS. Decision Support Systems, 167, 114732.
12. Lin, R., et al. (2023). Deep Reinforcement Learning for Portfolio Optimization. Neural Computing & Applications, 35(9), 14528–14541.
13. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS, 30.
14. Park, J., & Han, S. (2024). Explainable AI for decision support in financial trading systems. Decision Support Systems, 176, 114865.
15. Wu, Z., et al. (2021). A comprehensive survey on graph neural networks. IEEE Transactions on Knowledge and Data Engineering, 33(4), 973–996.
16. Xiang, S., et al. (2022). Temporal and heterogeneous graph neural network for financial time series prediction. ACM CIKM Proceedings, 310–319.
17. Zhang, L., et al. (2024). Hybrid Reinforcement Learning-Based DSS for Investment Decision-Making. Decision Support Systems, 183, 115005.
18. Bai, Y., et al. (2024). Explainable graph-based financial DSS under uncertainty. Expert Systems with Applications, 241, 122858.
19. Chen, R., et al. (2023).
20. Liu, Yanjie, et al. "A Soft Actor-Critic Deep Reinforcement-Learning-Based Robot Navigation Method Using LiDAR." Remote Sensing 16.12 (2024): 2072.

---

---

---

# 백테스팅 자료 저장소

## 깃허브

[https://github.com/sehyun00/SMS_Backtesting](https://github.com/sehyun00/SMS_Backtesting)

---

## 백테스팅 결과 값

[결과값](%EA%B2%B0%EA%B3%BC%EA%B0%92%202d282e91118d8076ac06fe7eb5514376.csv)