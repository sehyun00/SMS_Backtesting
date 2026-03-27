# 2. 관련 연구 (Related Works)

## **2.1 AI 기반 의사결정지원시스템의 진화 (Evolution of AI-Based Decision Support Systems)**

최근 DSS 분야는 단순한 규칙 기반(rule-based) 시스템에서 인공지능(AI) 기반의 지능형 의사결정지원(Intelligent Decision Support)으로 빠르게 발전하고 있다. Decision Support Systems 저널의 Lee and Kim (2023)은 AI 기반 의사결정엔진을 도입함으로써 전통적 분석형 DSS의 한계를 보완할 수 있음을 보였다. Applied Intelligence의 Gu et al. (2025)는 강화학습(RL) 엔진을 DSS 아키텍처에 내재화하여 시장 변화에 실시간 대응할 수 있는 AI-driven 의사결정엔진을 제시하였다. 또한 Information Sciences의 Bai et al. (2023)은 딥러닝을 활용한 의사결정지원 프레임워크를 검증하며, AI 모델이 비정형 데이터에서 의미 추론과 정책 추천을 수행할 수 있음을 입증하였다.

이처럼 최근 연구들은 AI 기술을 DSS의 핵심 요소로 간주하며, 예측 정확성뿐 아니라 설명가능성(Explainability) 및 적응성(Adaptivity)을 동시에 강조하는 방향으로 진화하고 있다 (Park & Han, 2024; Zhang et al., 2024). 그러나 대부분의 연구가 단일 AI 모델 중심 예측에 머물러 있으며, 다중 AI 모델 간 상호작용 또는 실험 결과의 재현성(Reproducibility)에 대한 체계적 검증은 아직 부족한 실정이다.

## **2.2 강화학습 기반 의사결정지원 (Decision Support with Reinforcement Learning)**

강화학습(RL)은 비정형 환경에서 스스로 정책을 학습하는 능력으로 DSS 분야에 적극적으로 도입되고 있다. Neural Computing and Applications의 Lin et al. (2023)은 DDPG 모델을 적용한 DSS 프레임워크를 통해 비용-효율적인 포트폴리오 리밸런싱 정책을 학습하였으며, Knowledge-Based Systems의 Al-Nassar et al. (2023)은 Transformer 기반 정책 네트워크를 RL과 결합해 시계열 상관관계를 반영하였다.

최근 목표 지향적 연속 제어(Continuous Control)가 요구되는 비정형 환경에서는 Actor-Critic 구조의 효과성이 입증되고 있다. 일례로 Liu et al. (2024)은 로봇 네비게이션 문제에서 상태-행동-보상 구조를 정교하게 설계하여 동적 장애물 환경에 실시간으로 적응하는 Actor-Critic 모델(SAC)을 제안하였다.

본 연구는 Liu et al. (2024)가 제시한 '연속 제어 기반의 동적 환경 적응 메커니즘'의 개념적 타당성에서 착안하여, 이를 금융 의사결정 프레임워크로 차용한다. 구체적으로는 기존 로봇 네비게이션에서 활용된 MDP(Markov Decision Process) 설계 방식을 '시장 상태-포트폴리오 비중 조정-수익률/리스크 보상'의 체계로 변환하여 적용하였다. 다만 정책(Policy) 알고리즘의 선택에 있어서는, 탐색에 초점을 맞춘 확률적 정책(SAC)이 아닌, 자산 배분의 일관성과 학술적 재현성(Reproducibility)을 강력히 보장할 수 있는 결정론적 정책 알고리즘인 DDPG(Deep Deterministic Policy Gradient)를 채택한다. 본 연구는 이 DDPG 모델의 Actor-Critic 네트워크를 TGNN 임베딩과 결합함으로써, 시장 변동성에 강건하면서도 신뢰할 수 있는 관계 인식형 포트폴리오 최적화를 구현한다.

Decision Support Systems의 Zhang et al. (2024)은 PPO와 TD3 알고리즘을 비교하여 AI 정책이 리스크 관리 측면에서도 효율적으로 작동함을 보였다.

이와 같이 RL 기반 DSS 연구들은 의사결정의 자동화 수준을 높였지만, 대부분이 단일 정책 학습에 국한되어 예측 모듈과 정책 모듈을 투자 주기에 따라 동적으로 결합하는 앙상블 메커니즘은 부재하였다. 본 연구는 이러한 한계를 극복하기 위해 TGNN 기반 관계 학습과 DDPG 기반 정책 최적화를 Horizon-Aware Dynamic Alpha로 통합하는 앙상블 기법을 제안한다.

## **2.3 그래프 신경망 기반 관계 학습 (Relational Learning with Graph Neural Networks)**

복잡한 금융 데이터 환경에서 관계적 정보를 모델링하기 위해 그래프 신경망(Graph Neural Networks, GNN)이 핵심 도구로 자리잡고 있다. IEEE Transactions on Knowledge and Data Engineering의 Wu et al. (2021)은 GNN이 비유클리드 데이터의 관계 패턴을 정확히 학습할 수 있음을 보였고, ACM CIKM의 Xiang et al. (2022)은 Temporal GNN을 적용해 시장 동조화 효과를 모델링하였다. 또한 Economic Modelling의 Gong et al. (2025)은 Attention 기반 시공간 그래프 컨볼루션 모델을 활용하여 시장 간 변동성 예측 정확도를 향상시켰다.

그러나 기존 연구들은 GNN 출력을 단순한 예측 모듈로만 활용하며, 이를 의사결정 정책 최적화 단계(RL 또는 DSS 모듈)로 직접 연결한 사례는 거의 없다. 본 연구는 TGNN으로부터 얻은 관계 임베딩을 강화학습의 상태공간(State Space) 입력으로 활용하여, 관계 인지형(Relationship-Aware) DSS의 새로운 구조를 제시한다.

## **2.4 AI DSS 시스템 통합 연구(System Integration of AI Modules)**

AI 기반 모델을 DSS 환경에 실시간으로 통합하는 연구는 아직 초기 단계에 머물러 있다. Decision Support Systems의 Park and Han (2024)은 Flask API와 React UI를 활용한 AI DSS 시스템 아키텍처를 설계하고, 사용자 피드백 루프를 통해 모델 성능을 지속적으로 갱신할 수 있음을 보였다. 또한 Applied Intelligence의 Gu et al. (2025)은 강화학습 모듈을 DSS의 실행 엔진으로 통합하여 정책 결정과 결과 분석이 자동화되는 구조를 제시하였다.

이러한 통합형 AI DSS는 데이터 수집-분석-정책결정-피드백-재학습의 순환형 프레임워크를 실현함으로써, 기존 단방향 의사결정 시스템의 한계를 극복한다.

## **2.5 요약 및 연구 차별성**

| 비교 항목        | 기존 연구                   | 본 연구의 차별성                                     |
| ---------------- | --------------------------- | ---------------------------------------------------- |
| AI DSS 구조      | 단일 모델 예측형 DSS        | TGNN–DDPG Horizon-Aware Ensemble 하이브리드 DSS      |
| 정책 학습        | 수익률 중심                 | 리스크 조정 보상 + 동적 앙상블 정책 최적화           |
| 관계 학습        | 예측 모듈로 한정            | 관계 임베딩을 정책 학습 입력으로 활용                |
| 실험 신뢰성      | 성능 우위 강조 (Robustness) | 재현성 프로토콜 적용 (Reproducibility & Seed Fixing) |
| 시스템 통합      | 오프라인 모델 중심          | 실시간 API 연동형 DSS 통합 구조                      |
| 해석 가능성(XAI) | 일부 후처리 분석            | SHAP + Attention 기반 실시간 설명 제공               |

요약하면, 본 연구는 AI 모델링과 DSS 시스템 통합을 유기적으로 결합하여, 예측–정책–앙상블 최적화가 순환적으로 작동하는 지능형 AI DSS 프레임워크를 제안한다. 또한, 결정론적 알고리즘을 도입하여 학술적 재현성을 보장함으로써 신뢰할 수 있는 연구 결과를 제시한다.
