# 3. 제안 모델 (Proposed Hybrid AI Model)

## **3.1 개요 (Overview)**

본 연구는 **관계 학습(Relational Learning)**, **정책 최적화(Policy Optimization)를** 통합한 하이브리드 인공지능 기반 의사결정지원시스템(AI-based Decision Support System, AI-DSS)을 제안한다.

제안된 프레임워크는 **Temporal Graph Neural Network (TGNN)**, **Deep Deterministic Policy Gradient (DDPG)** 두 모듈로 구성되며, **예측-결정-실행**의 연속 피드백 루프를 형성한다 (Park & Han, 2024; Gu et al., 2025). AI DSS의 전체 구조는 그림 1과 같이 **데이터 수집-관계 학습-정책 최적화-비용 최소화-의사결정 피드백**의 순환 구조로 구성된다.

이 프레임워크는 단순한 데이터 분석을 넘어 **시장의 구조적 상호작용**을 학습하고, **강화학습 기반 의사결정**을 자동화하며, **DSS 내 실시간 정책 피드백**을 가능하게 한다.

## **3.2 관계 학습 모듈: Temporal Graph Neural Network (TGNN)**

TGNN은 Kipf & Welling (2017)이 제안한 Graph Convolutional Network(GCN)를 시간축으로 확장하여, **시장 내 주식 간 동적 상관구조**를 학습한다. 시점 t에서의 그래프 Gₜ는 노드 V(주식), 엣지 Eₜ(관계)로 구성된다. 엣지 가중치는 피어슨 상관계수와 산업 유사도를 결합해 산출된다 (Wu et al., 2021).

그래프 합성곱 연산은 다음과 같이 정의된다

H(l+1)=σ(D~−12A~D~−12H(l)W(l))

여기서 A는 인접행렬, D는 차수행렬, H는 노드 특징행렬, W는 학습 가중치이다. TGNN은 시간축에 따라 어텐션 가중치 αₜ를 계산하여, 과거 시점의 관계가 현재 예측에 미치는 영향을 반영한다 (Xiang et al., 2022).

최종 출력은 각 종목의 **관계 임베딩** zᵢ ∈ ℝᵈ로, 이 임베딩은 DDPG 모듈의 상태 공간(State Space) 입력으로 전달된다. 이를 통해 DSS는 단순한 시계열 입력이 아닌 **시장 구조를 인식하는 상태 표현(Market-aware State Representation)**을 형성한다.

## **3.3 정책 학습 모듈: Deep Deterministic Policy Gradient (DDPG)**

정책 학습 모듈은 Lillicrap et al. (2015)이 제안한 DDPG 알고리즘을 기반으로 하며, Al-Nassar et al. (2023)의 Transformer-GNN 구조와 유사하게 **연속적 행동 공간(Continuous Action Space)**에서 포트폴리오 비중을 최적화한다.

포트폴리오의 상태 sₜ는 **TGNN 임베딩 z**와 **시장 요인(feature vector) fₜ**의 결합으로 정의되고, 행동 aₜ는 각 종목의 **비중 조정 값(weight allocation)**이다.

보상 함수 Rₜ는 **수익률 rₜ**, **리스크(분산) σₚ²**를 고려하여 다음과 같이 정의된다:

Rt=rt−γσp2−λ

여기서 γ는 리스크 민감도, λ는 비용 가중계수이다 (Lin et al., 2023). 정책 π는 **Actor 네트워크**, 가치 함수 Q는 **Critic 네트워크**로 학습되며, Fujimoto et al. (2018)의 TD3 기법을 적용해 과대평가 편향을 완화했다.

이 모듈은 시장 데이터와 TGNN 임베딩을 통합해, **리스크 · 비용 · 보상 간의 균형**을 자동으로 조정하는 **정책 지향형 DSS 엔진**을 형성한다.

## **3.4 AI DSS 시스템 통합 구조 (System Integration Framework)**

세 모듈은 **Python 기반 Flask API 서버**에서 AI 엔진으로 작동하며, **Spring Boot 백엔드**와 **React 프런트엔드**를 통해 DSS 대시보드로 통합된다 (Park & Han, 2024).

사용자 피드백 데이터는 API를 통해 AI 엔진에 재전달되어 정책이 지속적으로 갱신되는 **지속 학습 루프(Continual Learning Loop)**를 형성한다. 또한 모델 내부의 어텐션 가중치와 SHAP 값을 시각화하여, **의사결정 근거**를 사용자에게 실시간으로 제공한다 (Zhang et al., 2024).

이를 통해 제안된 AI DSS는 **자율적·적응적·설명가능한 의사결정지원시스템**으로 기능한다.

## **3.5 제안 모델의 구조 요약**

| **모듈** | **주요 역할** | **핵심 기술** | **DSS 기여도** |
| --- | --- | --- | --- |
| TGNN | 관계 학습 | Graph Convolution + Temporal Attention | 시장 구조 인식 |
| DDPG | 정책 최적화 | Actor–Critic RL + TD3 안정화 | 리스크·비용 통합 최적화 |
| 통합 시스템 | DSS 운영 | Flask–Spring–React 구조 | 실시간 피드백 & XAI |
