# 6. 결론 및 향후 연구 (Conclusion and Future Work)

## 6.1 연구 요약 (Summary of Findings)

본 연구는 관계 학습(Relational Learning)과 정책 최적화(Policy Optimization)를 통합한 하이브리드 인공지능 기반 의사결정지원시스템(AI-based Decision Support System, AI-DSS)을 제안하였다.

제안된 시스템은 Temporal Graph Neural Network (TGNN)을 이용해 시장 내 종목 간 동적 상관구조를 학습하고, Deep Deterministic Policy Gradient (DDPG)를 통해 리스크와 수익 간의 균형을 강화학습 기반으로 최적화하였다. 두 모듈은 고정 알파(α = 0.5) 균등 앙상블로 통합되며, 특히 본 연구는 AI 모델의 결과가 우연이 아님을 입증하기 위해 5개 시드($S = \{0, 42, 123, 456, 789\}$)를 사용한 **다중 시드 재현성 프로토콜(Multi-Seed Reproducibility Protocol)**을 적용하였다.

5-seed 반복 실험 결과, 제안된 **Hybrid (Quarterly) 모델은 CAGR 9.09 ± 2.86%로 Benchmark(6.35%) 대비 +2.74%p의 안정적 초과 수익**을 달성하였다. 이는 5회 반복 실험에서 일관되게 재현되어, 결과의 구조적 우위를 통계적으로 확인하였다. 또한 DDPG 단독 대비 MDD가 약 10%p 낮아 리스크 통제 효과도 입증하였다. 아울러 Flask–Spring–React 통합 아키텍처를 통해 AI 엔진이 실시간으로 DSS 환경에서 작동함을 검증하였다.

이러한 결과는 AI 모델링과 DSS 통합이 상호보완적으로 작용하여 AI DSS의 지능화(Intelligence), 자동화(Automation), 그리고 신뢰성(Trustworthiness)을 동시에 달성할 수 있음을 실증적으로 보여준다.

## 6.2 학문적 기여 (Academic Contributions)

본 연구의 학문적 기여는 다음 세 가지로 요약된다.

① **신뢰할 수 있는 AI DSS 방법론 확립**: 기존 연구가 성능(Performance)에 집중한 반면, 본 연구는 결정론적 알고리즘(Deterministic Algorithm)을 도입하여 실험의 재현성(Reproducibility)을 보장함으로써, 금융 AI 연구가 나아가야 할 '신뢰 가능한 연구(Reproducible Research)'의 표준을 제시하였다.

② **시장 구조 인식형 의사결정 (Structure-Aware Decision Making)**: TGNN에서 학습된 관계 임베딩을 DSS 의사결정 프로세스에 반영하여, 비정형 금융 데이터의 구조적 패턴을 학습하고 활용하는 새로운 DSS 설계 방식을 제시하였다.

③ **AI 모델링과 DSS 구조의 통합**: TGNN–DDPG 결합을 통해 데이터 학습–정책 최적화–실행 효율화를 하나의 지능형 DSS 프레임워크로 통합하였다. 이는 Decision Support Systems 및 Knowledge-Based Systems에서 강조하는 AI 기반 자동화(AI-driven Decision Automation) 방향과 일치한다 (Al-Nassar et al., 2023; Zhang et al., 2024).

## 6.3 실무적 시사점 (Practical Implications)

본 연구의 AI DSS 프레임워크는 금융 분야를 중심으로 다음과 같은 실무적 활용 가능성을 가진다.

(1) **금융기관의 신뢰성 높은 리스크 관리**: 설명가능성(XAI)과 결과의 일관성이 보장되므로, 금융 규제 준수가 필수적인 자산운용사의 실시간 위험관리 DSS 시스템에 즉시 적용 가능하다.

(2) **AI 로보어드바이저 시스템 고도화**: 기존 규칙 기반 로보어드바이저의 한계를 극복하고, 사용자별 맞춤형 정책 추천이 가능한 실시간 자동화 DSS 기술적 기반을 제공한다 (Gu et al., 2025).

(3) **금융 규제 및 윤리 대응**: 설명가능성(XAI)을 내재화함으로써 MiFID II (2022), EU AI Act (2024) 등 AI 관련 규제의 투명성 요건(Explainability Requirement)을 충족할 수 있다.

## 6.4 한계 및 향후 연구 과제 (Limitations and Future Research)

본 연구는 의미 있는 결과를 도출했으나, 다음과 같은 한계와 향후 연구 방향이 존재한다.

① **데이터 범위의 제한**: S&P 500 구성 종목을 중심으로 수행되었기 때문에, 향후 연구에서는 다자산 포트폴리오(ETF, 채권, 암호화폐)와 다중 시장(KOSPI, 유럽, 일본, 신흥국)으로 확장하여 일반화 가능성을 검증할 필요가 있다. 본 연구가 S&P 500이라는 특정 유니버스에 한정된 실험임을 인정하며 소형주나 신흥 시장으로의 일반화에는 추가 검증이 필요함을 명시한다.
① **DDPG Critic 불안정성**: 본 연구에서 DDPG 단독 모델은 전체 리밸런싱 주기에서 음수 CAGR을 기록하였다. 이는 DDPG Critic 네트워크의 Q-value 과추정(Q-value Overestimation) 문제가 주된 원인으로, 강화학습 학습 과정에서 Critic이 과적합되어 불안정한 포트폴리오 정책을 생성하는 것으로 분석된다. 향후 연구에서는 TD3(Twin Delayed DDPG, Fujimoto et al., 2018)의 이중 Critic 구조를 채택하거나 Soft Actor-Critic (SAC)을 적용하여 이를 완화할 수 있다.

② **특정 리밸런싱 주기에서의 고분산**: Hybrid Semiannual(std 7.12%)과 Annual(std 6.76%) 전략은 시드 간 표준편차가 평균에 근접하거나 초과하여 강건성이 제한된다. 이는 리밸런싱 이벤트 수가 적은 장기 주기에서 개별 리밸런싱 시점의 판단 오류가 전체 성과에 미치는 영향이 크기 때문이다. 본 연구의 핵심 결과는 분산이 상대적으로 낮은 Quarterly 주기에 한정하며, 장기 주기에 대한 일반화는 향후 연구에서 추가 검증이 필요하다.

③ **데이터 범위의 제한**: S&P 500 구성 종목을 중심으로 수행되었기 때문에, 향후 연구에서는 다자산 포트폴리오(ETF, 채권, 암호화폐)와 다중 시장(KOSPI, 유럽, 일본, 신흥국)으로 확장하여 일반화 가능성을 검증할 필요가 있다.

④ **모델 복잡도 및 학습 효율성**: TGNN–DDPG 구조는 계산비용이 높으므로, 향후에는 모델 경량화(Pruning, Distillation) 및 분산강화학습(Distributed RL) 기반의 효율적 학습 전략을 적용할 예정이다.

⑤ **멀티모달 DSS로의 확장**: 현재 모델은 수치형 데이터 중심으로 설계되어 있으나, 향후 연구에서는 뉴스·SNS·애널리스트 리포트 등 자연어 데이터를 포함한 멀티모달 DSS(Multimodal DSS)로 확장할 계획이다.

## 6.5 결론적 논의 (Concluding Remarks)

본 연구는 AI 모델링과 의사결정지원시스템을 통합한 신뢰할 수 있는 지능형 DSS 프레임워크(Trustworthy Intelligent DSS Framework)의 새로운 방향을 제시하였다. AI가 단순한 예측 도구를 넘어 지능형 의사결정 엔진(Decision Intelligence Engine)으로 기능할 수 있음을 실증적으로 보였으며, 이는 DSS의 패러다임을 데이터 중심 예측 → 신뢰 가능한 의사결정지능(Trustworthy Decision Intelligence)으로 전환시키는 중요한 사례가 된다.

본 연구는 Decision Support Systems 저널이 강조하는 AI-driven Decision Making, Explainability, Reproducibility의 세 축을 모두 충족하며, 향후 AI DSS 연구의 학문적 기반과 실무적 응용 가능성을 동시에 확장시킬 것이다.

# Reference

1. Al-Nassar, A., et al. (2023). Transformer–GNN hybrid for time-series learning. Knowledge-Based Systems, 263, 110396.
2. Brown, G., Wyatt, J., Harris, R., & Yao, X. (2005). Diversity creation methods: A survey and categorisation. Information Fusion, 6(1), 5–20.
3. Bai, J., et al. (2023). Deep learning-based decision support framework for unstructured data analysis. Information Sciences, 639, 119042.
4. Chen, H., & Goetzmann, W. N. (2020). Rebalancing frequency and portfolio performance. Journal of Financial Economics, 138(3), 742–766.
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
19. Liu, Yanjie, et al. "A Soft Actor-Critic Deep Reinforcement-Learning-Based Robot Navigation Method Using LiDAR." Remote Sensing 16.12 (2024): 2072.
