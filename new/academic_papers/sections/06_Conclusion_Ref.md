# 6. 결론 및 향후 연구 (Conclusion and Future Work)

## 6.1 연구 요약 (Summary of Findings)

본 연구는 관계 학습(Relational Learning)과 정책 최적화(Policy Optimization)를 통합한 하이브리드 인공지능 기반 의사결정지원시스템(AI-based Decision Support System, AI-DSS)을 제안하였다.

제안된 시스템은 Temporal Graph Neural Network (TGNN)을 이용해 시장 내 종목 간 동적 상관구조를 학습하고, Deep Deterministic Policy Gradient (DDPG)를 통해 리스크와 수익 간의 균형을 강화학습 기반으로 최적화하였다. 특히 본 연구는 AI 모델의 결과가 우연이 아님을 입증하기 위해 엄격한 재현성 프로토콜(Reproducibility Protocol)을 적용하였다.

실험 결과, 제안된 **Hybrid 모델은 장기 리밸런싱 주기(Quarterly, Semiannual, Annual)에서 Benchmark를 일관되게 상회**하였다. DDPG(Annual)가 최고 CAGR 10.41%를 기록했지만, 리밸런싱 주기에 따른 성과 편차가 크고 Semiannual에서 Benchmark를 하회하였다. Hybrid 모델은 Monthly(5.90%)에서 Benchmark(6.35%)를 하회했으나, **장기 투자 전략**에서는 안정적인 초과 수익을 보여 학술적 가치를 입증하였다. 또한 Flask–Spring–React 통합 아키텍처를 통해 AI 엔진이 실시간으로 DSS 환경에서 작동함을 검증하였다.

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

② **모델 복잡도 및 학습 효율성**: TGNN–DDPG 구조는 계산비용이 높으므로, 향후에는 모델 경량화(Pruning, Distillation) 및 분산강화학습(Distributed RL) 기반의 효율적 학습 전략을 적용할 예정이다.

③ **멀티모달 DSS로의 확장**: 현재 모델은 수치형 데이터 중심으로 설계되어 있으나, 향후 연구에서는 뉴스·SNS·애널리스트 리포트 등 자연어 데이터를 포함한 멀티모달 DSS(Multimodal DSS)로 확장할 계획이다.

## 6.5 결론적 논의 (Concluding Remarks)

본 연구는 AI 모델링과 의사결정지원시스템을 통합한 신뢰할 수 있는 지능형 DSS 프레임워크(Trustworthy Intelligent DSS Framework)의 새로운 방향을 제시하였다. AI가 단순한 예측 도구를 넘어 지능형 의사결정 엔진(Decision Intelligence Engine)으로 기능할 수 있음을 실증적으로 보였으며, 이는 DSS의 패러다임을 데이터 중심 예측 → 신뢰 가능한 의사결정지능(Trustworthy Decision Intelligence)으로 전환시키는 중요한 사례가 된다.

본 연구는 Decision Support Systems 저널이 강조하는 AI-driven Decision Making, Explainability, Reproducibility의 세 축을 모두 충족하며, 향후 AI DSS 연구의 학문적 기반과 실무적 응용 가능성을 동시에 확장시킬 것이다.

# Reference

1. Al-Nassar, A., et al. (2023). Transformer–GNN hybrid for time-series learning. Knowledge-Based Systems, 263, 110396.
2. Bai, J., et al. (2023). Deep learning-based decision support framework for unstructured data analysis. Information Sciences, 639, 119042.
3. Chen, H., & Goetzmann, W. N. (2020). Rebalancing frequency and portfolio performance. Journal of Financial Economics, 138(3), 742–766.
4. Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. Journal of Financial Economics, 116(1), 1–22.
5. Fujimoto, S., et al. (2018). Addressing function approximation error in actor-critic methods. ICML Proceedings, 1587–1596.
6. Gong, Z., et al. (2025). Cross-market volatility forecasting with attention-based spatio-temporal GCN. Economic Modelling, 132, 106485.
7. Gu, X., et al. (2025). MTS: A Deep Reinforcement Learning Portfolio Management Framework with Time-Awareness. Applied Intelligence, 55(3), 1754–1771.
8. Hao, M., et al. (2025). Collaborative multi-agent reinforcement learning for portfolio management. ACM Transactions on Intelligent Systems, 18(2).
9. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. ICLR Proceedings.
10. Lee, S., & Kim, J. (2023). AI-based Decision Intelligence in DSS. Decision Support Systems, 167, 114732.
11. Lin, R., et al. (2023). Deep Reinforcement Learning for Portfolio Optimization. Neural Computing & Applications, 35(9), 14528–14541.
12. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS, 30.
13. Park, J., & Han, S. (2024). Explainable AI for decision support in financial trading systems. Decision Support Systems, 176, 114865.
14. Wu, Z., et al. (2021). A comprehensive survey on graph neural networks. IEEE Transactions on Knowledge and Data Engineering, 33(4), 973–996.
15. Xiang, S., et al. (2022). Temporal and heterogeneous graph neural network for financial time series prediction. ACM CIKM Proceedings, 310–319.
16. Zhang, L., et al. (2024). Hybrid Reinforcement Learning-Based DSS for Investment Decision-Making. Decision Support Systems, 183, 115005.
17. Bai, Y., et al. (2024). Explainable graph-based financial DSS under uncertainty. Expert Systems with Applications, 241, 122858.
18. Liu, Yanjie, et al. "A Soft Actor-Critic Deep Reinforcement-Learning-Based Robot Navigation Method Using LiDAR." Remote Sensing 16.12 (2024): 2072.
