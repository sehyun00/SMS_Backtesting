# 📚 Research & Analysis Reports (연구 분석 보고서)

이 디렉토리는 Hybrid 모델 성능 개선 및 실험 결과 분석을 위한 심층 리포트를 모아둔 곳입니다.
각 리포트는 학술 논문(Methodology, Discussion) 작성 시 핵심 근거로 활용됩니다.

## 📂 Report List

### 1. [Hybrid 모델 심층 분석 (Frequency Sensitivity)](./06_Hybrid_Analysis_Report.md)
*   **주제**: Hybrid 모델이 왜 월간(Monthly)에서는 최고 성능이지만, 장기(Annual)에서는 저조한가?
*   **핵심 발견**:
    *   **Alpha Clamping**: 성능이 나쁜 DDPG에 강제로 20%를 배분해야 하는 구조적 제약.
    *   **Signal Decay**: TGNN의 단기 예측력이 장기 보유 시 소멸되는 현상.

### 2. [DDPG 실패 원인 및 향후 연구 (Failure Analysis)](./07_DDPG_Failure_and_Future_Work.md)
*   **주제**: DDPG는 왜 단독 모델로서 실패했는가? (Factor 추가 전 기준)
*   **핵심 발견**:
    *   **Regime Shift**: 학습(상승장)과 테스트(하락장)의 괴리.
    *   **Context Blindness**: 금리 등 거시 변수 부재로 인한 오버피팅.
*   **제언**: Regime-Aware RL, Transformer 도입 등 Future Work 제시.

### 3. [Factor 추가 영향 분석 (Trade-off Analysis)](./08_Factor_Impact_Analysis.md)
*   **주제**: Fama-French Factor를 추가하니 왜 DDPG는 살고 TGNN은 죽었는가?
*   **핵심 발견**:
    *   **DDPG 부활**: 매크로 정보(금리)를 보게 되면서 무지성 매수 중단 (-1.5% → +4.8%).
    *   **TGNN 몰락**: 전역 변수(Global Factor)가 개별 종목 신호(Local Signal)를 희석시킴 (Signal Dilution).

### 4. [TGNN 존재 가치 방어 논리 (Defense Logic)](./09_Defense_Logic_Why_TGNN.md)
*   **주제**: "성능 나쁜 TGNN을 왜 Hybrid에서 안 뺍니까?"라는 질문에 대한 대처법.
*   **핵심 논리**:
    *   **역할 분담**: DDPG는 시장 타이밍(Beta), TGNN은 종목 선정(Alpha).
    *   **생존력(Robustness)**: 국면 변화에 따라 주도 모델이 바뀌므로 둘 다 필요함.
    *   **연구 가치**: Feature Conflict 현상 발견 자체가 학술적 기여.

---
> **Note**: 이 리포트들은 `AI Research Mentor` 페르소나에 기반하여, 단순 결과 나열이 아닌 "논문 심사 통과"를 목적으로 작성되었습니다.
