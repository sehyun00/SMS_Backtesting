# 📊 [Research Mentor] 모델 성능 비교 분석 보고서

**작성일:** 2026-02-06
**분석 대상:** TGNN, Hybrid, DDPG 모델의 백테스팅 결과 (Monthly ~ Annual Rebalancing)
**작성자:** Research Mentor (AI Agent)

---

## 1. Executive Summary (요약)

본 분석은 코스피 및 S&P500 데이터를 기반으로 한 3가지 모델(TGNN, DDPG, Hybrid)의 성능을 비교 분석하였습니다. 주요 결과는 다음과 같습니다.

*   **🏆 Best Performer:** **DDPG (Annual Rebalancing)** 모델이 **Sharpe Ratio 0.42, CAGR 10.41%**로 가장 우수한 위험 조정 수익률을 기록했습니다.
*   **🛡️ Most Robust:** **Hybrid** 모델은 분기(Quarterly), 반기(Semiannual), 연간(Annual) 모든 구간에서 벤치마크를 상회하며 가장 일관성 있는 성능을 보였습니다.
*   **⚠️ Underperformer:** **TGNN** 단일 모델은 대부분의 구간에서 벤치마크를 하회하였으며, 예측 시그널이 포트폴리오 최적화로 직결되지 못하는 한계를 보였습니다.

---

## 2. 상세 성능 비교 (Detailed Analysis)

### 2.1. 모델별 핵심 지표 (vs Benchmark)

**Benchmark (Buy & Hold)**
*   CAGR: 6.35%
*   Sharpe: 0.25
*   MDD: 22.47%

| Model | Rebalancing | CAGR (%) | Sharpe | MDD (%) | 비고 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TGNN** | Monthly | 3.24 | 0.06 | 25.35 | ❌ Underperform |
| | Quarterly | 4.36 | 0.13 | 21.17 | ❌ Underperform |
| | Semiannual | **8.38** | **0.35** | 22.07 | ✅ Outperform |
| | Annual | 2.65 | 0.03 | 25.98 | ❌ Underperform |
| **Hybrid** | Monthly | 5.90 | 0.20 | 27.06 | ⚠️ Neutral |
| | Quarterly | **8.07** | **0.31** | 25.83 | ✅ Outperform |
| | Semiannual | **8.61** | **0.33** | 27.05 | ✅ Outperform |
| | Annual | **9.28** | **0.36** | 25.15 | ✅ Outperform |
| **DDPG** | Monthly | **7.58** | **0.28** | 27.59 | ✅ Outperform |
| | Quarterly | **8.45** | **0.31** | 30.76 | ✅ Outperform |
| | Semiannual | 6.23 | 0.21 | 26.09 | ⚠️ Neutral |
| | Annual | **10.41** | **0.42** | 24.41 | 🏆 **Best** |

> **범례:** **Bold**는 벤치마크 상회 항목.

### 2.2. 시각적 패턴 해석 (Plots)
*   **TGNN:** 수익 곡선이 벤치마크 아래에 머무르는 경향이 강함. 반기 리밸런싱에서만 일시적인 초과 수익이 발생했으나 추세적이지 않음.
*   **Hybrid:** 벤치마크와 유사한 흐름을 보이면서도 하락장에서 방어하거나 상승장에서 탄력을 받는 모습을 보임. TGNN의 구조적 정보와 RL의 정책 최적화가 상호 보완적으로 작용한 것으로 추정.
*   **DDPG:** 연간(Annual) 모델의 경우, 특정 시점의 리밸런싱이 시장의 큰 흐름(Macro Trend)을 잘 포착하여 계단식 상승을 만들어낸 것으로 보임.

---

## 3. Critical Insight (심층 비판 및 멘토링)

### 💡 Insight 1: "연간(Annual) 리밸런싱의 승리"가 의미하는 바는?
DDPG와 Hybrid 모두 **Annual Rebalancing**에서 최고의 성능을 보였습니다.
*   **금융공학적 해석:** 주식 시장, 특히 지수 추종형 포트폴리오에서 **장기 추세(Momentum)**는 노이즈가 많은 단기(Monthly)보다 연간 단위에서 더 뚜렷할 수 있습니다. RL 에이전트가 잦은 매매로 인한 손실(Whipsaw)을 피하고, 굵직한 추세를 타는 정책을 학습했을 가능성이 큽니다.
*   **⚠️ 통계적 주의 (Statistical Significance):** 2015~2024년 데이터에서 Annual 리밸런싱은 고작 **9~10번의 의사결정**만을 의미합니다. 표본이 너무 적기 때문에 이 결과가 모델의 우수성인지, 아니면 운 좋게 몇 번의 하락장을 피한 것인지(Overfitting to specific years) 경계해야 합니다. "DDPG Annual이 최고다"라고 단정 짓기엔 표본 수가 부족합니다.

### 💡 Insight 2: TGNN의 실패 원인
TGNN은 관계형 데이터(Graph)를 학습하지만, 그것이 곧바로 '수익률'로 연결되지 않습니다.
*   **원인:** TGNN의 Loss Function이 단순히 다음 스텝의 주가 예측(MSE 등)에 맞춰져 있다면, 이는 포트폴리오 최적화와는 다른 목표일 수 있습니다. (예측은 맞지만, 거래 비용이나 리스크를 고려하지 못함).
*   **Hybrid의 성공:** 이를 DDPG(RL)가 보완했습니다. TGNN이 추출한 Feature를 RL이 받아서 "어떻게 행동해야 보상(Sharpe)이 최대화되는가"를 학습했기에 Hybrid 모델이 더 우수한 성과를 낸 것입니다.

### 💡 Insight 3: DDPG Monthly의 선전
DDPG 단일 모델이 Monthly에서도 벤치마크를 상회(CAGR 7.58%)한 점은 고무적입니다. 이는 순수 RL 에이전트가 단기 변동성 안에서도 유의미한 패턴을 찾았음을 시사합니다. 다만 MDD(27.59%)가 가장 높으므로, 레버리지나 리스크 관리에 대한 추가 제약이 필요합니다.

---

## 4. 제안 사항 (Action Plan)

1.  **논문 전략 수정:**
    *   "TGNN 단일 모델이 우수하다"는 주장은 기각해야 합니다.
    *   대신 **"TGNN은 시장의 내재적 관계를 추출하는 Feature Extractor로 작동하고, 이를 DDPG가 활용할 때(Hybrid) 가장 강건(Robust)한 성능을 낸다"**는 논리로 전개하십시오.
2.  **추가 검증:**
    *   Annual 모델의 과적합 의심을 해소하기 위해, **Rolling Window Backtest** (예: 2015 시작, 2016 시작... 등으로 시작 시점을 달리하여 테스트)를 수행해볼 것을 권장합니다.
3.  **리스크 관리:**
    *   DDPG의 높은 MDD를 낮추기 위해 Reward Function에 변동성 패널티(Volatility Penalty) 가중치를 높이는 실험이 필요합니다.

---
**총평:** Hybrid 모델의 일관성은 학술적으로 매우 가치 있는 결과입니다. 단순히 수익률이 높은 모델보다, 다양한 기간에서 안정적인 모델이 논문 게재 확률이 높습니다.
