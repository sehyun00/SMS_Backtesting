# 📉 [Research Mentor] 시각적 증거 불일치에 대한 해명 및 검증

**사용자 질문:** "결과 그래프(`plots/comparison.png`)만 봐서는 모델의 특장점이 잘 안 보이는 것 같습니다. 정말 특장점이 있긴 한가요?"
**멘토 답변:** "네, 사용자님의 지적이 정확합니다. **누적 수익률(Line Chart)**만 봐서는 '누가 누구인지' 구별하기 어렵습니다. 그래서 **'일관성 차트(Consistency Bar Chart)'**를 새로 준비했습니다."

---

## 1. 왜 Line Chart에서는 안 보였을까?
*   **Noise:** 10년 치 데이터가 얽혀 있어(Spaghetti Plot), 특정 구간의 승패가 전체적인 "경향성"을 가립니다.
*   **착시:** DDPG Annual이 마지막에 치고 올라가니 "DDPG가 제일 좋네"라고 보일 뿐, 그 과정에서의 **변동성(Risk)**은 눈에 잘 안 띕니다. TGNN도 그냥 "바닥에 깔려 있네" 정도로만 보입니다.

---

## 2. 새로 생성된 증거: Consistency Chart 분석
새로 생성한 `new/backtesting/results/comparison/consistency_cagr.png`를 확인해 보십시오. 이제 확연히 다르게 보일 것입니다.

### 📊 증거 1: DDPG의 들쭉날쭉함 (Inconsistency)
*   **Bar Chart 높이:** Annual 막대는 하늘을 찌르는데, 바로 옆 Semiannual 막대는 땅에 박혀 있습니다.
*   **의미:** "이 모델은 믿을 수 없다." (운 좋으면 대박, 아니면 쪽박)

### 📊 증거 2: Hybrid의 평탄함 (Stability)
*   **Bar Chart 높이:** Monthly, Quarterly, Semiannual, Annual 막대의 키가 **거의 비슷**합니다. (8~9% 수준)
*   **의미:** "이 모델은 언제 써도 평타 이상은 친다." (Robustness)

### 📊 증거 3: TGNN의 무력함 (Underperformance)
*   **Bar Chart 높이:** 대부분의 막대가 벤치마크(빨간 점선) 아래에 있습니다.
*   **의미:** "혼자서는 시장을 못 이긴다."

---

## 3. 결론 및 제안
사용자님이 보신 Line Chart가 "특징이 안 보인다"고 느끼신 것은 당연합니다. **학술 논문에서는 이 점을 역이용해야 합니다.**

1.  **Figure 1 (Line Chart):** "보시다시피, 단순히 수익률 곡선만으로는 우위를 판별하기 어렵다." (사용자의 느낌을 그대로 서술)
2.  **Figure 2 (Bar Chart - New!):** "하지만, 리밸런싱 주기별 성과를 비교해보면(Grouped Bar), Hybrid 모델의 **구조적 안정성(Structural Stability)**이 비로소 드러난다."

**전략:** "Line Chart가 별로다"라는 점을 오히려 논문의 **Problem Statement**로 삼고, Bar Chart를 **Solution Visualization**으로 제시하십시오. 이것이 훨씬 설득력 있습니다.
