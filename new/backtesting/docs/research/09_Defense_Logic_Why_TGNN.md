# 예상 질문 방어: "왜 성능도 안 좋은 TGNN을 굳이 씁니까?"
> **작성일**: 2026-02-03
> **작성자**: AI Research Mentor

리뷰어가 **"TGNN이 오히려 방해만 되는데, 굳이 Hybrid로 엮을 이유가 있는가?"**라고 공격할 때, 다음과 같은 3단계 논리로 방어하십시오.

---

## 1. 논리 1: "역할이 다르다 (Local Alpha vs Global Beta)"

*   **TGNN의 존재 이유**: TGNN은 **"개별 종목의 고유한 패턴(Idiosyncratic Pattern)"**과 **"종목 간의 관계(Sector Correlation)"**를 포착하는 데 특화되어 있습니다. (이전 실험에서 7%대 수익률로 입증됨)
*   **DDPG의 존재 이유**: DDPG(Macro 포함)는 **"시장 전체의 흐름(Systematic Risk)"**을 읽고 현금 비중을 조절하는 데 특화되어 있습니다.
*   **방어 멘트**:
    > *"단순 수익률만 보면 DDPG가 우세해 보이지만, DDPG는 '시장 전체가 오를까/내릴까'만 봅니다. 반면 TGNN은 '어떤 종목이 더 오를까(Selection)'를 담당합니다. 이번 실험에서 TGNN의 부진은 Macro 변수의 단순 결합(Concatenation)으로 인한 일시적 신호 희석일 뿐, **종목 선별(Stock Selection)**이라는 TGNN의 본질적 가치가 사라진 것은 아닙니다."*

## 2. 논리 2: "구조적 안정성 (Robustness)"

*   **상호 보완성**: 만약 미래에 **"Macro 변동성은 적은데 종목별 차별화가 심한 장세(Stock Picker's Market)"**가 온다면, Macro만 보는 DDPG는 힘을 못 씁니다. 이때는 TGNN이 다시 빛을 발합니다.
*   **방어 멘트**:
    > *"금융 시장은 끊임없이 국면(Regime)이 변합니다. Macro가 지배하는 시장(2022년)에서는 DDPG가, 종목 장세(202X년)에서는 TGNN이 주도권을 잡습니다. Hybrid 모델의 목적은 특정 시점의 최대 수익률(Winner-takes-all)이 아니라, **어떤 국면이 와도 살아남는 생존력(Robustness)**에 있습니다. 실제로 Factor를 끄면 TGNN이 캐리하고, Factor를 켜면 DDPG가 캐리하는 모습 자체가 상호 보완성을 증명합니다."*

## 3. 논리 3: "실험적 의의 (Scientific Contribution)"

*   **실패도 연구다**: 모든 것이 완벽하게 작동하는 결과만 보여주는 것은 공학 보고서지, 과학 논문이 아닙니다.
*   **방어 멘트**:
    > *"본 연구의 핵심 기여 중 하나는 **'Graph Model과 Macro Factor 간의 충돌(Feature Conflict)'** 현상을 발견한 것입니다. 단순히 TGNN을 제거하는 것은 쉬운 길이지만, 우리는 이 현상을 보고함으로써 향후 **'Macro-Decoupled GNN'**이나 **'Hierarchical Hybrid'**와 같은 발전된 연구 방향을 제시하고자 합니다."*

---

## 🔥 필살기 한마디
*"TGNN을 빼는 건 모델을 단순화하는 것이지만, TGNN을 남겨두고 그 한계를 분석하는 건 **통찰(Insight)**을 제공하는 것입니다. 저희는 후자를 택했습니다."*
