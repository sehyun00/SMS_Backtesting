# 질의응답 루프 시나리오 예시

## 🔄 시나리오 1: 수치 데이터 미확인

### 상황
사용자가 "Results 섹션에 최신 결과 반영해줘"라고 요청했으나,
`all_strategies_comparison.csv`에 TGNN Annual 결과가 없는 경우.

### AI 응답 예시
```markdown
❓ **확인 필요**

Results 섹션 업데이트를 위해 데이터를 확인 중입니다.
`all_strategies_comparison.csv`에서 **TGNN Annual** 결과를 찾을 수 없습니다.

- **현재 상태**: TGNN의 Monthly, Quarterly, Semiannual 결과만 존재
- **필요한 정보**: TGNN Annual의 CAGR, Sharpe, MDD 값

다음 중 하나를 선택해 주세요:
1. TGNN Annual 실험을 아직 수행하지 않았다면, 해당 행을 제외하고 진행
2. 결과 값을 직접 알려주시면 반영
3. 백테스팅을 다시 실행하여 데이터 생성
```

---

## 🔄 시나리오 2: 코드와 논문 불일치

### 상황
`03_Methodology.md`의 Reward Function 수식이 
`src/models/ddpg/agent.py`의 실제 구현과 다른 경우.

### AI 응답 예시
```markdown
❓ **확인 필요**

Methodology 섹션의 Reward Function과 실제 코드가 불일치합니다.

**논문 (03_Methodology.md, Line 45):**
```latex
R_t = w^T r_t - \lambda \cdot \text{Var}(w^T r_t)
```

**코드 (agent.py, Line 127):**
```python
reward = portfolio_return - 0.5 * volatility_penalty
```

- **차이점**: 논문은 분산(Var)을 사용하지만, 코드는 변동성(volatility)을 사용
- **질문**: 어떤 버전이 최신이며 논문에 반영해야 합니까?
```

---

## 🔄 시나리오 3: 참고문헌 정보 불완전

### 상황
논문에서 인용한 "Park & Han (2024)" 논문의 정확한 정보가 필요한 경우.

### AI 응답 예시
```markdown
❓ **확인 필요**

`06_Conclusion_Ref.md`의 참고문헌 목록을 정리 중입니다.
**Park & Han (2024)** 논문의 정확한 정보가 필요합니다.

- **현재 상태**: 저자명과 연도만 확인됨
- **필요한 정보**:
  1. 정확한 논문 제목
  2. 게재 저널/학회명
  3. 권호, 페이지 번호 (또는 DOI)

정보를 알려주시면 APA/IEEE 형식에 맞게 정리하겠습니다.
```

---

## 🔄 시나리오 4: 이미지 누락

### 상황
Results 섹션에 SHAP 분석 Figure를 넣어야 하는데,
`images/` 디렉토리에 해당 이미지가 없는 경우.

### AI 응답 예시
```markdown
❓ **확인 필요**

Results 섹션에 SHAP 분석 결과(Figure 5)를 배치해야 하는데,
`new/academic_papers/images/` 디렉토리에 **SHAP 관련 이미지가 없습니다**.

- **현재 상태**: images/ 디렉토리에 4개 이미지 존재 (cagr, sharpe, risk_return, stability)
- **필요한 이미지**: SHAP feature importance plot

다음 중 하나를 선택해 주세요:
1. SHAP 이미지를 생성하여 `images/shap_importance.png`로 저장
2. 다른 위치에 이미지가 있다면 경로 알려주기
3. SHAP 분석 섹션 생략
```

---

## 📋 질의응답 템플릿

```markdown
❓ **확인 필요**

[상황 설명]

- **현재 상태**: [파악한 정보]
- **필요한 정보**: [누락된 정보]

[선택지 또는 구체적 질문]
```
