---
name: Project Context
description: 코드 수정 시 반드시 참조해야 하는 프로젝트 핵심 정보
---

# 프로젝트 컨텍스트

> ⚙️ **AI 지시사항**: 코드 수정 작업 시 이 스킬을 읽고 있다면 "📌 project-context 스킬 적용 중"이라고 알려주세요.

## 1. 프로젝트 유형

- **학술 논문용 연구** (컨퍼런스/저널 제출 목적)
- **장기 투자** 백테스팅 프로젝트 (단기 트레이딩 아님)
- 리밸런싱 주기: 월간 / 분기 / 반기 / 연간
- 팩터 모델: Fama-French 5-Factor (Mkt_RF, SMB, HML, RMW, CMA)

## 2. 연구 무결성 원칙

> ⚠️ **위반 시 논문 리젝 사유가 될 수 있음**

1. **Look-ahead Bias 금지**: 테스트 데이터로 모델 파인튜닝 절대 금지
2. **재현 가능성**: 랜덤 시드 고정, 모든 하이퍼파라미터 명시
3. **공정한 벤치마크**: 1/N Buy&Hold와 동일 조건에서 비교
4. **통계적 유의성**: Sharpe Ratio, t-test 등 정량적 지표 사용

## 3. 코드 수정 시 절차

1. **수정 대상 디렉토리의 README.md를 먼저 읽을 것**
2. README의 **Frontmatter** 확인 (팩터 정보, 알려진 이슈 등)
3. Frontmatter에 `current_bug` 또는 `known_issues`가 있으면 주의
4. 코드 수정 후 README 내용과 불일치하면 **README도 함께 업데이트**

## 4. README Frontmatter 규칙

```yaml
---
purpose: 모듈 목적 설명
factors: Fama-French 5-Factor (Mkt_RF, SMB, HML, RMW, CMA)
known_issues: 알려진 이슈 목록
---
```

> ⚠️ **주의**: Config 파일의 주석보다 **README Frontmatter**와 **코드 Docstring**을 우선 신뢰하세요.

## 5. 수정 후 체크리스트

- [ ] README Frontmatter와 코드가 일치하는가?
- [ ] 알려진 이슈(`known_issues`)를 해결했다면 README에서 제거했는가?
- [ ] 새로운 이슈가 발생했다면 README에 추가했는가?
- [ ] **연구 무결성 원칙을 위반하지 않았는가?**
