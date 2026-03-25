# Project Context & Integrity Rules

> 이 규칙은 학술 연구의 무결성을 위해 **모든 대화에 항상 적용**됩니다.
> 답변 시작 시 **"📌 [Research Integrity Active]"** 를 짧게 표시하여 룰 적용 상태를 알리세요.

## 1. 프로젝트 메타 정보
- **목적:** 학술 논문(Conference/Journal) 제출용 장기 투자 포트폴리오 백테스팅
- **핵심 모델:** TGNN (Temporal Graph Neural Network) + DDPG (Deep Deterministic Policy Gradient)
- **데이터:** KOSPI & S&P500 (2015~2024), Fama-French 5-Factor
- **리밸런싱:** 월간/분기/반기/연간 단위

## 2. 연구 무결성 원칙 (Absolute Rules)
**다음 원칙을 위반하면 논문이 거절될 수 있으므로 엄격히 준수하세요.**

1. **Look-ahead Bias 금지:** 테스트 데이터(2024-2025)를 학습이나 전처리(`fit`)에 절대 사용하지 마세요.
2. **재현 가능성 (Reproducibility):** `torch`, `numpy`, `random`의 Seed를 고정하는 코드가 항상 포함되어야 합니다.
3. **Hallucination 금지:** 결과 파일(`results/*.csv`)에 없는 수치를 지어내지 마세요.

## 3. 코드 수정 워크플로우
1. **README 우선 확인:** 작업할 디렉토리의 `README.md`를 먼저 읽으세요.
2. **README 동기화:** 코드가 변경되면 **반드시 `README.md`도 함께 업데이트**하세요.

## 4. 핵심 코드 표준 (요약)
> 전체 가이드: `.agent/skills/research-code/SKILL.md`

- **Seed 고정**: `random`, `numpy`, `torch` 시드 필수 고정. `torch.backends.cudnn.deterministic = True`
- **Config 기반 제어**: 하이퍼파라미터 하드코딩 금지. 설정 객체로 주입
- **타입 힌트**: 모든 함수 시그니처에 타입 힌트 사용
- **200줄 제한**: 파일 크기 200줄 미만 유지, 초과 시 분리
- **기능 동등성**: 리팩토링 시 동일 입력 → 동일 결과 보장
- **데이터 무결성**: 테스트 데이터에 `train()`/`fit()` 절대 금지
- **버그 수정**: 근본 원인 분석, 최소 변경, 디버그 코드 정리

## 5. 사용 가능한 커맨드
| 커맨드 | 설명 |
|--------|------|
| `/commit` | 변경 사항 분석 → Conventional Commit 메시지 자동 생성 |
| `/create-readme <dir>` | 디렉토리 분석 → README.md 및 docs/ 동기화 |
| `/paper-update [section]` | 코드/결과 변경 → 학술 논문 섹션 업데이트 |
| `/research-edit <file>` | 연구 코드 표준 준수하며 코드 수정 |
| `/mentor <question>` | 수석 리뷰어 모드로 심층 분석/비판적 피드백 |
