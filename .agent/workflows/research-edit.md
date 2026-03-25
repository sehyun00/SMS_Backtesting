---
description: research_code 스킬을 준수하며 코드를 수정/리팩토링하는 워크플로우
---
# Research Code Edit Workflow

이 워크플로우는 코드를 수정할 때 `research_code` 스킬의 학술 연구 표준(재현성, 강건성, 모듈화)을 강제로 적용하기 위해 설계되었습니다. 단순한 코드 수정이 아니라, 연구의 **신뢰성(Reliability)**을 유지하는 것이 목표입니다.

## 전제 조건 (Prerequisites)

1. **필수 스킬 로드**: 작업 시작 전 반드시 `research_code` 스킬을 확인해야 합니다.
   - `view_file .agent/skills/research-code/SKILL.md`

## 단계 (Steps)

1. **컨텍스트 분석 (Context Analysis)**

   - 수정할 대상 파일을 `view_file`로 읽습니다.
   - **Config 의존성 확인**: 하드코딩된 값(Learning Rate, 차원 수 등)이 있다면 `config.yaml` 참조로 변경할 것을 계획합니다.
   - **Type Hint 점검**: 함수의 인자와 반환값에 타입 힌트가 누락되어 있는지 확인합니다.
2. **수정 계획 수립 (Planning)**

   - **기능 동등성(Functional Parity)**: 리팩토링의 경우, 입력에 대한 출력이 기존과 수학적으로 동일한지 검증하는 로직을 생각합니다.
   - **재현성(Reproducibility)**: `random`, `torch` 등의 시드 고정 로직이 보존되는지 확인합니다.
   - **200줄 규칙**: 파일이 너무 커지면 분리를 제안합니다.
3. **코드 수정 (Execution)**

   - `replace_file_content` 또는 `multi_replace_file_content`를 사용합니다.
   - **Docstring 추가**: 변경된 함수나 클래스에 명확한 Docstring을 작성합니다.
   - **주석 작성**: 중요한 로직 변경에는 근거(Rationale)를 주석으로 남깁니다.
4. **검증 (Verification)**

   - 편집 후 Lint 에러가 없는지 확인합니다.
   - 필요 시 관련 테스트 코드를 실행하거나, `check_models.py`와 같은 검증 스크립트를 실행합니다.

## 예시 트리거

- `/research-edit src/training/trainer.py`
- "이 파일 리팩토링 해줘 (research-edit 워크플로우로)"
