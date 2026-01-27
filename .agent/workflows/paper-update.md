---
description: 코드 변경이나 새로운 결과에 따라 학술 논문의 섹션을 업데이트합니다.
---

# 논문 업데이트 워크플로우

이 워크플로우는 코드베이스의 변경 사항이나 새로운 실험 결과가 있을 때 `new/academic_papers/sections/`에 있는 학술 논문 섹션을 업데이트하는 과정을 안내합니다.

## 1. 컨텍스트 식별 (사용자 입력)
-   **트리거**: 사용자가 "새로운 결과로 논문 업데이트해줘" 또는 "방법론에 코드 변경 사항 반영해줘"라고 말할 때.
-   **조치**: 사용자에게 다음을 질문합니다:
    1.  **무엇이 변경되었나요?** (예: "하이브리드 보상 함수 업데이트", "새로운 테스트 결과")
    2.  **어떤 섹션**을 주로 업데이트해야 하나요? (예: 방법론, 결과)

## 2. 진실의 원천 로드 (소스 확인)
-   **조치**: 팩트를 확인하기 위해 관련 파일을 읽습니다.
    -   **방법론(Methodology)**인 경우: `src/models/*.py`, `config.yaml` 읽기.
    -   **결과(Results)**인 경우: `results/*.csv`, `results/*.log` 읽기.
    -   **실험 설정(Experimental Setup)**인 경우: `config.yaml`, `src/training/dataset.py` 읽기.

## 3. 타겟 섹션 로드
-   **조치**: `new/academic_papers/sections/`에 있는 해당 마크다운 파일을 읽습니다.
    -   `01_Abstract_Intro.md` (초록 및 서론)
    -   `02_Related_Works.md` (관련 연구)
    -   `03_Methodology.md` (방법론)
    -   `04_Experimental_Design.md` (실험 설계)
    -   `05_Results_Discussion.md` (결과 및 논의)
    -   `06_Conclusion_Ref.md` (결론 및 참고문헌)

## 4. 업데이트 적용 (`paper-updater` 스킬 사용)
-   **지침**:
    -   `.agent/skills/paper-updater/SKILL.md`를 `view_file`로 읽어 가이드라인을 상기합니다.
    -   영향을 받는 문단이나 표를 재작성합니다.
    -   **중요**: 주변 문맥을 유지하세요. 방정식 하나만 변경되었다면 전체 파일을 다시 쓰지 마세요.

## 5. 일관성 확인
-   **조치**: 검증.
    -   **결과(Results)**를 업데이트했다면, **초록(Abstract)**이나 **결론(Conclusion)**에서 이전 수치를 언급하는지 확인하고 필요하면 수정합니다.
    -   **방법론(Methodology)**(예: 모듈 이름 변경)을 업데이트했다면, **서론(Intro)**에서 이를 언급하는지 확인합니다.

## 6. 최종 검토
-   **조치**: 변경 사항(diff)이나 새로운 내용을 사용자에게 보여주고 승인을 받습니다.
