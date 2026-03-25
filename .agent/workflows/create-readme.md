---
description: 디렉토리 분석 기반 README.md 및 연구 문서(docs/) 동기화 워크플로우
---

# 문서 자동화 워크플로우 (README & Docs)

이 워크플로우는 소스 코드를 분석하여 `README.md`를 생성하고, `docs/` 폴더 내의 연구 노트가 실제 코드 구현과 일치하는지 확인하여 업데이트합니다.

## 전제 조건 (Prerequisites)
1.  **필수 스킬 로드**: `view_file .agent/skills/research-code/SKILL.md`

## 단계 (Steps)

1.  **대상 디렉토리 확인**
    -   `list_dir`로 사용자가 지정한 디렉토리와 `docs/` 서브 디렉토리를 확인합니다.

2.  **문서 스캔 (Document Scanning)**
    -   기존 `README.md`와 `docs/` 내의 주요 마크다운 파일(`research_notes.md` 등)을 읽습니다.
    -   현재 문서에 기술된 알고리즘/전략 로직을 파악합니다.

3.  **코드-문서 정합성 분석 (Consistency Check)**
    -   핵심 코드(`strategy.py`, `model.py`, `config.yaml` 등)를 분석합니다.
    -   **전략 비교**: 코드의 실제 구현(예: Full Universe Softmax)과 문서의 설명(예: Top-k 30%)이 다르면, **코드를 기준으로 문서를 수정**할 계획을 세웁니다.
    -   **파라미터 확인**: `config.yaml`의 설정값(Batch size, Window size)이 문서와 일치하는지 확인합니다.

4.  **문서 작성 (Writing)**
    -   **README.md**: `research_code` 표준에 맞춰 최신 상태로 작성합니다. `docs/` 링크를 포함시킵니다.
    -   **Research Notes**: 실험 전략이나 하이퍼파라미터가 변경되었다면, 해당 섹션을 업데이트하여 **연구 노트의 '구현 상세' 파트를 최신화**합니다.

5.  **파일 생성 (Execution)**
    -   `write_to_file`로 `README.md`를 저장합니다.
    -   만약 연구 노트 내용이 변경되었다면 `docs/research_notes.md`도 업데이트(Overwrite)합니다.

## 예시 트리거
- `/create-readme new/backtesting` (README 및 docs 업데이트)
