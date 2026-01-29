---
description: 디렉토리 분석 기반 상세 README.md 자동 생성 (research_code 스킬 연동)
---

# README 생성 워크플로우

이 워크플로우는 지정된 디렉토리의 소스 코드를 분석하여 `research_code` 스킬의 문서화 표준(재현성, 입력/출력 명세 등)을 준수하는 상세한 `README.md`를 생성합니다.

## 전제 조건 (Prerequisites)
1.  **필수 스킬 로드**: 이 워크플로우를 실행하기 전에 반드시 `research_code` 스킬을 숙지해야 합니다.
    -   `view_file .agent/skills/research_code.md`

## 단계 (Steps)

1.  **대상 디렉토리 확인**
    -   사용자가 지정한 디렉토리가 존재하는지 `list_dir`로 확인합니다.
    -   `__init__.py`가 있는지 확인하여 패키지 구조인지 파악합니다.

2.  **하위 모듈 스캔 (Sub-module Scanning)**
    -   현재 디렉토리의 하위 폴더에 `README.md`가 있는지 확인합니다.
    -   발견된 하위 README를 `view_file`로 읽어(상위 50줄), 해당 모듈의 **이름**과 **핵심 역할**을 추출합니다.
    -   이 정보는 부모 README의 "모듈 구조" 섹션에 요약하여 포함시킵니다.

3.  **코드 분석 (Deep Analysis)**
    -   `list_dir`로 파일 목록을 확보합니다.
    -   핵심 파일(주로 `agent.py`, `model.py`, `trainer.py` 등)을 `view_file`로 읽습니다.
    -   다음 정보를 추출합니다:
        -   **클래스/함수 아키텍처**: 주요 클래스의 역할과 상속 관계.
        -   **입력/출력 텐서 모양(Shape)**: `[B, N, T, F]` 등 차원 정보.
        -   **설정(Config) 의존성**: `config.yaml`에서 어떤 키값을 사용하는지.
        -   **핵심 알고리즘**: GAT, DDPG, Attention 등 사용된 핵심 기법.

4.  **문서 작성 (Writing)**
    -   `research_code` 스킬의 "문서화 원칙"을 적용하여 `README.md` 내용을 생성합니다.
    -   **필수 포함 항목**:
        -   **타이틀 & 한 줄 요약**: 무엇을 하는 모듈인가?
        -   **하위 모듈 요약**: (하위 README 존재 시) 각 모듈의 역할 요약 테이블.
        -   **아키텍처 다이어그램/설명**: 내부 데이터 흐름.
        -   **입력/출력 명세**: 정확한 Dimension과 데이터 타입.
        -   **사용법 (Usage)**: `config.yaml` 설정 예시.
        -   **파일 구조**: 주요 파일별 역할 설명.

4.  **파일 생성**
    -   `write_to_file` 도구를 사용하여 대상 디렉토리에 `README.md`를 저장합니다.
    -   기존 파일이 있다면 내용을 보강(Overwrite)하거나 중요 내용을 병합합니다.

## 예시 트리거
- `/create-readme new/backtesting/src/models/tgnn`
