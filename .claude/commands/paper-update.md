코드 변경이나 새로운 결과에 따라 학술 논문의 섹션을 업데이트합니다.

**대상 섹션 (선택)**: $ARGUMENTS

## 전제 조건
먼저 `.agent/skills/paper-updater/SKILL.md` 파일을 읽어 논문 업데이트 가이드라인을 확인하세요.
필요 시 예제 파일도 참조하세요:
- `.agent/skills/paper-updater/examples/figure_placement.md`
- `.agent/skills/paper-updater/examples/qa_loop_example.md`

## 단계

### 1. 컨텍스트 식별
사용자에게 질문합니다:
1. **무엇이 변경되었나요?** (예: "하이브리드 보상 함수 업데이트", "새로운 테스트 결과")
2. **어떤 섹션**을 주로 업데이트해야 하나요? (예: 방법론, 결과)

### 2. 소스 확인
- **방법론(Methodology)**: `src/models/*.py`, `config.yaml` 읽기
- **결과(Results)**: `results/*.csv`, `results/*.log` 읽기
- **실험 설정(Experimental Setup)**: `config.yaml`, `src/training/dataset.py` 읽기

### 3. 타겟 섹션 로드
`new/academic_papers/sections/`의 해당 마크다운 파일을 읽습니다:
- `01_Abstract_Intro.md`, `02_Related_Works.md`, `03_Methodology.md`
- `04_Experimental_Design.md`, `05_Results_Discussion.md`, `06_Conclusion_Ref.md`

### 4. 업데이트 적용
- `paper-updater` 스킬 가이드라인에 따라 영향받는 문단/표를 재작성합니다.
- **중요**: 주변 문맥을 유지하세요. 수식 하나만 변경되었다면 전체 파일을 다시 쓰지 마세요.

### 5. 일관성 확인
- **결과** 업데이트 시 → **초록/결론**에서 이전 수치를 언급하는지 확인
- **방법론** 업데이트 시 → **서론**에서 이를 언급하는지 확인

### 6. 최종 검토
변경 사항을 사용자에게 보여주고 승인을 받습니다.
