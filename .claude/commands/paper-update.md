코드 변경이나 새로운 결과에 따라 학술 논문의 섹션을 업데이트합니다.

**대상 섹션 (선택)**: $ARGUMENTS

## 전제 조건
먼저 `.agent/skills/paper-updater/SKILL.md` 파일을 읽어 논문 업데이트 가이드라인을 확인하세요.
필요 시 예제 파일도 참조하세요:
- `.agent/skills/paper-updater/examples/figure_placement.md`
- `.agent/skills/paper-updater/examples/qa_loop_example.md`

## 1. 🔄 질의응답 루프 (Interactive Q&A Loop)

> **핵심 원칙**: 정보가 불충분하면 **절대로 추측하지 말고** 사용자에게 질문하세요.

### 질의 트리거 조건
다음 상황에서는 작업을 중단하고 사용자에게 질문해야 합니다:

| 상황 | 질문 예시 |
|------|-----------|
| **수치 미확인** | "CSV에서 Hybrid Annual의 Sharpe Ratio를 찾을 수 없습니다. 정확한 값을 알려주시겠습니까?" |
| **코드 불일치** | "`actor.py`의 수식과 논문 3.2절 수식이 다릅니다. 어떤 버전이 최신입니까?" |
| **참고문헌 불완전** | "Park & Han (2024) 논문의 정확한 제목과 저널명을 알려주시겠습니까?" |
| **맥락 부족** | "이 실험의 목적이 '수익률 극대화'인지 '리스크 최소화'인지 명확히 해주시겠습니까?" |
| **이미지 누락** | "`images/` 디렉토리에 SHAP 분석 이미지가 없습니다. 생성해야 합니까?" |

### 질의 형식
```markdown
❓ **확인 필요**

[구체적인 질문 내용]

- **현재 상태**: [파악한 정보]
- **필요한 정보**: [누락된 정보]
```

## 2. 컨텍스트 식별
사용자에게 질문합니다:
1. **무엇이 변경되었나요?** (예: "하이브리드 보상 함수 업데이트", "새로운 테스트 결과")
2. **어떤 섹션**을 주로 업데이트해야 하나요? (예: 방법론, 결과)

## 3. 소스 확인 및 경로 매핑

### 소스 → 타겟 매핑

| 논문 섹션 | 주요 소스 | 검증 포인트 |
|-----------|-----------|-------------|
| `01_Abstract_Intro.md` | `05_Results_Discussion.md` | 초록은 **가장 마지막**에 수정 |
| `02_Related_Works.md` | 외부 논문, `docs/research/` | 인용 형식 일관성 |
| `03_Methodology.md` | `src/models/*.py`, `config.yaml` | 수식과 코드 일치 여부 |
| `04_Experimental_Design.md` | `config.yaml`, `README.md` | 하이퍼파라미터 정확성 |
| `05_Results_Discussion.md` | `results/**/*.csv` | **CSV에 없는 수치 생성 금지** |
| `06_Conclusion_Ref.md` | 전체 섹션, 참고문헌 | 결론과 결과 일치 |

### 데이터 소스 위치
```
new/backtesting/
├── results/
│   ├── comparison/
│   │   └── all_strategies_comparison.csv  ← 핵심 성과 데이터
│   ├── hybrid/
│   ├── tgnn/
│   └── ddpg/
├── docs/
│   ├── model_comparison_analysis.md       ← 분석 보고서
│   ├── research/                          ← 심층 연구 자료
│   └── ...
└── README.md

new/academic_papers/
├── sections/     ← 논문 섹션 (01~06)
└── images/       ← Figure/Table 이미지
```

## 4. 타겟 섹션 로드
`new/academic_papers/sections/`의 해당 마크다운 파일을 읽습니다:
- `01_Abstract_Intro.md`, `02_Related_Works.md`, `03_Methodology.md`
- `04_Experimental_Design.md`, `05_Results_Discussion.md`, `06_Conclusion_Ref.md`

## 5. 업데이트 적용
- `paper-updater` 스킬 가이드라인에 따라 영향받는 문단/표를 재작성합니다.
- **중요**: 주변 문맥을 유지하세요. 수식 하나만 변경되었다면 전체 파일을 다시 쓰지 마세요.

### Figure/Table 배치
- **이미지 경로**: `new/academic_papers/images/`
- **참조 이미지**: `new/backtesting/results/comparison/*.png`
- **배치 마커 형식**:
```markdown
<!-- [Figure 1] CAGR 비교 차트 삽입 위치 -->
<!-- 권장: images/cagr_comparison_all.png -->
```

| 섹션 | Figure | Table |
|------|--------|-------|
| **03_Methodology** | 모델 아키텍처 다이어그램 | 하이퍼파라미터 설정 |
| **04_Experimental** | 데이터 분포/시계열 | 데이터셋 통계, 학습 설정 |
| **05_Results** | CAGR 비교, Sharpe 비교, Risk-Return 산점도, 누적 수익률 | 전략별 성과 비교 (Table 1) |

## 6. ✅ 신빙성 검증 체크리스트

**결과 섹션 (`05_Results`) 작성 시:**
- [ ] 모든 수치가 `all_strategies_comparison.csv`와 일치하는가?
- [ ] Benchmark 값이 정확한가?
- [ ] 비교 대상 모델이 누락되지 않았는가?

**방법론 섹션 (`03_Methodology`) 작성 시:**
- [ ] 수식이 코드와 일치하는가?
- [ ] 하이퍼파라미터가 `config.yaml`과 일치하는가?

### Hallucination 방지 규칙

> ⚠️ **절대 금지**: 결과 파일에 없는 수치를 생성하지 마세요.

1. **수치 인용 시**: 반드시 소스 파일과 라인 번호 확인
2. **불확실한 경우**: "~로 추정된다" 대신 사용자에게 질문
3. **참고문헌**: 실제 존재하는 논문만 인용

## 7. 일관성 확인
- **결과** 업데이트 시 → **초록/결론**에서 이전 수치를 언급하는지 확인
- **방법론** 업데이트 시 → **서론**에서 이를 언급하는지 확인

## 8. 최종 검토
변경 사항을 사용자에게 보여주고 승인을 받습니다.

### 빠른 참조 - 핵심 벤치마크 수치 (2015-2024)
| 지표 | Benchmark (Buy & Hold) |
|------|------------------------|
| CAGR | 5.89% |
| Sharpe | 0.22 |
| MDD | 21.51% |
| Total Return | 25.20% |
