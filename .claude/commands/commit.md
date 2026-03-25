변경된 파일을 분석하여 Conventional Commit 형식의 커밋 메시지를 자동 생성합니다.

## 워크플로우 단계

### 1. 변경 사항 확인
```bash
git status --short
git diff --stat
```

### 2. 변경 내용 분석
- 변경된 파일 목록 확인
- 주요 변경 유형 파악 (기능 추가, 버그 수정, 문서, 리팩토링 등)
- 영향 범위(scope) 추론

### 3. 커밋 메시지 생성

#### Conventional Commit 형식
```
<type>(<scope>): <subject>

<body>
```

#### Type 종류
| Type | 설명 |
|------|------|
| `feat` | 새로운 기능 |
| `fix` | 버그 수정 |
| `docs` | 문서 변경 |
| `style` | 코드 포맷팅 |
| `refactor` | 리팩토링 |
| `test` | 테스트 |
| `chore` | 기타 작업 |

#### Scope 예시
| Scope | 대상 |
|-------|------|
| `auth` | 인증 관련 |
| `portfolio` | 포트폴리오 관련 |
| `ui` | UI 컴포넌트 |
| `db` | 데이터베이스 |
| `api` | API 엔드포인트 |
| `AI_PRD` | PRD 문서 |
| `workflow` | 워크플로우 |

### 4. 사용자 확인
생성된 커밋 메시지를 사용자에게 보여주고 승인을 요청하세요.
- Y: 그대로 커밋
- N: 취소
- 수정: 사용자 피드백 반영 후 재생성

### 5. 커밋 실행 (승인 후)
변경된 파일을 개별적으로 staging하고 커밋하세요. `git add .` 대신 파일을 명시적으로 추가하세요.

## 주의사항
- 커밋 전 `git diff`로 변경 내용 한번 더 확인
- 메시지가 길면 body에 상세 내용 작성
- 이슈 번호가 있으면 body에 `related: #이슈번호` 추가
