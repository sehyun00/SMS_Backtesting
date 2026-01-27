# SMS Backtesting Research Code (Refactored)

이 프로젝트는 기존의 `Hybrid_TGNN_DDPG` 모델을 연구용 코드로 리팩토링한 결과물입니다.
모듈화된 구조, 표준화된 인터페이스, 그리고 강화학습(RL) 인프라가 구축되어 있습니다.

## 📂 디렉토리 구조
- `src/models/`: 모델 구현 (TGNN, Hybrid, DDPG)
- `src/preprocessing/`: 데이터 전처리 및 5-Factor 계산
- `src/training/`: 학습 루프, Dataset, Environment, ReplayBuffer
- `config/`: 실험 설정 (`config.yaml`)
- `tests/`: 유닛 테스트

## 🚀 실행 방법

### 1. 환경 설정
`config/config.yaml`에서 실험 파라미터를 수정하세요.
- `project.selected_model`: `tgnn`, `ddpg`, `hybrid` 중 선택

### 2. 학습 실행
```bash
python new/main.py
```
학습 결과는 `results/{model_name}/` 폴더에 저장됩니다.
- 로그 파일: `train_{timestamp}.log`
- 모델 체크포인트: `best_model.pth`

### 3. 모델 검증 (Verify All)
모든 모델이 정상적으로 작동하는지 확인하려면:
```bash
python new/check_models.py
```

## 🛠️ 주요 기능
- **Multi-Model Support**: 설정 변경만으로 TGNN, DDPG, Hybrid 모델 전환 가능
- **Unified RL Infrastructure**: `RLTrainer`와 `PortfolioEnvironment`를 통해 일관된 강화학습 실험 가능
- **Clean Architecture**: 명확한 역할 분담 (Agent, Actor, Critic, Encoder, Head)
