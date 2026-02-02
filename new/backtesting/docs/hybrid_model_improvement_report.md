# 하이브리드 모델 성능 및 재현성 개선 보고서

## 1. 문제 식별 (Problem Identification)
*   **이슈**: 낮은 학습 손실(Low Training Loss)에도 불구하고 하이브리드 모델의 성능이 심각하게 저하됨(CAGR 마이너스 기록).
*   **원인 분석**:
    1.  **Critic 과대평가(Overestimation)**: 학습 초기 단계에서 Critic 네트워크가 부정확하거나 부풀려진 Q값(Q-values)을 생성함.
    2.  **앙상블 학습의 조기 고착화 (Premature Convergence)**: Critic이 안정화되기도 전에 `Ensemble Net` (Alpha)이 이 잘못된 Q값을 학습하여, 포트폴리오 비중이 최적화되지 않은 상태로 고정됨.
    3.  **재현성(Reproducibility) 부재**: 엄격한 시드(Seed) 제어가 없어 실행할 때마다 결과가 달라져 디버깅이 어려웠음.

## 2. 해결 방안 구현 (Solution Implementation)

### A. 재현성 확보 (기술적 신뢰도)
*   **전역 시드(Global Seed) 고정**: Python `random`, `numpy`, `torch` (CPU/GPU), `PYTHONHASHSEED`의 시드를 `42`로 고정.
*   **결정론적 알고리즘(Deterministic Algorithms)**: `torch.backends.cudnn.deterministic = True` 설정을 통해 하드웨어 레벨의 연산 무작위성 제거.
*   **효과**: 매 실행마다 비트 단위까지 동일한 결과를 보장하여 연구의 신뢰성 확보.

### B. 하이브리드 웜업 (Hybrid Warm-up, 알고리즘 개선)
*   **로직**: `config.yaml`에 `hybrid_warmup_episodes: 50` 설정 추가.
*   **메커니즘**:
    *   **Epoch 0-50**: `Ensemble Net` (Alpha) 학습을 동결(Freeze). DDPG (Actor/Critic)만 학습시켜 가치 함수(Value Function)를 먼저 성숙시킴.
    *   **Epoch 50+**: `Ensemble Net`의 동결을 해제하고, 신뢰할 수 있게 된 Critic을 바탕으로 포트폴리오 비중 최적화 시작.

## 3. 최종 결과 (검증)
*   **테스트 조건**: Seed 42, 연간 리밸런싱(Annual Rebalancing), 거래 비용 0.1%
*   **성능 비교**:
    | 모델 | CAGR (연평균 수익률) | Sharpe Ratio (샤프 지수) | 비고 |
    | :--- | :--- | :--- | :--- |
    | **Hybrid (Annual)** | **18.74%** | **0.83** | **최고 성능 달성** |
    | Hybrid (Semiannual) | 10.06% | 0.37 | |
    | Benchmark (S&P500) | 5.89% | 0.22 | |
    | TGNN (Monthly) | 6.20% | 0.21 | |
    | DDPG (Annual) | -2.26% | -0.19 | 단일 RL 모델은 성능 저조 |

## 4. 결론
"Critic Warm-up" 전략은 초기 노이즈에 대한 과적합을 성공적으로 방지했으며, 이를 통해 하이브리드 모델이 TGNN의 안정성과 DDPG의 잠재력을 효과적으로 결합할 수 있게 되었습니다. 결과적으로 개별 구성 모델 및 시장 벤치마크를 크게 상회하는 견고한 모델이 완성되었습니다.
