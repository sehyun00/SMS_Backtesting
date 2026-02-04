---
trigger: always_on
glob: "**/*"
description: 프로젝트 핵심 규칙, 연구 무결성 원칙, 코드 수정 가이드라인 (항상 적용)
---

# Project Context & Integrity Rules

> ⚙️ **System Notice**: 이 규칙은 학술 연구의 무결성을 위해 **모든 대화에 항상 적용**됩니다.
> 답변 시작 시 **"📌 [Research Integrity Active]"** 를 짧게 표시하여 룰 적용 상태를 알리세요.

## 1. 📂 프로젝트 메타 정보 (Context)
- **목적:** 학술 논문(Conference/Journal) 제출용 장기 투자 포트폴리오 백테스팅
- **핵심 모델:**
    - **예측:** TGNN (Temporal Graph Neural Network)
    - **정책:** DDPG (Deep Deterministic Policy Gradient)
- **데이터:** KOSPI & S&P500 (2015~2024), Fama-French 5-Factor 사용
- **리밸런싱:** 월간/분기/반기/연간 단위

## 2. 🛑 연구 무결성 원칙 (Absolute Rules)
**다음 원칙을 위반하면 논문이 거절될 수 있으므로 엄격히 준수하세요.**

1.  **Look-ahead Bias 금지:**
    - 테스트 데이터(2024-2025)를 학습이나 전처리(`fit`)에 절대 사용하지 마세요.
2.  **재현 가능성 (Reproducibility):**
    - `torch`, `numpy`, `random`의 Seed를 고정하는 코드가 항상 포함되어야 합니다.
3.  **Hallucination 금지:**
    - 결과 파일(`results/*.csv`)에 없는 수치를 지어내지 마세요.

## 3. 🛠️ 코드 수정 워크플로우
1.  **README 우선 확인:** 작업할 디렉토리의 `README.md`를 먼저 읽으세요.
2.  **README 동기화:** 코드가 변경되면 **반드시 `README.md`도 함께 업데이트**하세요.