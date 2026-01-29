# 🧬 Hybrid Agent (TGNN + DDPG Ensemble)

> **Status**: Active Research 🧪
> **Task**: Portfolio Optimization
> **Key Feature**: Dual-Path Ensemble (Graph-based + MLP-based) with Dynamic Constraints

## 1. 개요 (Overview)

**Hybrid Agent**는 시계열-그래프 특징을 잘 추출하는 **TGNN**과, 전체 시장 상황을 최적화하는 **DDPG**를 결합한 앙상블 모델입니다.
"종목 간 관계(Micro-view)"와 "시장 전체 흐름(Macro-view)"을 동시에 고려하여 최적의 포트폴리오 비중을 결정합니다.

---

## 2. 아키텍처 (Architecture)

### 2.1 Hybrid Actor (`src/models/hybrid/actor.py`)

Actor는 두 개의 독립적인 인코더 경로(Path)를 가지며, 학습 가능한 파라미터 `Alpha`를 통해 두 경로의 출력을 섞습니다.

1.  **TGNN Path (Micro-view)**
    - **Encoder**: 2-Layer GCN + Temporal Attention (`encoders.TGNNEncoder`)
    - **Input**: `(Batch, N, T, F)` + `Adj(N, N)`
    - **Logic**: 종목별 임베딩 생성 -> Softmax -> `Weights_TGNN`
    - **특징**: 종목 간 상관관계를 반영한 비중 생성.

2.  **DDPG Path (Macro-view)**
    - **Encoder**: MLP (`encoders.DDPGEncoder`)
    - **Input**: Flattened Features `(N*T*F)` + Flattened Adj `(N*N)`
    - **Logic**: 전체 상태 임베딩 -> MLP -> Softmax -> `Weights_DDPG`
    - **특징**: 전체 포트폴리오 관점의 비중 생성.

3.  **Ensemble Mechanism**
    - `Alpha`: Ensemble Head (`State` + `W_TGNN` + `W_DDPG` -> `Scalar Alpha`)
    - **Formula**: `Final_W = Alpha * W_TGNN + (1 - Alpha) * W_DDPG`
    - **Constraint**: `Alpha`는 `[0.2, 0.8]` 사이로 제한됨 (어느 한쪽에만 의존하는 것 방지).

### 2.2 Dynamic Constraints (`src/models/hybrid/constraints.py`)

시장 상황(MDD 등)에 따라 포트폴리오 제약 조건을 동적으로 조절합니다.

- **Normal State**: `Max Weight = 0.25`, `Min Weight = 0.00`
- **High Risk (MDD > 15%)**: `Max Weight = 0.15` (분산 투자 강제), `Min Weight = 0.07` (현금/안전자산 비중 확보 느낌)

---

## 3. 입력 및 출력 명세 (I/O Specification)

### Input Tensor
- **Features**: `[Batch, N, T, F]`
- **Adjacency**: `[Batch, N, N]` (Correlation Matrix)

### Output Tensor
- **Final Weights**: `[Batch, N]` (Sum=1.0)
- **Alpha**: `[Batch, 1]` (Ensemble Ratio)

---

## 4. 설정 (Configuration)

`config.yaml`에서 두 모델의 하이퍼파라미터를 모두 제어합니다.

```yaml
model:
  tgnn:
    hidden_dim: 64      # GNN 임베딩 차원
    num_heads: 4        # Attention Head 수
  
  ddpg:
    hidden_dim: 128     # DDPG MLP 차원
    
training:
  dropout: 0.1          # Encoding Layer Dropout
```

## 5. 재현성 및 주의사항

- **Constraint Enforcement**: `PortfolioConstraints` 클래스가 마지막 단계에서 비중의 합을 1.0으로 맞추고 Min/Max를 강제하므로, 모델의 Raw Output과 최종 비중이 다를 수 있습니다.
- **Fixed Universe**: DDPG Path가 포함되어 있으므로, **학습 시와 테스트 시 종목 수(N)가 동일해야 합니다.** (순수 TGNN과 다름)
