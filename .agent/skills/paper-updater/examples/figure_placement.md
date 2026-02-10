# Figure/Table 배치 예시

## 📊 Results 섹션에서의 Figure 배치

아래는 `05_Results_Discussion.md` 작성 시 Figure 배치 예시입니다.

---

### 예시: 성과 비교 섹션

```markdown
## 5.1 재무적 성과 비교 (Financial Performance Comparison)

제안된 하이브리드 AI DSS(TGNN+DDPG)의 성능을 검증하기 위해 
Benchmark, TGNN, DDPG, Hybrid 모델의 투자 성과를 비교하였다.

<!-- [Figure 1] 모델별 CAGR 비교 차트 -->
<!-- 권장: images/cagr_comparison_all.png -->

Figure 1은 각 모델의 리밸런싱 주기별 CAGR을 비교한 결과이다.
Hybrid 모델이 모든 주기에서 가장 높은 수익률을 기록하였다.

<!-- [Table 1] 전략별 성과 비교표 -->
<!-- 소스: results/comparison/all_strategies_comparison.csv -->

| Model | Period | CAGR (%) | Sharpe | MDD (%) |
|-------|--------|----------|--------|---------|
| Benchmark | - | 5.89 | 0.22 | 21.51 |
| Hybrid | Annual | 18.74 | 0.83 | 19.44 |

Table 1에서 보는 바와 같이, Hybrid (Annual) 전략이 
**CAGR 18.74%**, **Sharpe 0.83**을 달성하였다.

<!-- [Figure 2] Risk-Return 산점도 -->
<!-- 권장: images/risk_return_scatter_all.png -->

Figure 2는 리스크(MDD) 대비 수익률(CAGR) 관계를 시각화한 것이다.
```

---

## 🖼️ Methodology 섹션에서의 Figure 배치

```markdown
## 3.2 모델 아키텍처

<!-- [Figure 3] TGNN-DDPG 하이브리드 아키텍처 다이어그램 -->
<!-- 권장: images/hybrid_architecture.png -->

Figure 3은 제안된 하이브리드 모델의 전체 아키텍처를 보여준다.
TGNN이 시장 관계를 학습하고, DDPG가 포트폴리오 비중을 최적화한다.

<!-- [Table 2] 모델 하이퍼파라미터 설정 -->
<!-- 소스: config/config.yaml -->

| Parameter | Value | Description |
|-----------|-------|-------------|
| softmax_temperature | 10.0 | 포트폴리오 분산도 |
| episodes | 800 | 학습 에피소드 |
| buffer_size | 10000 | 리플레이 버퍼 크기 |
```

---

## 📋 배치 규칙 요약

1. **Figure 마커**: `<!-- [Figure N] 설명 -->`
2. **Table 마커**: `<!-- [Table N] 설명 -->`
3. **권장 이미지**: `<!-- 권장: images/파일명.png -->`
4. **데이터 소스**: `<!-- 소스: 경로/파일명.csv -->`
