"""
Multi-Seed Experiment Runner for Academic Reproducibility.

목적: 5개 seed로 전체 파이프라인(Train → Backtest)을 반복 실행하고
     mean ± std를 포함한 결과 테이블을 자동 생성한다.

사용법:
    python run_multiseed.py

출력:
    results/multiseed/seed_{N}/  — seed별 체크포인트 및 로그
    results/multiseed/summary.csv — mean ± std 집계 결과
"""

import os
import copy
import json
import yaml
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

# [FIX] OpenMP 중복 런타임 억제
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 실험할 Seed 목록
SEEDS: List[int] = [0, 42, 123, 456, 789]

# 결과 저장 루트
MULTISEED_DIR: str = "results/multiseed"


def load_config(path: str = "config/config.yaml") -> Dict[str, Any]:
    """config.yaml 로드."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.join(script_dir, path)
    with open(abs_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """모든 RNG에 seed 고정."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Fixed: {seed}, Deterministic: {deterministic}")


def redirect_results(config: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """
    seed별로 results 경로를 분리하여 실험 간 덮어쓰기 방지.
    예: results/multiseed/seed_42/
    """
    seed_dir = os.path.join(MULTISEED_DIR, f"seed_{seed}")
    config = copy.deepcopy(config)
    config["paths"]["results_dir"] = seed_dir
    return config


def run_single_seed(seed: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    단일 seed로 Train → Backtest 전체 파이프라인 실행.

    Args:
        seed: 실험 seed 값
        config: 기본 config 딕셔너리

    Returns:
        해당 seed의 backtest metrics 딕셔너리
    """
    from src.pipelines.train_pipeline import run_train
    from src.pipelines.backtest_pipeline import run_backtest

    print(f"\n{'='*60}")
    print(f"  SEED {seed} 실험 시작")
    print(f"{'='*60}")

    # seed별 결과 경로 분리
    seed_config = redirect_results(config, seed)
    seed_dir = seed_config["paths"]["results_dir"]
    os.makedirs(seed_dir, exist_ok=True)

    # seed 고정
    set_seed(seed)

    # 사용할 모델 목록 (ALL 모드)
    models_to_run = ["tgnn", "ddpg", "hybrid"]

    for model in models_to_run:
        print(f"\n[Train] Model: {model.upper()} | Seed: {seed}")
        model_config = copy.deepcopy(seed_config)
        model_config["project"]["selected_model"] = model
        # seed를 config에도 반영 (재현성 로그용)
        model_config["model"]["reproducibility"]["seed"] = seed
        try:
            run_train(model_config)
        except Exception as e:
            print(f"[ERROR] Train {model.upper()} seed={seed}: {e}")
            continue

    # backtest_metrics.csv 컬럼 → aggregate_results 기대 컬럼 매핑
    col_map = {
        "CAGR": "CAGR (%)",
        "Sharpe": "Sharpe Ratio",
        "MDD": "MDD (%)",
        "Total_Return": "Total Return (%)",
    }

    # Backtest — TGNN, DDPG, Hybrid 모두 실행
    all_model_results: Dict[str, Any] = {}
    for backtest_model in ["tgnn", "ddpg", "hybrid"]:
        print(f"\n[Backtest] {backtest_model.upper()} | Seed: {seed}")
        bt_config = copy.deepcopy(seed_config)
        bt_config["project"]["selected_model"] = backtest_model
        bt_config["model"]["reproducibility"]["seed"] = seed
        try:
            run_backtest(bt_config, model_path=None)
        except Exception as e:
            print(f"[ERROR] Backtest {backtest_model.upper()} seed={seed}: {e}")
            continue

        metrics_path = os.path.join(
            seed_dir, backtest_model, "logs", "backtest_metrics.csv"
        )
        if not os.path.exists(metrics_path):
            print(f"[WARN] {backtest_model.upper()} 결과 파일 없음: {metrics_path}")
            continue

        df = pd.read_csv(metrics_path)
        df = df.rename(columns=col_map)
        for _, row in df.iterrows():
            strategy = row["Strategy"]
            # Benchmark는 최초 1회만 저장
            if strategy == "Benchmark" and strategy in all_model_results:
                continue
            all_model_results[strategy] = row.to_dict()
            all_model_results[strategy]["seed"] = seed

    if all_model_results:
        print(f"[OK] Seed {seed} 결과 수집 완료: {len(all_model_results)}개 전략")
        return all_model_results
    else:
        print(f"[WARN] Seed {seed} 수집된 결과 없음")
        return {}


def aggregate_results(all_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    5개 seed 결과를 집계하여 mean ± std 테이블 생성.

    Args:
        all_results: seed별 결과 딕셔너리 리스트

    Returns:
        전략별 mean ± std DataFrame
    """
    # 전략 목록 추출
    strategies = set()
    for result in all_results:
        strategies.update(result.keys())

    numeric_cols = ["CAGR (%)", "Sharpe Ratio", "MDD (%)", "Total Return (%)"]

    rows = []
    for strategy in sorted(strategies):
        seed_values: Dict[str, List[float]] = {col: [] for col in numeric_cols}

        for result in all_results:
            if strategy in result:
                for col in numeric_cols:
                    val = result[strategy].get(col)
                    if val is not None:
                        try:
                            seed_values[col].append(float(val))
                        except (ValueError, TypeError):
                            pass

        row = {"Strategy": strategy}
        for col in numeric_cols:
            vals = seed_values[col]
            if vals:
                mean_val = np.mean(vals)
                std_val = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                row[f"{col} Mean"] = round(mean_val, 4)
                row[f"{col} Std"] = round(std_val, 4)
                row[f"{col} (mean±std)"] = f"{mean_val:.2f} ± {std_val:.2f}"
            else:
                row[f"{col} Mean"] = None
                row[f"{col} Std"] = None
                row[f"{col} (mean±std)"] = "N/A"
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    """메인 실행 함수."""
    print("\n" + "="*60)
    print("  Multi-Seed Reproducibility Experiment")
    print(f"  Seeds: {SEEDS}")
    print("="*60)

    # 출력 디렉토리 생성
    os.makedirs(MULTISEED_DIR, exist_ok=True)

    # Config 로드
    config = load_config()
    config["project"]["selected_model"] = "ALL"

    # 각 seed 실험 실행
    all_results: List[Dict[str, Any]] = []
    failed_seeds: List[int] = []

    for seed in SEEDS:
        result = run_single_seed(seed, config)
        if result:
            all_results.append(result)
        else:
            failed_seeds.append(seed)

    if failed_seeds:
        print(f"\n[WARN] 실패한 Seed: {failed_seeds}")

    if not all_results:
        print("[ERROR] 수집된 결과 없음. 실험을 확인하세요.")
        return

    # 결과 집계
    print(f"\n[집계] {len(all_results)}개 seed 결과 집계 중...")
    summary_df = aggregate_results(all_results)

    # 저장
    summary_path = os.path.join(MULTISEED_DIR, "summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # 논문용 테이블 (mean ± std 컬럼만 추출)
    paper_cols = ["Strategy"] + [
        f"{col} (mean±std)" for col in
        ["CAGR (%)", "Sharpe Ratio", "MDD (%)", "Total Return (%)"]
    ]
    paper_df = summary_df[paper_cols]
    paper_path = os.path.join(MULTISEED_DIR, "paper_table.csv")
    paper_df.to_csv(paper_path, index=False, encoding="utf-8-sig")

    # 실험 메타 정보 저장
    meta = {
        "seeds": SEEDS,
        "completed_seeds": [s for s in SEEDS if s not in failed_seeds],
        "failed_seeds": failed_seeds,
        "n_strategies": len(summary_df),
    }
    with open(os.path.join(MULTISEED_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print("  실험 완료")
    print(f"  - 요약 결과: {summary_path}")
    print(f"  - 논문용 테이블: {paper_path}")
    print(f"{'='*60}")
    print("\n[논문용 결과 (mean ± std)]")
    print(paper_df.to_string(index=False))


if __name__ == "__main__":
    main()
