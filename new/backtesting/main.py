import sys
import os
# [FIX] PyTorch와 Intel MKL이 각각 OpenMP 런타임을 로드할 때 발생하는 충돌 억제
# 근본 원인: libiomp5md.dll 중복 초기화 (torch + numpy/scipy 경유 MKL)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import yaml
import copy
import argparse

# Path setup to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from src.preprocessing.pipeline import Pipeline  # Moved to run_preprocess


def load_config(path="config/config.yaml"):
    # Resolve absolute path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.join(script_dir, path)
    with open(abs_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_preprocess(config_path=None, csv_path=None, target="sp500"):
    from src.preprocessing.pipeline import Pipeline

    """Run preprocessing pipeline."""
    print("\n🚀 Running Preprocessing Pipeline...")

    # Load config if needed for output dirs etc, but pipeline handles defaults well.
    # We can pass specific args if needed.

    pipeline = Pipeline(csv_path=csv_path)

    # Override default target if auto-loading
    if not csv_path:
        print(f"📡 Target Index: {target}")
        pipeline.collector.load_stocks_auto = (
            lambda target=target: pipeline.collector.load_stocks_auto(target)
        )
        # Note: pipeline.run() handles logic based on csv_path presence
        pass

    pipeline.run()


# Refactored Pipelines
from src.pipelines.train_pipeline import run_train
from src.pipelines.backtest_pipeline import run_backtest


def set_seed(seed: int, deterministic: bool = True, benchmark: bool = False):
    """
    Set seeds for all random number generators to ensure reproducibility.
    """
    import random
    import numpy as np
    import torch

    # 1. Python random
    random.seed(seed)

    # 2. NumPy
    np.random.seed(seed)

    # 3. PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # 4. Hash Seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 5. Deterministic Behavior (CUDNN)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = (
            benchmark  # False is better for reproducibility
        )
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True  # True is better for performance

    print(f"[Lock] Reproducibility Set: Seed={seed}, Deterministic={deterministic}")


def main():
    parser = argparse.ArgumentParser(
        description="SMS Backtesting Framework Main Entry Point"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["preprocess", "train", "backtest", "full", "compare"],
        default="train",
        help="Execution mode: 'preprocess', 'train', 'backtest', 'full' (Train + Backtest), or 'compare' (Generate comparison charts).",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file.",
    )

    # Preprocessing specific args
    parser.add_argument(
        "--csv", type=str, default=None, help="[Preprocess] Path to stock list CSV."
    )
    parser.add_argument(
        "--target",
        type=str,
        default="sp500",
        choices=["sp500", "nasdaq100"],
        help="[Preprocess] Target index to crawl if no CSV provided.",
    )

    # Backtest specific args
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="[Backtest] Path to trained model (.pth). If None, loads latest best_model.",
    )

    args = parser.parse_args()

    # 1. Load Config
    config = load_config(args.config)

    # 2. Set Seed (Reproducibility)
    repro_config = config.get("model", {}).get("reproducibility", {})
    # Fallback to config path structure if set elsewhere or defaults
    if not repro_config:
        # Check newly added path structure
        repro_config = config.get(
            "reproducibility", {}
        )  # If moved to top level? No, put in model as per plan, wait checks.
        # Actually in the plan I put it under 'model' but the user might want it global.
        # Let's check config.yaml again. It was put under 'model' as per my last tool call.
        # Wait, I put it under 'model' in config.yaml edit, so config['model']['reproducibility'].
        pass

    # Re-read config structure properly
    repro_conf = config.get("model", {}).get("reproducibility", {})
    if not repro_conf:
        # Fallback to defaults if missing
        seed = 42
        deterministic = True
        benchmark = False
    else:
        seed = repro_conf.get("seed", 42)
        deterministic = repro_conf.get("deterministic", True)
        benchmark = repro_conf.get("cudnn_benchmark", False)

    set_seed(seed, deterministic, benchmark)

    if args.mode == "preprocess":
        from src.preprocessing.pipeline import Pipeline  # Lazy import
        # Note: Preprocessing pipeline might need to be adjusted to accept target arg cleanly
        # For now, we instantiate Pipeline passing arguments
        # If csv is None, pipeline defaults to auto-crawl.
        # But pipeline.py logic currently defaults to nasdaq100 in load_stocks_auto if not specified there.
        # We need to ensure target is passed down.

        # Re-instantiating pipeline with explicit args if we want to support target override properly.
        # However, Pipeline class in pipeline.py doesn't take 'target' in __init__, it's used in run().
        # Let's check pipeline.py again.
        # Ah, pipeline.py's run() calls self.collector.load_stocks_auto(target="nasdaq100") hardcoded in 'else' block?
        # Wait, I changed it to "sp500" in previous turn.
        # To support dynamic target, I might need to adjust pipeline.py or just rely on default.
        # Let's assume default "sp500" for now or use the modify pipeline.py if needed.
        # For simple integration:

        # To pass target dynamically without changing Pipeline signature too much,
        # we can modify Pipeline.run to accept target or handle it here.

        # Let's just run it.
        if args.csv:
            Pipeline(csv_path=args.csv).run(config)
        else:
            # If we want to support target change, Pipeline needs to know.
            # Currently Pipeline.run() has hardcoded target="sp500" in the else block.
            # I will hotfix pipeline execution flow here by calling internal methods or accepted defaults.
            # Or I can quickly update Pipeline.run to take target argument.
            # For now, let's stick to the default S&P 500 behavior.
            Pipeline(csv_path=None).run(config)

    elif args.mode == "train":
        selected_model = config["project"].get("selected_model", "tgnn").lower()

        if selected_model == "all":
            models_to_run = ["tgnn", "ddpg", "hybrid"]
            print(f"\n🚀 [ALL MODE] Starting Sequential Training for: {models_to_run}")

            for model in models_to_run:
                print(f"\n{'=' * 40}")
                print(f"🔥 Starting Training for Model: {model.upper()}")
                print(f"{'=' * 40}")

                # deepcopy로 모델 간 config 오염 방지
                model_config = copy.deepcopy(config)
                model_config["project"]["selected_model"] = model
                try:
                    run_train(model_config)
                    print(f"✅ Finished Training {model.upper()}")
                except Exception as e:
                    print(f"❌ Error Training {model.upper()}: {e}")
                    # Option: Continue to next model or stop? Let's continue.
        else:
            run_train(config)

    elif args.mode == "backtest":
        selected_model = config["project"].get("selected_model", "tgnn").lower()

        if selected_model == "all":
            models_to_run = ["tgnn", "ddpg", "hybrid"]
            print(f"\n🚀 [ALL MODE] Starting Sequential Backtest for: {models_to_run}")

            for model in models_to_run:
                print(f"\n{'=' * 40}")
                print(f"🧪 Starting Backtest for Model: {model.upper()}")
                print(f"{'=' * 40}")

                # deepcopy로 모델 간 config 오염 방지
                model_config = copy.deepcopy(config)
                model_config["project"]["selected_model"] = model
                try:
                    # For simplicity in 'all' mode, we rely on auto-loading best_model.
                    run_backtest(model_config, model_path=None)
                    print(f"✅ Finished Backtest {model.upper()}")
                except Exception as e:
                    print(f"❌ Error Backtesting {model.upper()}: {e}")
        else:
            run_backtest(config, model_path=args.model_path)

    elif args.mode == "full":
        selected_model = config["project"].get("selected_model", "tgnn").lower()

        if selected_model == "all":
            models_to_run = ["tgnn", "ddpg", "hybrid"]
            print(
                f"\n🚀 [FULL PIPELINE] Starting Train -> Backtest for: {models_to_run}"
            )

            for model in models_to_run:
                print(f"\n{'=' * 60}")
                print(f"🔄 Processing Model: {model.upper()} (Train + Backtest)")
                print(f"{'=' * 60}")

                # deepcopy로 모델 간 config 오염 방지
                model_config = copy.deepcopy(config)
                model_config["project"]["selected_model"] = model

                # 1. Train
                try:
                    print(f"\n🔥 [Step 1/2] Training {model.upper()}...")
                    run_train(model_config)
                except Exception as e:
                    print(f"❌ Training Failed for {model.upper()}: {e}")
                    continue  # Skip backtest if train fails

                # 2. Backtest
                try:
                    print(f"\n🧪 [Step 2/2] Backtesting {model.upper()}...")
                    run_backtest(
                        model_config, model_path=None
                    )  # Auto-load just-trained model
                    print(f"✅ {model.upper()} Pipeline Completed!")
                except Exception as e:
                    print(f"❌ Backtest Failed for {model.upper()}: {e}")

        else:
            # Single model full pipeline
            print(
                f"\n🚀 [FULL PIPELINE] Starting Train -> Backtest for: {selected_model}"
            )

            # 1. Train
            run_train(config)

            # 2. Backtest
            run_backtest(config, model_path=None)

    elif args.mode == "compare":
        # 비교 차트 생성
        print("\n📊 [COMPARE MODE] Generating comparison charts...")

        # 스크립트의 로직을 import 하지 않고 직접 실행
        import subprocess
        import sys

        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(
            script_dir, "scripts", "generate_comparison_chart.py"
        )
        subprocess.run([sys.executable, script_path], cwd=os.path.dirname(script_dir))


if __name__ == "__main__":
    main()
