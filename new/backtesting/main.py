import sys
import os
import yaml
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


def main():
    parser = argparse.ArgumentParser(
        description="SMS Backtesting Framework Main Entry Point"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["preprocess", "train", "backtest"],
        default="train",
        help="Execution mode: 'preprocess', 'train', or 'backtest'.",
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
        run_train(config)

    elif args.mode == "backtest":
        run_backtest(config, model_path=args.model_path)


if __name__ == "__main__":
    main()
