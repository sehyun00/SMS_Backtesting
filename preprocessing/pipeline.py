# 전체 파이프라인 통합

import pandas as pd
from .data_collector import DataCollector
from .technical_indicators import TechnicalIndicators
from .factor_calculator import FactorCalculator
from .fama_french_loader import FamaFrenchLoader
from .data_splitter import DataSplitter


class Pipeline:
    def __init__(self, csv_path, output_dir="data"):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.collector = DataCollector()
        self.ff_loader = FamaFrenchLoader()
        self.start_year = 2006
        self.end_year = 2025

    def run(self):
        print("🚀 Starting Preprocessing Pipeline...")

        # 1. 종목 로드
        self.collector.load_stocks_from_csv(self.csv_path)

        # 🔥 수정: Train/Test 종목 분리 필터링
        survivors = self.collector.filter_survivor_stocks(
            start_year=self.start_year, end_year=self.end_year
        )

        train_candidate_stocks = survivors["train"]
        test_candidate_stocks = survivors["test"]

        if not train_candidate_stocks:
            print("❌ No train stocks found. Exiting.")
            return

        # 2. Fama-French 데이터 미리 다운로드
        self.ff_loader.download_factors(
            start_date=f"{self.start_year}-01-01", end_date=f"{self.end_year}-12-31"
        )

        # 🔥 수정: Train 데이터 수집 (2006-2020)
        print(
            f"\n📊 Processing {len(train_candidate_stocks)} TRAIN stocks (2006-2020)..."
        )
        train_data_list = []

        for stock_info in train_candidate_stocks:
            symbol = stock_info["Symbol"]

            # Train 기간만 수집
            df = self.collector.fetch_daily_data(
                symbol, start_date="2006-01-01", end_date="2020-12-31"
            )

            if df is None or df.empty:
                continue

            # 기술적 지표 & 팩터 추가
            df = TechnicalIndicators.add_all_indicators(df)
            df = FactorCalculator.calculate_factors(df)
            df = FactorCalculator.calculate_weighted_score(df)

            # 메타데이터
            df["Sector"] = stock_info["Sector"]
            df["Industry"] = stock_info.get("Industry", "Unknown")
            df["Date"] = df.index

            train_data_list.append(df)

        # 🔥 수정: Test 데이터 수집 (2021-2025)
        print(
            f"\n📊 Processing {len(test_candidate_stocks)} TEST stocks (2021-2025)..."
        )
        test_data_list = []

        for stock_info in test_candidate_stocks:
            symbol = stock_info["Symbol"]

            # Test 기간만 수집
            df = self.collector.fetch_daily_data(
                symbol, start_date="2021-01-01", end_date="2025-12-31"
            )

            if df is None or df.empty:
                continue

            # 동일 처리
            df = TechnicalIndicators.add_all_indicators(df)
            df = FactorCalculator.calculate_factors(df)
            df = FactorCalculator.calculate_weighted_score(df)

            df["Sector"] = stock_info["Sector"]
            df["Industry"] = stock_info.get("Industry", "Unknown")
            df["Date"] = df.index

            test_data_list.append(df)

        if not train_data_list:
            print("❌ No train data collected.")
            return

        # 🔥 수정: Train/Test 병합
        train_full_df = pd.concat(train_data_list, ignore_index=True)
        test_full_df = pd.concat(test_data_list, ignore_index=True)

        print(f"📊 Train Total: {len(train_full_df):,} rows")
        print(f"📊 Test Total: {len(test_full_df):,} rows")

        # Fama-French 병합
        train_full_df = self.ff_loader.merge_with_stock_data(train_full_df)
        test_full_df = self.ff_loader.merge_with_stock_data(test_full_df)

        # 🔥 수정: 섹터별 종목 선택
        splitter = DataSplitter(
            train_full_df, test_full_df, train_candidate_stocks, test_candidate_stocks
        )

        final_train_df, final_test_df, train_symbols, test_symbols = (
            splitter.split_by_sector(
                train_per_sector=5,  # 섹터당 5개
                test_total=7,  # 총 7개 고정
            )
        )

        # 저장
        splitter.save_datasets(final_train_df, final_test_df, self.output_dir)

        print("\n✅ Pipeline Completed Successfully.")
        print(f"   Train Stocks ({len(train_symbols)}): {train_symbols}")
        print(f"   Test Stocks ({len(test_symbols)}): {test_symbols}")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    # 🔥 절대 경로 계산 로직 추가
    SCRIPT_DIR = Path(__file__).resolve().parent  # .../preprocessing
    PROJECT_ROOT = SCRIPT_DIR.parent  # .../SMS_Back_Testing

    # 기본 경로 설정
    DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "nasdaq100_stock_list.csv"
    DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data"

    parser = argparse.ArgumentParser(
        description="SMS Backtesting Data Preprocessing Pipeline"
    )

    parser.add_argument(
        "--csv",
        type=str,
        default=str(DEFAULT_CSV_PATH),
        help=f"Path to stock list CSV file (default: {DEFAULT_CSV_PATH})",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to save output files (default: {DEFAULT_OUTPUT_DIR})",
    )

    args = parser.parse_args()

    # 파이프라인 실행
    pipeline = Pipeline(csv_path=args.csv, output_dir=args.output)
    pipeline.run()



