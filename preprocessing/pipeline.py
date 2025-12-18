# 전체 파이프라인 통합

import pandas as pd
from data_collector import DataCollector
from technical_indicators import TechnicalIndicators
from factor_calculator import FactorCalculator
from fama_french_loader import FamaFrenchLoader
from data_splitter import DataSplitter


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

        # 1. 종목 로드 및 생존 종목 필터링
        self.collector.load_stocks_from_csv(self.csv_path)
        survivor_stocks = self.collector.filter_survivor_stocks(
            start_year=self.start_year, end_year=self.end_year
        )

        if not survivor_stocks:
            print("❌ No survivor stocks found. Exiting.")
            return

        # 2. Fama-French 데이터 미리 다운로드
        self.ff_loader.download_factors(
            start_date=f"{self.start_year}-01-01", end_date=f"{self.end_year}-12-31"
        )

        all_stocks_data = []

        # 3. 각 종목별 데이터 처리 루프
        print(f"\n🔄 Processing {len(survivor_stocks)} stocks...")
        for stock_info in survivor_stocks:
            symbol = stock_info["Symbol"]

            # 3-1. OHLCV 데이터 수집
            df = self.collector.fetch_daily_data(
                symbol,
                start_date=f"{self.start_year}-01-01",
                end_date=f"{self.end_year}-12-31",
            )

            if df is None or df.empty:
                continue

            # 3-2. 기술적 지표 추가
            df = TechnicalIndicators.add_all_indicators(df)

            # 3-3. 팩터 점수 계산
            df = FactorCalculator.calculate_factors(df)
            df = FactorCalculator.calculate_weighted_score(df)

            # 메타데이터(섹터 등) 보존
            df["Sector"] = stock_info["Sector"]
            df["Industry"] = stock_info.get("Industry", "Unknown")
            df["Date"] = df.index  # 인덱스를 컬럼으로

            all_stocks_data.append(df)

        if not all_stocks_data:
            print("❌ No data collected.")
            return

        # 4. 전체 데이터 병합
        full_df = pd.concat(all_stocks_data, ignore_index=True)
        print(f"📊 Total Records Collected: {len(full_df)}")

        # 5. Fama-French 병합
        full_df = self.ff_loader.merge_with_stock_data(full_df)

        # 6. 학습/테스트 데이터 분할 (섹터별 분할 로직 적용)
        splitter = DataSplitter(full_df, survivor_stocks)
        train_df, test_df, train_list, test_list = splitter.split_by_sector_and_date(
            train_end_year=2020
        )

        # 7. 저장
        splitter.save_datasets(train_df, test_df, self.output_dir)

        print("\n✅ Pipeline Completed Successfully.")
        print(f"   Train Stocks ({len(train_list)}): {train_list}")
        print(f"   Test Stocks ({len(test_list)}): {test_list}")


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



