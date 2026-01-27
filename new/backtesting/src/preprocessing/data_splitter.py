"""
데이터 분할 모듈
Train/Test 데이터를 섹터별로 분할하고 저장합니다.
"""

import pandas as pd
import os
from typing import List, Tuple, Dict, Any


class DataSplitter:
    """Train/Test 데이터 분할 클래스."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_stocks_info: List[Dict[str, Any]],
        test_stocks_info: List[Dict[str, Any]],
    ):
        """
        Args:
            train_df: Train 전체 데이터
            test_df: Test 전체 데이터
            train_stocks_info: Train 종목 정보 리스트
            test_stocks_info: Test 종목 정보 리스트
        """
        self.train_df = train_df
        self.test_df = test_df
        self.train_stocks_info = train_stocks_info
        self.test_stocks_info = test_stocks_info

    def split_by_sector(
        self, train_per_sector: int = 5, test_total: int = 7
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
        """
        섹터별로 종목을 선택하여 분할합니다.

        Args:
            train_per_sector: 섹터당 Train 종목 수
            test_total: Test 총 종목 수

        Returns:
            (train_df, test_df, train_symbols, test_symbols)
        """
        print("\n✂️ 섹터별 종목 선택 중...")

        # Train 종목 선택
        train_symbols_selected = []
        train_sectors = self.train_df["Sector"].unique()

        print(f"\n[Train] 섹터당 최대 {train_per_sector}개")
        for sector in sorted(train_sectors):
            sector_symbols = self.train_df[self.train_df["Sector"] == sector][
                "Symbol"
            ].unique()

            # 데이터 품질 체크 (최소 1000행)
            valid_symbols = []
            for sym in sector_symbols:
                count = len(self.train_df[self.train_df["Symbol"] == sym])
                if count >= 1000:
                    valid_symbols.append(sym)

            selected = valid_symbols[:train_per_sector]
            train_symbols_selected.extend(selected)
            print(f"   [{sector:30s}] {len(selected):2d}개")

        # Test 종목 선택
        test_symbols_selected = []
        test_sectors = self.test_df["Sector"].unique()
        stocks_per_sector = max(1, test_total // len(test_sectors))

        print(f"\n[Test] 총 {test_total}개 (섹터당 약 {stocks_per_sector}개)")
        for sector in sorted(test_sectors):
            sector_symbols = self.test_df[self.test_df["Sector"] == sector][
                "Symbol"
            ].unique()

            valid_symbols = []
            for sym in sector_symbols:
                count = len(self.test_df[self.test_df["Symbol"] == sym])
                if count >= 600:
                    valid_symbols.append(sym)

            selected = valid_symbols[:stocks_per_sector]
            test_symbols_selected.extend(selected)
            print(f"   [{sector:30s}] {len(selected):2d}개")

            if len(test_symbols_selected) >= test_total:
                break

        test_symbols_selected = test_symbols_selected[:test_total]

        # DataFrame 필터링
        final_train_df = self.train_df[
            self.train_df["Symbol"].isin(train_symbols_selected)
        ].copy()

        final_test_df = self.test_df[
            self.test_df["Symbol"].isin(test_symbols_selected)
        ].copy()

        print(f"\n✅ 최종 선택:")
        print(
            f"   Train: {len(train_symbols_selected)}개 종목, {len(final_train_df):,}행"
        )
        print(
            f"   Test:  {len(test_symbols_selected)}개 종목, {len(final_test_df):,}행"
        )

        return (
            final_train_df,
            final_test_df,
            train_symbols_selected,
            test_symbols_selected,
        )

    def save_datasets(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str = "."
    ) -> None:
        """
        Train/Test 데이터를 CSV로 저장합니다.

        Args:
            train_df: Train 데이터
            test_df: Test 데이터
            output_dir: 출력 디렉토리
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        train_path = os.path.join(output_dir, "train_data.csv")
        test_path = os.path.join(output_dir, "test_data.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"\n💾 저장 완료: {train_path} ({len(train_df):,}행)")
        print(f"💾 저장 완료: {test_path} ({len(test_df):,}행)")
