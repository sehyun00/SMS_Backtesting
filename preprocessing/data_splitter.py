# 학습/테스트 분할

import pandas as pd
import os


class DataSplitter:
    def __init__(self, train_df, test_df, train_stocks_info, test_stocks_info):
        """
        🔥 수정: Train/Test DataFrame을 별도로 받음

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

    def split_by_sector(self, train_per_sector=5, test_total=7):
        """
        🔥 새 메서드: 섹터별 종목 선택

        Train: 섹터당 최대 N개 선택
        Test: 총 M개 선택 (섹터 균형 고려)

        Args:
            train_per_sector: 섹터당 Train 종목 수
            test_total: Test 총 종목 수

        Returns:
            train_df, test_df, train_symbols, test_symbols
        """
        print("\n✂️ Selecting stocks by sector...")

        # ============== Train 종목 선택 ==============
        train_symbols_selected = []
        train_sectors = self.train_df["Sector"].unique()

        print(f"\n[Train Selection] 섹터당 최대 {train_per_sector}개")
        for sector in sorted(train_sectors):
            # 해당 섹터의 종목들
            sector_symbols = self.train_df[self.train_df["Sector"] == sector][
                "Symbol"
            ].unique()

            # 데이터 품질 체크 (최소 1000행 이상)
            valid_symbols = []
            for sym in sector_symbols:
                count = len(self.train_df[self.train_df["Symbol"] == sym])
                if count >= 1000:  # 약 4년치
                    valid_symbols.append(sym)

            # 상위 N개 선택
            selected = valid_symbols[:train_per_sector]
            train_symbols_selected.extend(selected)

            print(f"   [{sector:30s}] {len(selected):2d}개 선택")

        # ============== Test 종목 선택 ==============
        test_symbols_selected = []
        test_sectors = self.test_df["Sector"].unique()

        # 섹터당 할당 개수 계산
        stocks_per_sector = max(1, test_total // len(test_sectors))

        print(f"\n[Test Selection] 총 {test_total}개 (섹터당 약 {stocks_per_sector}개)")
        for sector in sorted(test_sectors):
            sector_symbols = self.test_df[self.test_df["Sector"] == sector][
                "Symbol"
            ].unique()

            # 데이터 품질 체크
            valid_symbols = []
            for sym in sector_symbols:
                count = len(self.test_df[self.test_df["Symbol"] == sym])
                if count >= 600:  # 약 2.5년치
                    valid_symbols.append(sym)

            # 상위 N개 선택
            selected = valid_symbols[:stocks_per_sector]
            test_symbols_selected.extend(selected)

            print(f"   [{sector:30s}] {len(selected):2d}개 선택")

            # 목표 달성 시 중단
            if len(test_symbols_selected) >= test_total:
                break

        # 정확히 test_total개 맞추기
        test_symbols_selected = test_symbols_selected[:test_total]

        # ============== DataFrame 필터링 ==============
        final_train_df = self.train_df[
            self.train_df["Symbol"].isin(train_symbols_selected)
        ].copy()

        final_test_df = self.test_df[
            self.test_df["Symbol"].isin(test_symbols_selected)
        ].copy()

        print(f"\n✅ Final Selection:")
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

    def save_datasets(self, train_df, test_df, output_dir="."):
        """CSV 저장"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        train_path = os.path.join(output_dir, "train_data.csv")
        test_path = os.path.join(output_dir, "test_data.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"\n💾 Saved: {train_path} ({len(train_df):,} rows)")
        print(f"💾 Saved: {test_path} ({len(test_df):,} rows)")
