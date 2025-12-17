# 학습/테스트 분할

import pandas as pd
import os

class DataSplitter:
    def __init__(self, df, stocks_info):
        """
        Args:
            df: 전체 주식 데이터 (DataFrame)
            stocks_info: 종목 정보 리스트 (List of Dicts: [{'Symbol': 'AAPL', 'Sector': 'Technology'}, ...])
        """
        self.df = df
        self.stocks_info = stocks_info

    def split_by_sector_and_date(self, train_end_year=2020):
        """
        섹터별로 1개는 학습용, 1개는 테스트용으로 분할합니다.
        단, 모든 종목은 '생존 종목'이어야 합니다 (DataCollector에서 이미 필터링됨).
        
        Rule:
        - 학습 데이터: 2006 ~ train_end_year (섹터별 종목 A)
        - 테스트 데이터: (train_end_year + 1) ~ 2025 (섹터별 종목 B)
        """
        print("\n✂️ Splitting data by Sector and Date...")
        
        # DataFrame에 Sector 정보가 없다면 병합
        if 'Sector' not in self.df.columns:
            sector_map = {s['Symbol']: s['Sector'] for s in self.stocks_info}
            self.df['Sector'] = self.df['Symbol'].map(sector_map)

        # 섹터별 종목 분류
        sectors = self.df['Sector'].unique()
        train_stocks = []
        test_stocks = []
        
        # 섹터별 대표 종목 선정
        unique_symbols = self.df['Symbol'].unique()
        
        for sector in sectors:
            # 현재 데이터에 존재하는 해당 섹터의 종목들 찾기
            sector_symbols = [
                s for s in unique_symbols 
                if self.df[self.df['Symbol'] == s]['Sector'].iloc[0] == sector
            ]
            
            if len(sector_symbols) >= 2:
                train_stocks.append(sector_symbols[0]) # 첫 번째 종목 -> 학습
                test_stocks.append(sector_symbols[1])  # 두 번째 종목 -> 테스트
                print(f"  Sector [{sector}]: Train={sector_symbols[0]}, Test={sector_symbols[1]}")
            elif len(sector_symbols) == 1:
                train_stocks.append(sector_symbols[0])
                print(f"  Sector [{sector}]: Train={sector_symbols[0]} (Only 1 survivor)")
            else:
                pass

        # 날짜 기준 분할
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        split_date = f"{train_end_year}-12-31"

        # 학습 데이터 추출
        train_df = self.df[
            (self.df['Symbol'].isin(train_stocks)) & 
            (self.df['Date'] <= split_date)
        ].copy()

        # 테스트 데이터 추출
        test_df = self.df[
            (self.df['Symbol'].isin(test_stocks)) & 
            (self.df['Date'] > split_date)
        ].copy()

        return train_df, test_df, train_stocks, test_stocks

    def save_datasets(self, train_df, test_df, output_dir="."):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        train_path = os.path.join(output_dir, "train_data.csv")
        test_path = os.path.join(output_dir, "test_data.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"\n💾 Saved Train Data: {train_path} ({len(train_df)} rows)")
        print(f"💾 Saved Test Data: {test_path} ({len(test_df)} rows)")
