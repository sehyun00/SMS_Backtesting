# Fama-French 전담

import pandas as pd
import pandas_datareader.data as web
from datetime import datetime

class FamaFrenchLoader:
    def __init__(self):
        self.ff_data = None

    def download_factors(self, start_date='2006-01-01', end_date='2025-12-31'):
        """
        Fama-French 5 Factor 데이터를 다운로드합니다.
        """
        print("📥 Downloading Fama-French 5 Factors...")
        try:
            # Kenneth French Data Library
            ds = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench', start=start_date, end=end_date)
            
            # 0번 인덱스가 일별 데이터 (보통 딕셔너리 형태로 반환됨)
            self.ff_data = ds[0].copy()
            
            # 백분율 단위를 소수로 변환 (예: 0.5% -> 0.005)
            self.ff_data = self.ff_data / 100.0
            
            # 인덱스 이름 변경 및 Timezone 제거
            self.ff_data.index.name = 'Date'
            if self.ff_data.index.tz is not None:
                self.ff_data.index = self.ff_data.index.tz_localize(None)
                
            # 컬럼 이름 변경 (Mkt-RF -> Mkt_RF)
            self.ff_data.columns = [col.replace('-', '_').replace(' ', '') for col in self.ff_data.columns]
            
            print(f"✅ Fama-French data loaded: {len(self.ff_data)} rows")
            return self.ff_data
            
        except Exception as e:
            print(f"❌ Failed to download Fama-French data: {e}")
            return None

    def merge_with_stock_data(self, stock_df):
        """
        주식 데이터와 Fama-French 데이터를 날짜 기준으로 병합합니다.
        """
        if self.ff_data is None:
            print("⚠️ Fama-French data is missing. Skipping merge.")
            return stock_df

        # 날짜 형식 통일
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        if stock_df['Date'].dt.tz is not None:
            stock_df['Date'] = stock_df['Date'].dt.tz_localize(None)
            
        # 병합 (Left Join)
        merged_df = pd.merge(stock_df, self.ff_data, on='Date', how='left')
        
        # 결측치 처리 (휴일 등 FF 데이터가 없는 경우 직전 값 사용)
        cols_to_fill = self.ff_data.columns
        merged_df[cols_to_fill] = merged_df[cols_to_fill].fillna(method='ffill')
        
        return merged_df
