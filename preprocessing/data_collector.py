# 데이터 수집 전담

import pandas as pd
import yfinance as yf
from datetime import datetime
import os

class DataCollector:
    def __init__(self):
        self.stocks = []
        
    def load_stocks_from_csv(self, file_path):
        """
        CSV 파일에서 종목 리스트를 로드합니다.
        필수 컬럼: Symbol, Sector, Industry (Name은 선택)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        df = pd.read_csv(file_path)
        
        # 컬럼 이름 매핑 (Symbol이나 Ticker 혼용 대응)
        if 'Ticker' in df.columns:
            df = df.rename(columns={'Ticker': 'Symbol'})
        
        required_cols = ['Symbol', 'Sector']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")

        self.stocks = df.to_dict('records')
        print(f"✅ Loaded {len(self.stocks)} stocks from {file_path}")
        return self.stocks

    def filter_survivor_stocks(self, start_year=2006, end_year=2025):
        """
        지정된 기간(start_year ~ end_year) 동안 계속 상장되어 있던 종목만 필터링합니다.
        TGNN 학습을 위해 노드 수를 고정하고 데이터 연속성을 확보하기 위함입니다.
        """
        print(f"\n🔍 Filtering survivors from {start_year} to {end_year}...")
        
        valid_stocks = []
        start_date_str = f"{start_year}-01-01"
        end_date_str = f"{end_year}-12-31"

        for stock in self.stocks:
            symbol = stock['Symbol']
            try:
                hist = yf.download(
                    symbol, 
                    start=start_date_str, 
                    end=end_date_str, 
                    progress=False, 
                    auto_adjust=True
                )
                
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.droplevel(1)

                if len(hist) > 0:
                    first_date = hist.index[0]
                    # 데이터 시작일이 기준년도(2006) 이전이거나 같아야 함
                    if first_date.year <= start_year:
                        valid_stocks.append(stock)
                    else:
                        pass
                else:
                    print(f"  ❌ {symbol}: No data found")
            
            except Exception as e:
                print(f"  ❌ {symbol}: Error ({e})")

        self.stocks = valid_stocks
        print(f"✅ Filtered: {len(valid_stocks)} survivor stocks remaining.")
        return valid_stocks

    def fetch_daily_data(self, symbol, start_date, end_date):
        """
        특정 종목의 일별 OHLCV 데이터를 가져옵니다.
        """
        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            if df.empty:
                return None
                
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df['Symbol'] = symbol
            return df
            
        except Exception as e:
            print(f"Failed to fetch data for {symbol}: {e}")
            return None