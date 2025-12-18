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
<<<<<<< HEAD
    
        # 컬럼 이름 매핑 (Symbol이나 Ticker 혼용 대응)
        if 'Ticker' in df.columns:
            df = df.rename(columns={'Ticker': 'Symbol'})
    
        # 🔥 industry (소문자) -> Industry (대문자) 변환
        if 'industry' in df.columns and 'Industry' not in df.columns:
            df = df.rename(columns={'industry': 'Industry'})
    
        # 🔥 Industry 컬럼이 없으면 'Unknown'으로 채우기
        if 'Industry' not in df.columns:
            df['Industry'] = 'Unknown'
    
        required_cols = ['Symbol', 'Sector']
=======

        # 컬럼 이름 매핑 (Symbol이나 Ticker 혼용 대응)
        if "Ticker" in df.columns:
            df = df.rename(columns={"Ticker": "Symbol"})

        required_cols = ["Symbol", "Sector"]
>>>>>>> feat/models/Hybrid_TGNN_DDPG_refactor
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")

        self.stocks = df.to_dict("records")
        print(f"✅ Loaded {len(self.stocks)} stocks from {file_path}")
        return self.stocks

    def filter_survivor_stocks(self, start_year=2006, end_year=2025):
        """
        🔥 수정: Train/Test 종목을 별도로 필터링

        - Train용: 2006-2020 기간에 데이터가 있는 종목 (생존 편향 제거)
        - Test용: 2021-2025 기간까지 생존한 종목 (실전 투자 가능)

        Returns:
            dict: {'train': [stock_info, ...], 'test': [stock_info, ...]}
        """
        print(f"\n🔍 Filtering stocks for Train & Test periods...")

        survivors = {
            "train": [],  # Train용 종목
            "test": [],  # Test용 종목
        }

        train_end = 2020
        test_start = 2021

        for stock in self.stocks:
            symbol = stock["Symbol"]

            try:
                # 전체 기간 데이터 다운로드
                hist = yf.download(
                    symbol,
                    start=f"{start_year}-01-01",
                    end=f"{end_year}-12-31",
                    progress=False,
                    auto_adjust=True,
                )

                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.droplevel(1)

                if hist.empty:
                    continue

                # Train 기간 데이터 확인 (2006-2020)
                train_data = hist[hist.index.year <= train_end]
                if len(train_data) >= 5 * 200:  # 최소 5년 데이터 (연간 약 200거래일)
                    survivors["train"].append(stock)

                # Test 기간 데이터 확인 (2021-2025)
                test_data = hist[hist.index.year >= test_start]
                if len(test_data) >= 3 * 200:  # 최소 3년 데이터
                    survivors["test"].append(stock)

            except Exception as e:
                print(f"  ❌ {symbol}: Error ({e})")

        print(f"   ✅ Train 후보: {len(survivors['train'])}개 종목")
        print(f"   ✅ Test 후보: {len(survivors['test'])}개 종목")

        return survivors

    def fetch_daily_data(self, symbol, start_date, end_date):
        """
        특정 종목의 일별 OHLCV 데이터를 가져옵니다.
        """
        try:
<<<<<<< HEAD
            df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
            
=======
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)

>>>>>>> feat/models/Hybrid_TGNN_DDPG_refactor
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            if df.empty:
                return None

            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df["Symbol"] = symbol
            return df

        except Exception as e:
            print(f"Failed to fetch data for {symbol}: {e}")
            return None
