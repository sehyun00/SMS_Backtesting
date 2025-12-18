import pandas as pd
import numpy as np

class TechnicalIndicators:
    """
    주가 데이터(DataFrame)에 기술적 지표를 추가하는 클래스
    """
    
    @staticmethod
    def add_all_indicators(df):
        """모든 기술적 지표를 한 번에 계산하여 추가합니다."""
        df = df.copy()
        
        # 1. 이동평균 및 모멘텀
        df = TechnicalIndicators.add_momentum(df)
        
        # 2. 변동성
        df = TechnicalIndicators.add_volatility(df)
        
        # 3. RSI
        df = TechnicalIndicators.add_rsi(df)
        
        # 4. MACD
        df = TechnicalIndicators.add_macd(df)
        
        # 5. Beta, PBR, MarketCap 등은 외부 데이터가 필요하므로 
        #    여기서는 순수 가격 기반 지표만 계산하거나 
        #    별도 메서드로 분리하는 것이 좋습니다.
        #    (기존 코드에 있던 MarketCap 등은 Volume * Close로 추정 가능하지만
        #     정확한 시가총액/PBR은 재무 데이터가 필요함. 일단 기존 로직 유지)Momentum1M 
        return df

    @staticmethod
    def add_momentum(df):
        # 1개월(20일), 3개월(60일), 6개월(120일), 12개월(252일) 모멘텀
        df['Momentum1M'] = df['Close'].pct_change(periods=20)
        df['Momentum3M'] = df['Close'].pct_change(periods=60)
        df['Momentum6M'] = df['Close'].pct_change(periods=120)
        df['Momentum12M'] = df['Close'].pct_change(periods=252)
        return df

    @staticmethod
    def add_volatility(df):
        # 20일 기준 연환산 변동성
        df['Volatility'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        return df

    @staticmethod
    def add_rsi(df, window=14):
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 결측치 처리 (초기 데이터)
        df['RSI'] = df['RSI'].fillna(50)
        return df

    @staticmethod
    def add_macd(df, short=12, long=26, signal=9):
        short_ema = df['Close'].ewm(span=short, adjust=False).mean()
        long_ema = df['Close'].ewm(span=long, adjust=False).mean()
        
        df['MACD'] = short_ema - long_ema
        df['Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']
        return df