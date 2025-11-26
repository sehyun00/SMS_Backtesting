#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Preprocessing for 5-Factor Stock Model - 10 Stocks, 10 Years
Converted from: data_preprocessing(5_19)이거쓸거임_ipynb의_사본.ipynb
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import warnings
import time
import os
from bs4 import BeautifulSoup
import requests
from sklearn.preprocessing import RobustScaler

# GPU 라이브러리 추가
try:
    import cupy as cp
    import cudf
    import cuml
    HAS_GPU = True
    print("GPU 라이브러리 로드 성공 - GPU 가속 사용 가능")
except ImportError:
    HAS_GPU = False
    print("GPU 라이브러리 사용 불가 - CPU 처리로 진행합니다")

# 경고 메시지 무시
warnings.filterwarnings('ignore')


class DailyStockFactorModel:
    def __init__(self):
        print("일별 5팩터 모델 데이터 처리 시작 (10년치)")
        self.start_time = time.time()

        # 현재 날짜 설정
        self.current_date = datetime.now()
        # 10년 전 날짜 계산
        self.ten_years_ago = self.current_date - relativedelta(years=10)

        # 종목 리스트
        self.stocks = []

        # 데이터 저장용 변수
        self.stock_data = {}
        self.daily_dates = []
        self.factor_model_data = pd.DataFrame()

        # 팩터 가중치 설정
        self.factor_weights = {
            'Beta_Factor': 0.20,
            'Value_Factor': 0.20,
            'Size_Factor': 0.20,
            'Momentum_Factor': 0.20,
            'Volatility_Factor': 0.20
        }

    def load_stocks_from_csv(self, csv_path='stock_list.csv'):
        """stock_list.csv 파일에서 10개 종목 정보를 가져옵니다"""
        print(f"\n{csv_path} 파일에서 종목 정보 가져오기...")
        
        try:
            # CSV 파일 로드
            df = pd.read_csv(csv_path, encoding='utf-8')
            
            # 종목 리스트로 변환
            self.stocks = df[['ticker', 'name', 'sector', 'industry']].to_dict('records')
            
            print(f"총 {len(self.stocks)}개 종목 로드 완료:")
            for stock in self.stocks:
                print(f"  - {stock['name']} ({stock['ticker']}): {stock['sector']} / {stock['industry']}")
            
            return self.stocks
            
        except Exception as e:
            print(f"CSV 파일 로드 실패: {e}")
            return []

    def get_trading_days(self, start_date, end_date, market='NYSE'):
        """특정 기간의 모든 거래일을 찾습니다"""
        try:
            # 미국 시장 캘린더 생성
            exchange = mcal.get_calendar(market)

            # 해당 기간의 거래일 가져오기
            trading_days = exchange.valid_days(start_date=start_date, end_date=end_date)

            # 날짜 객체로 변환
            trading_days = [day.date() for day in trading_days]

            return trading_days
        except Exception as e:
            print(f"거래일 정보 가져오기 실패: {e}")
            return []

    def generate_daily_dates(self, market='NYSE'):
        """지난 10년간의 모든 거래일 목록을 생성합니다"""
        start_date = self.ten_years_ago.strftime('%Y-%m-%d')
        end_date = self.current_date.strftime('%Y-%m-%d')

        trading_days = self.get_trading_days(start_date, end_date, market)

        print(f"{market} 시장의 지난 10년간 거래일 {len(trading_days)}개 찾음")

        # 일별 날짜 저장
        self.daily_dates = trading_days
        return trading_days

    def calculate_indicators_for_stock(self, symbol, name, daily_dates, market_index):
        """
        특정 종목의 일별 지표를 계산 (Rolling Beta + 벡터화)
        - 출력 컬럼은 기존 코드와 동일하게 맞춤 (Beta, Momentum1M, ... 등)
        """
        import yfinance as yf
        import pandas as pd
        import numpy as np
        from dateutil.relativedelta import relativedelta

        results = []

        try:
            # -----------------------
            # 1. 기간 설정 및 데이터 다운로드
            # -----------------------
            start_date = min(daily_dates) - relativedelta(months=15)
            end_date = max(daily_dates) + relativedelta(days=5)

            # 가격 데이터 (종목)
            hist_data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False,
                multi_level_index=False
            )

            if len(hist_data) < 30:
                print(f"{symbol}: 데이터가 충분하지 않습니다 (행 수: {len(hist_data)})")
                return []

            # 벤치마크 지수 (Beta 계산용)
            index_data = yf.download(
                market_index,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False,
                multi_level_index=False
            )

            # -----------------------
            # 2. 재무 정보 (PBR, 시총, 섹터/산업) - 날짜에 따라 고정값
            # -----------------------
            ticker_info = yf.Ticker(symbol)
            info = ticker_info.info

            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')

            # PBR
            if 'priceToBook' in info and info['priceToBook'] is not None and not pd.isna(info['priceToBook']):
                pbr = float(info['priceToBook'])
            else:
                pbr = 1.0

            # 시가총액 (10억 단위로 변환)
            if 'marketCap' in info and info['marketCap'] and not pd.isna(info['marketCap']):
                market_cap = info['marketCap'] / 1_000_000_000
            else:
                market_cap = 1.0  # 10억 달러 기본값

            # -----------------------
            # 3. Rolling Beta 계산
            # -----------------------
            # 수익률 계산
            hist_data['ret'] = hist_data['Adj Close'].pct_change()
            index_data['mkt_ret'] = index_data['Adj Close'].pct_change()

            merged = pd.merge(
                hist_data[['ret']],
                index_data[['mkt_ret']],
                left_index=True,
                right_index=True,
                how='inner'
            )

            rolling_cov = merged['ret'].rolling(window=252, min_periods=60).cov(merged['mkt_ret'])
            rolling_var = merged['mkt_ret'].rolling(window=252, min_periods=60).var()
            merged['beta'] = rolling_cov / rolling_var

            # 원본 hist_data에 베타 매핑
            hist_data['Beta'] = merged['beta'].reindex(hist_data.index)
            hist_data['Beta'] = hist_data['Beta'].fillna(1.0)  # 초기 구간은 1.0으로

            # -----------------------
            # 4. 모멘텀 (1M, 3M, 6M, 12M) - 일단위 근사 (21, 63, 126, 252 거래일)
            # -----------------------
            # 현재 Adj Close 기준
            adj = hist_data['Adj Close']

            hist_data['Momentum1M'] = (adj / adj.shift(21) - 1) * 100
            hist_data['Momentum3M'] = (adj / adj.shift(63) - 1) * 100
            hist_data['Momentum6M'] = (adj / adj.shift(126) - 1) * 100
            hist_data['Momentum12M'] = (adj / adj.shift(252) - 1) * 100

            # -----------------------
            # 5. 변동성 (30일 기준 연율화)
            # -----------------------
            daily_ret = adj.pct_change()
            hist_data['Volatility'] = daily_ret.rolling(30).std() * np.sqrt(252) * 100

            # -----------------------
            # 6. RSI (14일)
            # -----------------------
            delta = adj.diff()
            up = delta.where(delta > 0, 0.0)
            down = (-delta).where(delta < 0, 0.0)

            roll_up = up.rolling(window=14).mean()
            roll_down = down.rolling(window=14).mean()

            rs = roll_up / roll_down.replace(0, 0.001)
            rsi = 100 - (100 / (1 + rs))
            hist_data['RSI'] = rsi

            # -----------------------
            # 7. MACD (12, 26, 9)
            # -----------------------
            exp1 = adj.ewm(span=12, adjust=False).mean()
            exp2 = adj.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            hist = macd - signal

            hist_data['MACD'] = macd
            hist_data['Signal'] = signal
            hist_data['MACD_Hist'] = hist

            # -----------------------
            # 8. daily_dates 기준으로 캘린더 맞추기 (가장 가까운 이전 거래일 값 사용)
            # -----------------------
            # 히스토리 index를 Datetime으로 보장
            hist_data = hist_data.sort_index()
            hist_data.index = pd.to_datetime(hist_data.index)

            # 우리가 쓰려는 날짜 캘린더
            calendar = pd.to_datetime(daily_dates)
            calendar_df = pd.DataFrame(index=calendar)

            # 캘린더에 hist_data를 맞추되, 가장 가까운 과거 값으로 forward-fill
            feature_cols = [
                'Beta', 'Momentum1M', 'Momentum3M', 'Momentum6M', 'Momentum12M',
                'Volatility', 'RSI', 'MACD', 'Signal', 'MACD_Hist'
            ]
            feat = hist_data[feature_cols].reindex(calendar_df.index, method='ffill')

            # 완전히 NaN인 초기 구간 제거
            feat = feat.dropna(how='all')

            # -----------------------
            # 9. 결과 dict 리스트로 변환 (기존 키 그대로)
            # -----------------------
            for dt, row in feat.iterrows():
                results.append({
                    'Symbol': symbol,
                    'Name': name,
                    'Date': dt.strftime('%Y-%m-%d'),
                    'Beta': round(row['Beta'], 2) if not pd.isna(row['Beta']) else 1.0,
                    'PBR': round(pbr, 2),
                    'MarketCap': round(market_cap, 2),  # 이미 10억 단위
                    'Momentum1M': round(row['Momentum1M'], 2) if not pd.isna(row['Momentum1M']) else 0.0,
                    'Momentum3M': round(row['Momentum3M'], 2) if not pd.isna(row['Momentum3M']) else 0.0,
                    'Momentum6M': round(row['Momentum6M'], 2) if not pd.isna(row['Momentum6M']) else 0.0,
                    'Momentum12M': round(row['Momentum12M'], 2) if not pd.isna(row['Momentum12M']) else 0.0,
                    'Volatility': round(row['Volatility'], 2) if not pd.isna(row['Volatility']) else 0.0,
                    'RSI': round(row['RSI'], 2) if not pd.isna(row['RSI']) else 0.0,
                    'MACD': round(row['MACD'], 2) if not pd.isna(row['MACD']) else 0.0,
                    'Signal': round(row['Signal'], 2) if not pd.isna(row['Signal']) else 0.0,
                    'MACD_Hist': round(row['MACD_Hist'], 2) if not pd.isna(row['MACD_Hist']) else 0.0,
                    'Sector': sector,
                    'Industry': industry,
                    # 아래 값들은 이후 calculate_factor_scores에서 다시 채움
                    'Beta_Factor': 0,
                    'Value_Factor': 0,
                    'Size_Factor': 0,
                    'Momentum_Factor': 0,
                    'Volatility_Factor': 0,
                    'weighted_score': 0,
                    'factor_percentile': 0,
                    'smart_signal': 'NEUTRAL',
                    'signal_strength': 'MEDIUM',
                    'rebalance_priority': 0,
                    'to_rebalance': 0
                })

        except Exception as e:
            print(f"{symbol} 처리 중 오류 발생: {e}")

        return results

    def calculate_all_indicators(self):
        """모든 종목에 대한 일별 지표를 계산합니다"""
        all_results = []

        # 미국 주식 시장 일별 날짜 생성 (10년치)
        us_dates = self.generate_daily_dates('NYSE')

        # CSV에서 종목 목록 가져오기
        if not self.stocks:
            self.load_stocks_from_csv('stock_list.csv')

        if not self.stocks:
            print("처리할 종목이 없습니다!")
            return pd.DataFrame()

        # 10개 종목 처리
        print(f"\n{len(self.stocks)}개 종목의 10년치 데이터 처리 중...")
        for idx, stock in enumerate(self.stocks, 1):
            symbol = stock['ticker']
            name = stock['name']

            print(f"[{idx}/{len(self.stocks)}] {name} ({symbol}) 처리 중...")
            results = self.calculate_indicators_for_stock(symbol, name, us_dates, '^GSPC')
            all_results.extend(results)

        # 데이터프레임으로 변환
        self.factor_model_data = pd.DataFrame(all_results)

        return self.factor_model_data

    def calculate_factor_scores(self):
        """각 일별 날짜에 대한 팩터 점수를 계산합니다"""
        if len(self.factor_model_data) == 0:
            print("계산할 데이터가 없습니다")
            return

        print("\n팩터 점수 계산 중...")

        for date in self.factor_model_data['Date'].unique():
            date_df = self.factor_model_data[self.factor_model_data['Date'] == date].copy()

            if len(date_df) < 5:  # 충분한 종목이 없으면 건너뜀
                continue

            # 팩터 순위 계산 (퍼센타일로)
            date_df['Beta_Factor'] = -date_df['Beta'].rank(pct=True)  # 베타가 낮을수록 좋음
            date_df['Value_Factor'] = -date_df['PBR'].rank(pct=True)  # PBR이 낮을수록 좋음
            date_df['Size_Factor'] = -date_df['MarketCap'].rank(pct=True)  # 소형주가 선호됨
            date_df['Momentum_Factor'] = date_df['Momentum12M'].rank(pct=True)  # 모멘텀이 높을수록 좋음
            date_df['Volatility_Factor'] = -date_df['Volatility'].rank(pct=True)  # 변동성이 낮을수록 좋음

            # 가중 점수 계산
            date_df['weighted_score'] = (
                date_df['Beta_Factor'] * self.factor_weights['Beta_Factor'] +
                date_df['Value_Factor'] * self.factor_weights['Value_Factor'] +
                date_df['Size_Factor'] * self.factor_weights['Size_Factor'] +
                date_df['Momentum_Factor'] * self.factor_weights['Momentum_Factor'] +
                date_df['Volatility_Factor'] * self.factor_weights['Volatility_Factor']
            )

            # 가중 점수의 퍼센타일 계산
            date_df['factor_percentile'] = date_df['weighted_score'].rank(pct=True)

            # 신호 생성
            date_df['smart_signal'] = 'NEUTRAL'
            date_df.loc[date_df['factor_percentile'] > 0.7, 'smart_signal'] = 'BUY'
            date_df.loc[date_df['factor_percentile'] < 0.3, 'smart_signal'] = 'SELL'

            # 신호 강도
            date_df['signal_strength'] = 'MEDIUM'
            date_df.loc[date_df['factor_percentile'] > 0.9, 'signal_strength'] = 'STRONG'
            date_df.loc[date_df['factor_percentile'] < 0.1, 'signal_strength'] = 'STRONG'

            # 리밸런싱 우선순위
            date_df['rebalance_priority'] = date_df['factor_percentile'].rank(ascending=False)

            # 리밸런싱 플래그
            date_df['to_rebalance'] = 0
            date_df.loc[date_df['smart_signal'] != 'NEUTRAL', 'to_rebalance'] = 1

            # 메인 데이터프레임 업데이트
            for index, row in date_df.iterrows():
                mask = (self.factor_model_data['Symbol'] == row['Symbol']) & (self.factor_model_data['Date'] == date)
                for col in ['Beta_Factor', 'Value_Factor', 'Size_Factor', 'Momentum_Factor',
                           'Volatility_Factor', 'weighted_score', 'factor_percentile',
                           'smart_signal', 'signal_strength', 'rebalance_priority', 'to_rebalance']:
                    self.factor_model_data.loc[mask, col] = row[col]

        print("팩터 점수 계산 완료")
        return self.factor_model_data

    def save_data(self, output_dir='.'):
        """계산된 데이터를 저장합니다"""
        if len(self.factor_model_data) == 0:
            print("저장할 데이터가 없습니다")
            return None

        # 날짜와 티커로 정렬
        self.factor_model_data = self.factor_model_data.sort_values(['Date', 'Symbol'])

        # 컬럼 순서 재정렬
        cols = ['Symbol', 'Name', 'Date', 'Beta', 'PBR', 'MarketCap',
               'Momentum1M', 'Momentum3M', 'Momentum6M', 'Momentum12M',
               'Volatility', 'RSI', 'MACD', 'Signal', 'MACD_Hist',
               'Sector', 'Industry', 'Beta_Factor', 'Value_Factor',
               'Size_Factor', 'Momentum_Factor', 'Volatility_Factor',
               'weighted_score', 'factor_percentile', 'smart_signal',
               'signal_strength', 'rebalance_priority', 'to_rebalance']

        self.factor_model_data = self.factor_model_data[cols]

        # CSV 저장
        date_str = self.current_date.strftime('%Y%m%d')
        output_file = os.path.join(output_dir, f"processed_daily_5factor_model_10stocks_10years_{date_str}.csv")
        self.factor_model_data.to_csv(output_file, index=False)
        print(f"\n데이터가 {output_file}에 저장되었습니다")

        # 요약 정보 출력
        print(f"\n데이터 요약:")
        print(f"- 처리된 종목 수: {self.factor_model_data['Symbol'].nunique()}")
        print(f"- 처리된 일 수: {self.factor_model_data['Date'].nunique()}")
        print(f"- 총 행 수: {len(self.factor_model_data)}")

        # 매수/매도 신호 개수
        buy_count = len(self.factor_model_data[self.factor_model_data['smart_signal'] == 'BUY'])
        sell_count = len(self.factor_model_data[self.factor_model_data['smart_signal'] == 'SELL'])
        print(f"- 매수 신호 수: {buy_count}")
        print(f"- 매도 신호 수: {sell_count}")

        return output_file

    def remove_duplicates(self):
        """같은 날짜와 같은 주식 코드의 중복 데이터를 제거합니다"""
        if len(self.factor_model_data) == 0:
            print("제거할 데이터가 없습니다")
            return

        # 중복 제거 전 데이터 개수
        before_count = len(self.factor_model_data)

        # 같은 날짜와 같은 주식 코드(Symbol) 중복 제거
        self.factor_model_data = self.factor_model_data.drop_duplicates(
            subset=['Date', 'Symbol'],
            keep='first'
        ).reset_index(drop=True)

        # 중복 제거 후 데이터 개수
        after_count = len(self.factor_model_data)
        removed_count = before_count - after_count

        if removed_count > 0:
            print(f"중복 제거 완료: {removed_count}개 행이 제거되었습니다")
            print(f"제거 전: {before_count}개 → 제거 후: {after_count}개")
        else:
            print("중복된 데이터가 없습니다")

        return self.factor_model_data

    def run_pipeline(self, csv_path='stock_list.csv', output_dir='.'):
        """전체 데이터 파이프라인을 실행합니다"""
        print(f"시작 시간: {self.current_date.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"데이터 기간: {self.ten_years_ago.strftime('%Y-%m-%d')} ~ {self.current_date.strftime('%Y-%m-%d')}")

        # CSV에서 종목 목록 가져오기
        self.load_stocks_from_csv(csv_path)

        # 일별 지표 계산 (10년치)
        self.calculate_all_indicators()

        # 중복 제거
        self.remove_duplicates()

        # 팩터 점수 계산
        self.calculate_factor_scores()

        # 데이터 저장
        self.save_data(output_dir)

        # 총 실행 시간 출력
        elapsed_time = time.time() - self.start_time
        print(f"\n전체 처리 완료! 총 실행 시간: {elapsed_time:.2f}초")

        return self.factor_model_data


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='5-Factor Stock Model - 10 Stocks, 10 Years')
    parser.add_argument('--csv', type=str, default='stock_list.csv',
                       help='Path to stock list CSV file')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for processed data')
    
    args = parser.parse_args()
    
    # 모델 실행
    model = DailyStockFactorModel()
    result = model.run_pipeline(
        csv_path=args.csv,
        output_dir=args.output_dir
    )
    
    return result


if __name__ == "__main__":
    main()
