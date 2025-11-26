#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Preprocessing for 5-Factor Stock Model
Converted from: data_preprocessing(11/26)한글주식제거_ipynb의_수정본.ipynb
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
        print("일별 5팩터 모델 데이터 처리 시작")
        self.start_time = time.time()

        # 현재 날짜 설정
        self.current_date = datetime.now()
        # 1년 전 날짜 계산
        self.one_year_ago = self.current_date - relativedelta(years=10)

        # 미국 대표 주식 리스트
        self.us_stocks = []

        # 데이터 저장용 변수
        self.stock_data = {}  # stock_data 속성 추가
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

    def get_trading_days(self, start_date, end_date):
        """특정 기간의 모든 거래일을 찾습니다"""
        try:
            # 시장의 캘린더 생성
            exchange = mcal.get_calendar('NYSE')

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
        start_date = self.one_year_ago.strftime('%Y-%m-%d')
        end_date = self.current_date.strftime('%Y-%m-%d')

        trading_days = self.get_trading_days(start_date, end_date, market)

        print(f"{market} 시장의 지난 10년간 거래일 {len(trading_days)}개 찾음")

        # 일별 날짜 저장
        self.daily_dates = trading_days
        return trading_days

    def get_us_stocks(self, csv_path='US_Stock_Master.csv'):
        print("\n미국 대표 주식 가져오기...")
        # CSV 파일 로드
        df = pd.read_csv(csv_path, dtype={'ACT Symbol': str})

        # 결측값 제거 (심볼이나 이름이 없는 행 제외)
        df = df[['ACT Symbol', 'Company Name']].dropna(subset=['ACT Symbol', 'Company Name'])

        # 컬럼명 통일 
        df = df.rename(columns={'ACT Symbol': 'symbol', 'Company Name': 'name'})

        # 딕셔너리 리스트로 변환
        self.us_stocks = df.to_dict('records')

        print(f"미국 주식 {len(self.us_stocks)}개 로드 완료")
        return self.us_stocks

    def calculate_indicators_for_stock(self, symbol, name, daily_dates, market_index):
        """특정 종목의 일별 지표를 계산합니다"""
        results = []

        try:
            # 데이터 다운로드 기간 설정 (충분한 데이터를 위해 여유 있게 설정)
            start_date = min(daily_dates) - relativedelta(months=15)
            end_date = max(daily_dates) + relativedelta(days=5)

            # 히스토리 데이터 다운로드
            hist_data = yf.download(symbol, start=start_date, end=end_date, progress=False,
                                   auto_adjust=False, multi_level_index=False)

            time.sleep(1)

            if len(hist_data) < 30:
                print(f"{symbol}: 데이터가 충분하지 않습니다 (행 수: {len(hist_data)})")
                return []

            # 시장 지수 데이터 다운로드
            index_data = yf.download(market_index, start=start_date, end=end_date, progress=False,
                                    auto_adjust=False, multi_level_index=False)

            # 재무 정보 가져오기
            ticker_info = yf.Ticker(symbol)
            info = ticker_info.info

            # 섹터, 산업 정보
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')

            # 베타값 가져오기
            if 'beta' in info and info['beta'] is not None and not pd.isna(info['beta']):
                beta = float(info['beta'])
                beta = min(max(beta, -2.0), 4.0)  # 이상치 방지
            else:
                # Yahoo Finance API에서 베타값을 찾을 수 없는 경우   
                beta = self.default_beta_values.get(symbol, 1.0)
                print(f"{symbol}: 베타값 없음, 기본값 {beta} 사용")

            # PBR 값
            if 'priceToBook' in info and info['priceToBook'] is not None and not pd.isna(info['priceToBook']):
                pbr = float(info['priceToBook'])
                print(f"{symbol}: Yahoo Finance에서 PBR값 {pbr:.4f} 가져옴")
            else:
                pbr = 1.0
                print(f"{symbol}: PBR 정보 없음, 기본값 1.0 사용")

            # 시가총액 처리
            if 'marketCap' in info and info['marketCap'] and not pd.isna(info['marketCap']):
                market_cap = info['marketCap']
                print(f"{symbol}: 시가총액 {market_cap:,.0f} USD")
            else:
                market_cap = 1000000000  # 기본값: 10억 USD

            # 각 일별 날짜에 대한 지표 계산
            for target_date in daily_dates:
                # 해당일 또는 그 이전 가장 가까운 거래일 찾기
                available_dates = hist_data.index[hist_data.index <= pd.Timestamp(target_date)]
                if len(available_dates) == 0:
                    print(f"{symbol}: {target_date.strftime('%Y-%m-%d')}에 해당하는 데이터 없음")
                    continue

                closest_date = available_dates.max()
                date_str = closest_date.strftime('%Y-%m-%d')

                # 해당 날짜까지의 데이터 추출
                data_until_date = hist_data.loc[:closest_date]

                # 모멘텀 계산
                current_price = data_until_date['Adj Close'][-1]

                # 1개월 모멘텀
                one_month_ago = closest_date - relativedelta(months=1)
                one_month_prices = data_until_date[data_until_date.index <= one_month_ago]
                momentum_1m = ((current_price / one_month_prices['Adj Close'][-1]) - 1) * 100 if len(one_month_prices) > 0 else 0

                # 3개월 모멘텀
                three_months_ago = closest_date - relativedelta(months=3)
                three_month_prices = data_until_date[data_until_date.index <= three_months_ago]
                momentum_3m = ((current_price / three_month_prices['Adj Close'][-1]) - 1) * 100 if len(three_month_prices) > 0 else 0

                # 6개월 모멘텀
                six_months_ago = closest_date - relativedelta(months=6)
                six_month_prices = data_until_date[data_until_date.index <= six_months_ago]
                momentum_6m = ((current_price / six_month_prices['Adj Close'][-1]) - 1) * 100 if len(six_month_prices) > 0 else 0

                # 12개월 모멘텀
                twelve_months_ago = closest_date - relativedelta(months=12)
                twelve_month_prices = data_until_date[data_until_date.index <= twelve_months_ago]
                momentum_12m = ((current_price / twelve_month_prices['Adj Close'][-1]) - 1) * 100 if len(twelve_month_prices) > 0 else 0

                # 변동성 계산
                returns = data_until_date['Adj Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 30 else 0  # 연간화, 퍼센트 변환

                # RSI 계산
                delta = data_until_date['Adj Close'].diff().dropna()
                up = delta.copy()
                up[up < 0] = 0
                down = -delta.copy()
                down[down < 0] = 0

                avg_gain = up.rolling(window=14).mean()
                avg_loss = down.rolling(window=14).mean()

                rs = avg_gain / avg_loss.replace(0, 0.001)  # 0으로 나누기 방지
                rsi = 100 - (100 / (1 + rs))
                rsi_value = rsi.iloc[-1]

                # MACD 계산
                exp1 = data_until_date['Adj Close'].ewm(span=12, adjust=False).mean()
                exp2 = data_until_date['Adj Close'].ewm(span=26, adjust=False).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9, adjust=False).mean()
                hist = macd - signal

                macd_value = macd.iloc[-1]
                signal_value = signal.iloc[-1]
                hist_value = hist.iloc[-1]

                # 결과 저장
                results.append({
                    'Symbol': symbol,
                    'Name': name,
                    'Date': date_str,
                    'Beta': round(beta, 2),
                    'PBR': round(pbr, 2),
                    'MarketCap': round(market_cap / 1_000_000_000, 2),  # 10억 단위로 변환
                    'Momentum1M': round(momentum_1m, 2),
                    'Momentum3M': round(momentum_3m, 2),
                    'Momentum6M': round(momentum_6m, 2),
                    'Momentum12M': round(momentum_12m, 2),
                    'Volatility': round(volatility, 2),
                    'RSI': round(rsi_value, 2),
                    'MACD': round(macd_value, 2),
                    'Signal': round(signal_value, 2),
                    'MACD_Hist': round(hist_value, 2),
                    'Sector': sector,
                    'Industry': industry,
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

        # 미국 주식 시장 일별 날짜 생성
        us_dates = self.generate_daily_dates('NYSE')

        if not self.us_stocks:
            self.get_us_stocks()

        # 미국 주식 처리
        print("\n미국 주식 데이터 처리 중...")
        for idx, stock in enumerate(self.us_stocks, 1):
            symbol = stock['symbol']
            name = stock['name']

            print(f"[{idx}/{len(self.us_stocks)}] {name} ({symbol}) 처리 중...")
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
                print(f"{date} 날짜에 충분한 종목 데이터가 없어 팩터 점수 계산을 건너뜁니다")
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
        output_file = os.path.join(output_dir, f"processed_daily_5factor_model_{date_str}.csv")
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

    def check_and_remove_duplicates(self):
        """중복 데이터를 확인하고 제거합니다"""
        # 중복 확인
        duplicates = self.factor_model_data.duplicated(subset=['Date', 'Symbol'], keep=False)

        if duplicates.any():
            print("\n발견된 중복 데이터:")
            duplicate_data = self.factor_model_data[duplicates].sort_values(['Date', 'Symbol'])
            print(duplicate_data[['Symbol', 'Name', 'Date']].to_string())

            # 중복 제거
            self.remove_duplicates()
        else:
            print("중복된 데이터가 없습니다")

    def run_pipeline(self, us_csv_path='US_Stock_Master.csv', output_dir='.'):
        """전체 데이터 파이프라인을 실행합니다"""
        print(f"시작 시간: {self.current_date.strftime('%Y-%m-%d %H:%M:%S')}")

        # 주식 목록 가져오기
        self.get_us_stocks(us_csv_path)

        # 일별 지표 계산
        self.calculate_all_indicators()

        # 중복 제거 (팩터 점수 계산 전에 실행)
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
    
    parser = argparse.ArgumentParser(description='5-Factor Stock Model Data Preprocessing')
    parser.add_argument('--us-csv', type=str, default='US_Stock_Master.csv',
                       help='Path to US stock master CSV file')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for processed data')
    
    args = parser.parse_args()
    
    # 모델 실행
    model = DailyStockFactorModel()
    result = model.run_pipeline(
        us_csv_path=args.us_csv,
        output_dir=args.output_dir
    )
    
    return result


if __name__ == "__main__":
    main()
