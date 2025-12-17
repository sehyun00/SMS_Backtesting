#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nasdaq-100 종목 리스트 자동 생성 스크립트
- Wikipedia에서 나스닥 100 구성 종목 크롤링
- 섹터/산업 정보 포함
- CSV 파일로 저장
"""

import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import time
from pathlib import Path


class Nasdaq100ListCreator:
    """나스닥 100 종목 리스트 생성기"""
    
    def __init__(self):
        self.stocks = []
        print("Nasdaq-100 종목 리스트 생성기 초기화")
    
    def fetch_from_wikipedia(self):
        """
        Wikipedia에서 나스닥 100 구성 종목 가져오기 (헤더 추가 수정됨)
        """
        print("\n📡 Wikipedia에서 나스닥 100 종목 크롤링 중...")
        
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        
        try:
            # 💡 [핵심 수정] 403 에러 방지를 위한 User-Agent 헤더 추가
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            # requests로 HTML 먼저 가져오기
            response = requests.get(url, headers=headers)
            response.raise_for_status() # 200 OK 아니면 에러 발생
            
            # HTML 텍스트를 pandas로 파싱
            tables = pd.read_html(response.text)
            
            # 구성 종목 테이블 찾기
            nasdaq_df = None
            for table in tables:
                # 'Ticker' 또는 'Symbol' 컬럼이 있는 테이블 찾기
                if 'Ticker' in table.columns or 'Symbol' in table.columns:
                    nasdaq_df = table
                    break
            
            if nasdaq_df is None:
                raise ValueError("나스닥 100 테이블을 찾을 수 없습니다")
            
            # 컬럼명 표준화
            nasdaq_df.columns = nasdaq_df.columns.str.strip()
            
            # 필요한 컬럼만 선택
            if 'Ticker' in nasdaq_df.columns:
                ticker_col = 'Ticker'
            elif 'Symbol' in nasdaq_df.columns:
                ticker_col = 'Symbol'
            else:
                raise ValueError("티커 컬럼을 찾을 수 없습니다")
            
            # 컬럼 매핑
            column_mapping = {
                ticker_col: 'ticker',
                'Company': 'name',
                'GICS Sector': 'sector',
                'GICS Sub-Industry': 'industry'
            }
            
            # 존재하는 컬럼만 선택
            available_cols = [col for col in column_mapping.keys() if col in nasdaq_df.columns]
            nasdaq_df = nasdaq_df[available_cols].copy()
            nasdaq_df = nasdaq_df.rename(columns=column_mapping)
            
            # 결측치 제거
            nasdaq_df = nasdaq_df.dropna(subset=['ticker'])
            
            # 티커 정리 (공백 제거)
            nasdaq_df['ticker'] = nasdaq_df['ticker'].str.strip()
            
            print(f"✅ {len(nasdaq_df)}개 종목 크롤링 완료")
            
            return nasdaq_df
            
        except Exception as e:
            print(f"❌ Wikipedia 크롤링 실패: {e}")
            return pd.DataFrame()
    
    def enrich_with_yfinance(self, df):
        """
        yfinance로 추가 정보 보완 (섹터/산업이 없는 경우)
        
        Args:
            df (pd.DataFrame): 기본 종목 정보
            
        Returns:
            pd.DataFrame: 보완된 종목 정보
        """
        print("\n🔍 yfinance로 종목 정보 보완 중...")
        
        enriched_data = []
        
        for idx, row in df.iterrows():
            ticker = row['ticker']
            
            try:
                # yfinance로 정보 조회
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # 기존 데이터 우선, 없으면 yfinance 사용
                name = row.get('name', info.get('longName', ticker))
                sector = row.get('sector', info.get('sector', 'Unknown'))
                industry = row.get('industry', info.get('industry', 'Unknown'))
                
                enriched_data.append({
                    'ticker': ticker,
                    'name': name,
                    'sector': sector,
                    'industry': industry
                })
                
                print(f"  ✅ {ticker}: {name}")
                
                # API 제한 방지
                time.sleep(0.1)
                
            except Exception as e:
                print(f"  ⚠️  {ticker}: 정보 조회 실패 - {e}")
                
                # 실패 시 기존 데이터 사용
                enriched_data.append({
                    'ticker': ticker,
                    'name': row.get('name', ticker),
                    'sector': row.get('sector', 'Unknown'),
                    'industry': row.get('industry', 'Unknown')
                })
        
        result_df = pd.DataFrame(enriched_data)
        print(f"\n✅ {len(result_df)}개 종목 정보 보완 완료")
        
        return result_df
    
    def filter_by_sector_diversity(self, df, target_count=None, min_per_sector=1):
        """
        섹터 다양성을 고려하여 종목 선택
        
        Args:
            df (pd.DataFrame): 전체 종목 리스트
            target_count (int): 목표 종목 수 (None이면 전체)
            min_per_sector (int): 섹터당 최소 종목 수
            
        Returns:
            pd.DataFrame: 필터링된 종목 리스트
        """
        if target_count is None:
            return df
        
        print(f"\n🎯 섹터별 균형을 고려하여 {target_count}개 종목 선택 중...")
        
        # 섹터별 그룹화
        sector_groups = df.groupby('sector')
        
        # 섹터 수
        num_sectors = len(sector_groups)
        stocks_per_sector = max(min_per_sector, target_count // num_sectors)
        
        selected = []
        
        for sector, group in sector_groups:
            # 각 섹터에서 상위 n개 선택 (시가총액 기준이면 좋지만, 여기서는 순서대로)
            sector_stocks = group.head(stocks_per_sector)
            selected.append(sector_stocks)
            
            print(f"  {sector}: {len(sector_stocks)}개 선택")
        
        result = pd.concat(selected, ignore_index=True)
        
        # 목표 개수 초과 시 자르기
        if len(result) > target_count:
            result = result.head(target_count)
        
        print(f"\n✅ 총 {len(result)}개 종목 선택 완료")
        
        return result
    
    def save_to_csv(self, df, output_path="nasdaq100_stock_list.csv"):
        """
        CSV 파일로 저장
        
        Args:
            df (pd.DataFrame): 종목 데이터
            output_path (str): 저장 경로
        """
        # 경로를 Path 객체로 변환
        output_path = Path(output_path)
        
        # 디렉토리 생성
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # CSV 저장
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"\n💾 {output_path}에 저장 완료")
        print(f"   총 {len(df)}개 종목")
        print(f"   섹터 수: {df['sector'].nunique()}")
        
        # 섹터별 분포 출력
        print("\n📊 섹터별 분포:")
        sector_counts = df['sector'].value_counts()
        for sector, count in sector_counts.items():
            print(f"   - {sector}: {count}개")
        
        return output_path
    
    def create_nasdaq100_list(self, output_path="data/nasdaq100_stock_list.csv", 
                               target_count=None, enrich=True):
        """
        나스닥 100 종목 리스트 생성 전체 파이프라인
        
        Args:
            output_path (str): 저장 경로
            target_count (int): 목표 종목 수 (None이면 전체 100개)
            enrich (bool): yfinance로 정보 보완 여부
            
        Returns:
            pd.DataFrame: 생성된 종목 리스트
        """
        print("=" * 60)
        print("Nasdaq-100 종목 리스트 생성 시작")
        print("=" * 60)
        
        # 1. Wikipedia에서 크롤링
        df = self.fetch_from_wikipedia()
        
        if df.empty:
            print("❌ 종목 리스트 생성 실패")
            return pd.DataFrame()
        
        # 2. yfinance로 정보 보완 (선택)
        if enrich:
            df = self.enrich_with_yfinance(df)
        
        # 3. 섹터 다양성 고려 필터링 (선택)
        if target_count:
            df = self.filter_by_sector_diversity(df, target_count=target_count)
        
        # 4. CSV 저장
        self.save_to_csv(df, output_path)
        
        print("\n✅ 나스닥 100 종목 리스트 생성 완료!")
        
        return df


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="나스닥 100 종목 리스트 생성")
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/nasdaq100_stock_list.csv',
        help='출력 CSV 파일 경로'
    )
    parser.add_argument(
        '--count', '-c',
        type=int,
        default=None,
        help='선택할 종목 수 (기본: 전체 100개)'
    )
    parser.add_argument(
        '--no-enrich',
        action='store_true',
        help='yfinance 정보 보완 생략'
    )
    
    args = parser.parse_args()
    
    # 리스트 생성
    creator = Nasdaq100ListCreator()
    df = creator.create_nasdaq100_list(
        output_path=args.output,
        target_count=args.count,
        enrich=not args.no_enrich
    )
    
    # 미리보기
    if not df.empty:
        print("\n" + "=" * 60)
        print("생성된 종목 리스트 미리보기 (처음 10개):")
        print("=" * 60)
        print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
