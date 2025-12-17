# 팩터 점수 계산

import pandas as pd
import numpy as np

class FactorCalculator:
    """
    기술적 지표를 바탕으로 팩터 스코어링 및 시그널을 생성하는 클래스
    """

    @staticmethod
    def calculate_factors(df):
        """
        각 팩터별 백분위(Percentile) 점수를 계산합니다.
        주의: 이 함수는 단일 종목이 아닌, '전체 종목이 합쳐진 DataFrame'에서 
        날짜별로 그룹화하여 수행하는 것이 가장 정확하지만, 
        여기서는 단일 종목 시계열 흐름 내에서의 상대적 위치를 평가하거나
        전체 데이터 병합 후 호출해야 합니다.
        
        여기서는 개별 종목 데이터 처리를 가정하고 단순화된 로직을 적용합니다.
        """
        df = df.copy()

        # 1. Beta Factor (변동성이 낮을수록 높은 점수라고 가정하거나, 시장 민감도 반영)
        # Beta 값이 없으므로 임시로 Volatility 역수를 사용하거나, 
        # 추후 시장 지수와 비교하여 계산해야 함. 여기서는 예시로 Volatility 역수 사용
        if 'Volatility' in df.columns:
            df['Beta_Factor'] = df['Volatility'].rank(pct=True, ascending=False)
        else:
            df['Beta_Factor'] = 0.5

        # 2. Value Factor (PBR, PER 등이 필요하지만 기술적 지표만 있다면 생략 가능)
        # 예시로 RSI가 낮을수록(과매도) Value 점수를 높게 부여
        df['Value_Factor'] = df['RSI'].rank(pct=True, ascending=False)

        # 3. Momentum Factor (모멘텀이 강할수록 높은 점수)
        if 'Momentum12M' in df.columns:
            df['Momentum_Factor'] = df['Momentum12M'].rank(pct=True)
        else:
            df['Momentum_Factor'] = 0.5

        # 4. Volatility Factor (저변동성 선호 -> 변동성 낮을수록 고득점)
        if 'Volatility' in df.columns:
            df['Volatility_Factor'] = df['Volatility'].rank(pct=True, ascending=False)
        else:
            df['Volatility_Factor'] = 0.5

        return df

    @staticmethod
    def calculate_weighted_score(df):
        """
        팩터별 가중치를 적용하여 최종 점수와 시그널을 생성합니다.
        """
        df = df.copy()
        
        # 가중치 설정 (사용자 전략에 따라 변경 가능)
        weights = {
            'Value_Factor': 0.3,
            'Momentum_Factor': 0.3,
            'Volatility_Factor': 0.2,
            'Beta_Factor': 0.2
        }
        
        # 가중 평균 점수 계산
        df['weighted_score'] = (
            df['Value_Factor'] * weights['Value_Factor'] +
            df['Momentum_Factor'] * weights['Momentum_Factor'] +
            df['Volatility_Factor'] * weights['Volatility_Factor'] +
            df['Beta_Factor'] * weights['Beta_Factor']
        ) * 100  # 100점 만점 환산

        # 시그널 생성
        conditions = [
            (df['weighted_score'] >= 80),
            (df['weighted_score'] >= 60),
            (df['weighted_score'] <= 40),
            (df['weighted_score'] <= 20)
        ]
        choices = ['STRONG_BUY', 'BUY', 'SELL', 'STRONG_SELL']
        
        df['smart_signal'] = np.select(conditions, choices, default='NEUTRAL')
        
        return df
