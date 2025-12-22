import pandas as pd
import numpy as np

# 데이터 로드
print("=" * 60)
print("📊 데이터 분석")
print("=" * 60)

train_df = pd.read_csv('data/train_data.csv')
test_df = pd.read_csv('data/test_data.csv')

print("\n=== Train Data 기간 ===")
print(f"시작: {train_df['Date'].min()}")
print(f"종료: {train_df['Date'].max()}")
print(f"총 행: {len(train_df):,}")

print("\n=== Test Data 기간 ===")
print(f"시작: {test_df['Date'].min()}")
print(f"종료: {test_df['Date'].max()}")
print(f"총 행: {len(test_df):,}")

print("\n=== Momentum 통계 (Test Data) ===")
momentum_cols = ['Momentum1M', 'Momentum3M', 'Momentum6M', 'Momentum12M']
print(test_df[momentum_cols].describe())

print("\n=== Momentum 범위 확인 ===")
for col in momentum_cols:
    data = test_df[col].dropna()
    print(f"{col:15s}: min={data.min():8.4f}, max={data.max():8.4f}, mean={data.mean():8.4f}")

print("\n=== 샘플 데이터 (Momentum이 있는 행) ===")
sample = test_df[test_df['Momentum1M'].notna()][['Date', 'Symbol', 'Close', 'Momentum1M', 'Momentum3M']].head(20)
print(sample.to_string(index=False))

print("\n=== 실제 수익률 계산 검증 ===")
# 특정 종목의 실제 수익률 확인
test_df_sorted = test_df.sort_values(['Symbol', 'Date'])
abnb = test_df_sorted[test_df_sorted['Symbol'] == 'ABNB'].reset_index(drop=True)
if len(abnb) > 30:
    for i in range(20, 25):
        if pd.notna(abnb.loc[i, 'Momentum1M']):
            curr_close = abnb.loc[i, 'Close']
            prev_close = abnb.loc[i-1, 'Close'] if i > 0 else None
            momentum = abnb.loc[i, 'Momentum1M']
            if prev_close and prev_close > 0:
                actual_return = (curr_close - prev_close) / prev_close
                print(f"Date: {abnb.loc[i, 'Date']}, Close: {curr_close:.2f}, Prev: {prev_close:.2f}")
                print(f"  실제 수익률: {actual_return:.6f} ({actual_return*100:.2f}%)")
                print(f"  Momentum1M: {momentum:.6f}")
                print()
