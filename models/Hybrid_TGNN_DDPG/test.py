import pandas as pd
import numpy as np

# Test 데이터 로드
df = pd.read_csv("../../data/test_data.csv")
df["Date"] = pd.to_datetime(df["Date"])

# 7개 종목 필터
symbols = ["AMD", "AZN", "BKNG", "COST", "CTAS", "EXC", "GOOG"]
df = df[df["Symbol"].isin(symbols)]

# 월말 데이터
monthly = (
    df.set_index("Date").groupby(["Symbol", pd.Grouper(freq="ME")]).last().reset_index()
)

# Momentum1M 평균
print("Momentum1M 통계:")
print(monthly["Momentum1M"].describe())
print(f"합계: {monthly['Momentum1M'].sum():.2f}%")
print(f"평균: {monthly['Momentum1M'].mean():.2f}%")
