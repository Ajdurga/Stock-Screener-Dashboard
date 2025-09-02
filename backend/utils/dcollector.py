import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Step 1: Get S&P 500 companies from Wikipedia
def sp():
    df = pd.read_csv("./data/sp500.csv")
    return df
sp500_companies = sp()

# Step 2: Define function to get technicals for a ticker
def get_technicals(symbol):
    try:
        df = yf.download(symbol, period='13mo', interval='1d')
        if df.empty or len(df) < 200:
            return None  # Not enough data
        close = df['Close']
        # SMAs
        current_price = close.iloc[-1]
        sma_5 = close.rolling(window=5).mean().iloc[-1]
        sma_10 = close.rolling(window=10).mean().iloc[-1]
        sma_20 = close.rolling(window=20).mean().iloc[-1]
        sma_50 = close.rolling(window=50).mean().iloc[-1]
        sma_100 = close.rolling(window=100).mean().iloc[-1]
        sma_125 = close.rolling(window=125).mean().iloc[-1]
        sma_200 = close.rolling(window=200).mean().iloc[-1]

        # RSI
        #rsi = RSIIndicator(close).rsi().iloc[-1]

        #52-week high and low
        high_52wk = df['High'].rolling(window=252).max().iloc[-1]
        low_52wk = df['Low'].rolling(window=252).min().iloc[-1]

        return {
            'Current_Price': round(current_price.item(), 2),
            '52_Week_High': round(high_52wk.item(), 2),
            '52_Week_Low': round(low_52wk.item(), 2),
            'SMA_5': round(sma_5.item(), 2),
            'SMA_10': round(sma_10.item(), 2),
            'SMA_20': round(sma_20.item(), 2),
            'SMA_50': round(sma_50.item(), 2),
            'SMA_100': round(sma_100.item(), 2),
            'SMA_125': round(sma_125.item(), 2),
            'SMA_200': round(sma_200.item(), 2),
            #'RSI_14': round(rsi, 2)
        }

    except Exception as e:
        return None

# Step 3: Loop through each ticker and collect data
results = []
for index, row in sp500_companies.iterrows():
    symbol = row['Symbol']
    company = row['Security']
    gics_sector = row['GICS Sector']
    gics_sub_industry = row['GICS Sub-Industry']    

    data = get_technicals(symbol)
    
    if data:
        results.append({
            'Symbol': symbol,
            'Company': company,
            'GICS_Sector': gics_sector,
            'GICS_Sub_Industry': gics_sub_industry,
            **data
        })

# Step 4: Save results to DataFrame
final_df = pd.DataFrame(results)

# Step 5: Save to CSV
final_df.to_csv('data/sp500_technicals.csv', index=False)
