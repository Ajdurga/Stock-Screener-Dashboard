# indicators - https://www.investopedia.com/top-7-technical-analysis-tools-4773275
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.momentum import StochRSIIndicator
from ta.volume import VolumeWeightedAveragePrice
from ta.volatility import AverageTrueRange
from ta.volume import AccDistIndexIndicator
from ta.trend import AroonIndicator
from ta.trend import MACD
from ta.trend import ADXIndicator


# def compute_rsi(data, window=14):
#     delta = data.diff()
#     gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
#     rs = gain / loss
#     rsi = 100 - (100 / (1 + rs))
#     return rsi

def sp():
    df = pd.read_csv("./data/sp500e.csv")
    return df
sp500_companies = sp()

def calculate_days_from_prev_year_month_start():
    """
    Calculates the number of days from the beginning of the previous month of the previous year
    to the given current date.
    Args:
        current_date (date): The current date as a datetime.date object.
    Returns:
        int: The number of days.
    """

    current_date = date.today()
    # Get the year of the previous year
    prev_year = current_date.year - 1

    # Create a date object for the first day of the current month in the previous year
    start_date = date(prev_year, current_date.month-1, 1)

    # Calculate the difference in days
    delta = current_date - start_date
    return delta.days

def downloader(p_symbol, p_period='30mo', p_interval='1d'):
    df = yf.download(p_symbol, group_by="Ticker", period=p_period
                     #start='2021-01-01', end='2025-08-11' 
                     , interval=p_interval)
    return df

def generateVisuals(p_symbol, p_company, p_data, p_projected_value, p_projected_pcnt, p_current_price):
    # df = downloader(p_symbol=p_symbol, p_period='12mo', p_interval='1mo')
    
    # if df.empty:
    #     print(f"No data found for {p_symbol} : skipping.....")
    #     return

    #df = df.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
    df = p_data.tail(calculate_days_from_prev_year_month_start())
    stop_loss = df['bb_bbl'].iloc[-1]

    # plt.figure(figsize=(10, 6)) # Create a figure and axes for the plot
    # plt.plot(df["Close"])  # Plot the 'Adj Close' column
    # plt.title(f"{p_symbol} Monthly Adjusted Close Price") # Set plot title
    # plt.xlabel("Date")  # Label the x-axis
    # plt.ylabel("Close Price") # Label the y-axis
    # plt.grid(True) # Add grid lines for better readability
    # plt.show()  # Display the plot

    # Create two subplots: one for the adjusted close price and one for the volume
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]}) 

    # Plot Adjusted Close Price on the first subplot
    ax1.plot(df.index, df['Close'], label=f'{p_symbol} Daily Close', color='blue') 
    ax1.axhline(y=p_projected_value, color='b', linestyle='-', label=f'Target Price: {p_projected_value:.2f}')    
    ax1.axhline(y=p_current_price, color='g', linestyle='-', label=f'Close Price: {p_current_price:.2f}')
    ax1.axhline(y=stop_loss, color='r', linestyle='-', label=f'Stop Loss: {stop_loss}')
    ax1.text(df.index[-1], p_projected_value, f'{p_projected_value:.2f}', fontsize=14, color='b', ha='left', va='bottom')
    ax1.text(df.index[-1], p_current_price, f'{p_current_price:.2f}', fontsize=14, color='g', ha='left', va='bottom')
    ax1.text(df.index[-1], stop_loss, f'{stop_loss:.2f}', fontsize=14, color='red', ha='left', va='bottom')
    sma5 = df['SMA5'].iloc[-1]
    sma10 = df['SMA10'].iloc[-1]
    sma20 = df['SMA20'].iloc[-1]
    sma50 = df['SMA50'].iloc[-1]
    sma200 = df['SMA200'].iloc[-1]
    ax1.plot(df.index, df['SMA5'], label=f'SMA 5: {sma5}', color='black', linewidth=0.5) 
    ax1.plot(df.index, df['SMA10'], label=f'SMA 10: {sma10}', color='grey', linewidth=0.5) 
    ax1.plot(df.index, df['SMA20'], label=f'SMA 20: {sma20}', color='grey', linestyle=':') 
    ax1.plot(df.index, df['SMA50'], label=f'SMA 50: {sma50}', color='grey', linestyle=':') 
    ax1.plot(df.index, df['SMA200'], label=f'SMA 200: {sma200}', color='grey', linestyle=':') 
    ax1.set_ylabel('Daily Close Price (USD)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title(f'{p_company} Daily Stock Performance and Volume')
    ax1.grid(True)
    ax1.legend()
    #ax1.text(0.0, 0.8, f'Projected Upside {p_projected_pcnt}%, Projected Value {p_projected_value:.2f}', transform=ax1.transAxes, 
    #        fontsize=14, color='red', ha='left', va='center')
    #ax1.text(0.5, 0.9, f'Projected Value {p_projected_value:.2f}', transform=ax1.transAxes, 
    #        fontsize=12, color='red', ha='left', va='center')
    #ax1.text(0.5, 0.9, 'Top-left Text', transform=ax1.transAxes, ha='center', va='center', fontsize=12, color='blue')
    #ax1.text(0.1, 0.1, 'Bottom-left Text', transform=ax1.transAxes, ha='left', va='bottom', fontsize=10, color='red')


    # Plot Volume on the second subplot as a bar chart
    ax2.bar(df.index, df['Volume']/ 1_000_000, label=f'{p_symbol} Daily Volume in Millions', color='gray'
            #, alpha=0.7
            , width=3.0) 
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volume', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.grid(True)
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
    ax2.legend()


    # Improve layout and display the plot
    fig.tight_layout()
    plt.savefig(f"images/{p_symbol}.png")
    #plt.plot()

    


# Step 3: Loop through each ticker and collect data
results = []
for index, row in sp500_companies.iterrows():
    symbol = row['Symbol']
    symbol = symbol.replace('.','-')
    company = row['Security']
    gics_sector = row['GICS Sector']
    gics_sub_industry = row['GICS Sub-Industry']    

    df = downloader(p_symbol=symbol)
    
    if df.empty:
        print(f"No data found for {symbol} - {company}: skipping.....")
        continue
        
    #col_names = df.columns
    #print(col_names)
    df = df.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
    #stage_df = df.copy() 
    df["SMA5"] = df["Close"].rolling(window=5).mean()
    df["SMA10"] = df["Close"].rolling(window=10).mean()
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()
    df["SMA200"] = df["Close"].rolling(window=200).mean()
    df['52_week_high'] = df['Close'].rolling(window=252).max()
    df['52_week_low'] = df['Close'].rolling(window=252).min()
        
    
    indicator_bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    # Add Bollinger Band indicators to the DataFrame
    df['bb_bbm'] = indicator_bb.bollinger_mavg()  # Middle Band (Moving Average)
    df['bb_bbh'] = indicator_bb.bollinger_hband()  # Upper Band
    df['bb_bbl'] = indicator_bb.bollinger_lband()  # Lower Band
    # You can also get indicators for band signals, width, and percentage
    df['bb_bbhi'] = indicator_bb.bollinger_hband_indicator() # High Band Indicator (1 if above upper band, 0 otherwise)
    df['bb_bbli'] = indicator_bb.bollinger_lband_indicator() # Low Band Indicator (1 if below lower band, 0 otherwise)
    df['bb_bbw'] = indicator_bb.bollinger_wband()  # Bollinger Band Width
    df['bb_bbp'] = indicator_bb.bollinger_pband()  # Bollinger Band Percentage
    rsi_indicator = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi_indicator.rsi()
    stochrsi_indicator = StochRSIIndicator(
        close=df["Close"],
        window=14,  # Recommended window for SMI (adjust as needed)
        fillna=False
    )
    df['StochRSI'] = stochrsi_indicator.stochrsi()
    df['StochRSI_K'] = stochrsi_indicator.stochrsi_k()
    df['StochRSI_D'] = stochrsi_indicator.stochrsi_d()
    vwap = VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'])
    df['VWAP'] = vwap.volume_weighted_average_price()
    atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'])
    df['ATR'] = atr.average_true_range()
    adi = AccDistIndexIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'])
    df['ADI'] = adi.acc_dist_index()
    aroon = AroonIndicator(high=df['High'], low=df['Low'], window=25) #(df['Close'])
    df['Aroon_Up'] = aroon.aroon_up()
    df['Aroon_Down'] = aroon.aroon_down()
    df['Aroon_Indicator'] = aroon.aroon_indicator()

    df['RSI_Signal'] = 0
    df['RSI_Signal'][df['RSI'] < 30] = 1  # Buy signal
    df['RSI_Signal'][df['RSI'] > 70] = -1 # Sell signal
    df['SMA_Signal'] = 0
    df['SMA_Signal'][(df['SMA50'] > df['SMA200']) & (df['SMA20'] > df['SMA50'])] = 1  # Buy signal df['SMA_Signal'][(df['SMA50'] > df['SMA200']) & (df['SMA20'] > df['SMA50'])] 
    df['SMA_Signal'][df['SMA20'] < df['SMA50']] = -1 # Sell signal
    macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Line'] = macd.macd_signal()
    df['MACD_Histogram'] = macd.macd_diff()
    df['MACD_Signal'] = 0
    df['MACD_Signal'][(df['MACD'] < 0) & (df['MACD'] >= df['MACD_Line'])] = 1  # Buy signal
    df['MACD_Signal'][(df['MACD'] > 0) & (df['MACD'] <= df['MACD_Line'])] = -1 # Sell signal
    df['BB_Signal'] = 0
    df['BB_Signal'][df['bb_bbhi'] == 1] = 1  # Buy signal
    df['BB_Signal'][df['bb_bbli'] == 1] = -1 # Sell signal
    adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ADX'] = adx.adx()
    df['ADX_Positive'] = adx.adx_pos()
    df['ADX_Negative'] = adx.adx_neg()
    df['ADX_Signal'] = 0
    df['ADX_Signal'][(df['ADX'] > 20) & (df['ADX_Positive'] > df['ADX_Negative'])] = 1  # Buy signal
    df['ADX_Signal'][(df['ADX'] > 20) & (df['ADX_Negative'] > df['ADX_Positive'])] = -1  # Sell signal
    df['VWAP_Signal'] = 0
    df['VWAP_Signal'][(df['VWAP'] > df['Close'])] = 1  # Buy signal
    df['VWAP_Signal'][(df['VWAP'] < df['Close'])] = -1  # Sell signal

    
    df['Score'] = 0.0
    df['Score'] = (df['RSI_Signal'] * 0.2 + df['SMA_Signal'] * 0.1  + df['VWAP_Signal'] * 0.1 + df['MACD_Signal'] * 0.2 + df['BB_Signal'] * 0.2 + df['ADX_Signal'] * 0.2)
    #(df['RSI_Signal'] + df['SMA_Signal'] + df['MACD_Signal'] + df['BB_Signal'] + df['ADX_Signal']) / 5
    
    #df['NewScore'] = (df['RSI_Signal'] * 0.2 + df['SMA_Signal'] * 0.2 + df['MACD_Signal'] * 0.2 + df['BB_Signal'] * 0.2 + df['ADX_Signal'] * 0.2)
    #df['NewScore2'] = (df['RSI_Signal'] * 0.2 + df['SMA_Signal'] * 0.1  + df['VWAP_Signal'] * 0.1 + df['MACD_Signal'] * 0.2 + df['BB_Signal'] * 0.2 + df['ADX_Signal'] * 0.2)
    

    df = df.round(2)
    df.to_csv(f"data/stage/{symbol}.csv")

    target_percent = round((df['bb_bbh'].iloc[-1] - df['Close'].iloc[-1]) / df['Close'].iloc[-1] * 100, 2)
    target_price = round(df['Close'].iloc[-1] * (1 + target_percent / 100), 2)

    # print(f"Symbol: {symbol}, Company: {company}, Target Price: {target_price}, Target Percent: {target_percent}")
    # print(f'Score:{df['Score'].iloc[-1]}, NewScore:{df['NewScore'].iloc[-1]}, NewScore2:{df['NewScore2'].iloc[-1]}')
    

    generateVisuals(p_symbol=symbol, p_company=company, p_data=df, p_projected_pcnt=target_percent,p_projected_value=target_price,p_current_price=df['Close'].iloc[-1])

    results.append({
        'Symbol': symbol,
        'Company': company,
        'GICS_Sector': gics_sector,
        'GICS_Sub_Industry': gics_sub_industry,
        'Current_Price': df['Close'].iloc[-1],
        'High_52_Week': df['52_week_high'].iloc[-1],
        'Low_52_Week': df['52_week_low'].iloc[-1],
        'SMA5': df['SMA5'].iloc[-1], 
        'SMA10': df['SMA10'].iloc[-1],
        'SMA20': df['SMA20'].iloc[-1], 
        'SMA50': df['SMA50'].iloc[-1],
        'SMA200': df['SMA200'].iloc[-1],        
        'Score': df['Score'].iloc[-1],        
        'Target_Pcnt': target_percent,
        'Target_Price': target_price,
        'Stop_Loss': df['bb_bbl'].iloc[-1]
    })

#print(results)    
results_df = pd.DataFrame(results)
results_df.to_csv('data/sp500_technical_scores.csv', index=False)

portfolio = results_df[results_df['Score'] > 0]
portfolio.to_csv('data/portfolio.csv', index=False)

# indicator_bb = BollingerBands(close=stage_df["Close"], window=20, window_dev=2)

# # Add Bollinger Bands features to DataFrame
# stage_df['bb_bbm'] = indicator_bb.bollinger_mavg()  # Middle Band
# stage_df['bb_bbh'] = indicator_bb.bollinger_hband()  # Upper Band
# stage_df['bb_bbl'] = indicator_bb.bollinger_lband()  # Lower Band
# stage_df['bb_shift'] = stage_df['bb_bbh'].shift(1)
# stage_df['BB_Signal'] = 0
# stage_df['BB_Signal'][stage_df['Close'] > stage_df['bb_bbh'].shift(1)] = 1  # Buy signal
# stage_df['BB_Signal'][stage_df['Close'] < stage_df['bb_bbl'].shift(1)] = -1 # Sell signal

# stage_df.to_csv("aapl_bb.csv")
