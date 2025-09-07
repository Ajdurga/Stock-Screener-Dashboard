import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import feedparser
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
import time
import mplfinance as mpf
import numpy as np
import requests

# Suppress downcasting warning
pd.set_option('future.no_silent_downcasting', True)


#@st.cache_data(ttl=60)  # Cache for 1 minute
st.set_page_config(
        page_title="S&P Performance Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
) 


def fetch_news(ticker, counter):
    try:
        response = requests.get(f"https://api.tickertick.com/feed?q=z:{ticker}&n={counter}")
        return response.json() if response.status_code == 200 else []
    except:
        print("Error fetching news data")
        return []



def get_data():
    with st.spinner('Fetching stock details...'):
        return main()

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

def validateSymbol(symbol):
    
    ticker = yf.Ticker(symbol)    
    #print(ticker.info.get('currency'))
    try:
        ticker_info = ticker.info
        #print(ticker_info)
        if ("quoteType" in ticker_info and ticker_info["quoteType"] == "EQUITY") or (symbol == "^SPX") or (symbol == "SPX"):
            return True, ticker.info.get('currency'), ticker.info.get('fullExchangeName')
        else:
            return False, 'USD', ''
    except Exception as e:
        print(e)
        return False, 'USD', ''

def get_monthly_data(p_symbol, p_period='20y', p_interval='1mo'):
    df = yf.download(p_symbol, group_by="Ticker", period=p_period
                     #start='2021-01-01', end='2025-08-11' 
                     , interval=p_interval)
    if df.empty:
        print(f"No monthly data found for {p_symbol} : skipping.....")
        return

    df = df.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)

    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()
    df["SMA10"] = df["Close"].rolling(window=10).mean()
    df["SMA5"] = df["Close"].rolling(window=5).mean()
    df = df.round(2)
    df = df.tail(26)

    return df

def get_weekly_data(p_symbol, p_period='210mo', p_interval='1wk'):
    df = yf.download(p_symbol, group_by="Ticker", period=p_period
                     #start='2021-01-01', end='2025-08-11'
                     , interval=p_interval)
    if df.empty:
        print(f"No weekly data found for {p_symbol} : skipping.....")
        return

    df = df.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)

    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()
    df["SMA10"] = df["Close"].rolling(window=10).mean()
    df["SMA5"] = df["Close"].rolling(window=5).mean()
    df = df.round(2)
    df = df.tail(110)
    return df

def downloader(p_symbol, p_period='13mo', p_interval='1d'):
    df = yf.download(p_symbol, group_by="Ticker", period=p_period
                     #start='2021-01-01', end='2025-08-11' 
                     , interval=p_interval)
    return df

def generate_monthly_chart(p_symbol):
    df_monthly = get_monthly_data(p_symbol=p_symbol)
    
    # scatter_data = pd.Series(np.nan, index=df_monthly.index) #[0 for _ in range(26)]
    # scatter_data[-1] = df_monthly['SMA5'].iloc[-1]
    
    add_plots = [
        #mpf.make_addplot(mav_20.loc[last_20_rows.index], panel=0, color='red', width=1.5, type='line')
        mpf.make_addplot(df_monthly['SMA10'], panel=0, type='line', color='red', width=1.5),
        mpf.make_addplot(df_monthly['SMA20'], panel=0, type='line', color='green', width=1.5),
        mpf.make_addplot(df_monthly['SMA50'], panel=0, type='line', color='brown', width=1.5),
        mpf.make_addplot(df_monthly['SMA5'], panel=0, type='line', color='blue', width=1.5),
        #mpf.make_addplot(scatter_data, panel=0, type='scatter', color='red', markersize=200, marker='^',)
    ]
    #fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), sharex=True)     
    #mpf.plot(df_monthly, type='candle', ax=ax1, style='yahoo')    
    
    # Plot the candlestick chart with the overlaid line graph
    # fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), sharex=True)     
    # apds = [
    #     mpf.make_addplot(df_monthly['SMA20'], ax=ax1, panel=0, type='line', color='red', width=1)
    # ]
    # #mpf.plot(df_monthly, type='candle', ax=ax1, style='yahoo')   
    # mpf.plot(df_monthly, type='candle',ax=ax1, title='Candlestick with SMA', style='yahoo')

    fig, axlist = mpf.plot(
        df_monthly, 
        type='candle', 
        style='yahoo',
        #mav=(20, 50, 200),
        addplot=add_plots,
        #title=f'{p_symbol} Candlestick with SMA',
        ylabel='Price ($)',
        #volume=True,
        #figscale=1.2,
        returnfig=True  # Set to True to return the figure and axes
    )

    #x_pos = len(df_monthly) - 1  # Use numeric position instead of datetime
    # axlist[0].annotate(f'SMA5: {df_monthly['SMA5'].iloc[-1]}', 
    #             xy=(x_pos, df_monthly['SMA5'].iloc[-1]), 
    #             xytext=(10, 15), 
    #             textcoords='offset points',
    #             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
    #             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    # axlist[0].annotate(f'SMA10: {df_monthly['SMA10'].iloc[-1]}', 
    #             xy=(x_pos, df_monthly['SMA10'].iloc[-1]), 
    #             xytext=(10, 15), 
    #             textcoords='offset points',
    #             #fontweight='bold',
    #             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
    #             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    # axlist[0].annotate(f'SMA20: {df_monthly['SMA20'].iloc[-1]}', 
    #             xy=(x_pos, df_monthly['SMA20'].iloc[-1]), 
    #             xytext=(10, 15), 
    #             textcoords='offset points',
    #             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
    #             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    # axlist[0].annotate(f'SMA50: {df_monthly['SMA50'].iloc[-1]}', 
    #             xy=(x_pos, df_monthly['SMA50'].iloc[-1]), 
    #             xytext=(10, 15), 
    #             textcoords='offset points',
    #             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
    #             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    # axlist[0].text(x_pos, df_monthly['SMA5'].iloc[-1], f'SMA5:{df_monthly['SMA5'].iloc[-1]}',
    #     ha='center', va='bottom', fontsize=9, color='blue', fontweight='bold')
    # axlist[0].text(x_pos, df_monthly['SMA10'].iloc[-1], f'SMA10:{df_monthly['SMA10'].iloc[-1]}',
    #     ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')
    # axlist[0].text(x_pos, df_monthly['SMA20'].iloc[-1], f'SMA20:{df_monthly['SMA20'].iloc[-1]}',
    #     ha='center', va='bottom', fontsize=9, color='green', fontweight='bold')
    # axlist[0].text(x_pos, df_monthly['SMA50'].iloc[-1], f'SMA50:{df_monthly['SMA50'].iloc[-1]}',
    #     ha='center', va='bottom', fontsize=9, color='brown', fontweight='bold')
    graph_label=f"""
SMA5: {df_monthly['SMA5'].iloc[-1]}
SMA10:{df_monthly['SMA10'].iloc[-1]}
SMA20:{df_monthly['SMA20'].iloc[-1]}
SMA50:{df_monthly['SMA50'].iloc[-1]}
    """
    axlist[0].text(0, df_monthly['Close'].max() * 0.95, 
            graph_label,
          ha='left', va='bottom', fontsize=9, color='black', fontweight='normal',
          fontname='monospace')

    fig.tight_layout() 
    return fig

def generate_weekly_chart(p_symbol):    
    #fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), sharex=True) 
    df_weekly = get_weekly_data(p_symbol=p_symbol)
    
    #mpf.plot(df_weekly, type='candle', ax=ax1, style='yahoo')

    add_plots = [
       mpf.make_addplot(df_weekly['SMA10'], panel=0, type='line', color='red', width=1.5),
       mpf.make_addplot(df_weekly['SMA20'], panel=0, type='line', color='green', width=1.5),
       mpf.make_addplot(df_weekly['SMA50'], panel=0, type='line', color='brown', width=1.5),
       mpf.make_addplot(df_weekly['SMA5'], panel=0, type='line', color='blue', width=1.5)
    ]
    fig, axlist = mpf.plot(df_weekly, type='candle', addplot=add_plots, 
                           style='yahoo', returnfig=True)

    graph_label=f"""
SMA5: {df_weekly['SMA5'].iloc[-1]}
SMA10:{df_weekly['SMA10'].iloc[-1]}
SMA20:{df_weekly['SMA20'].iloc[-1]}
SMA50:{df_weekly['SMA50'].iloc[-1]}
    """
    axlist[0].text(0, df_weekly['Close'].max() * 0.95, 
            graph_label,
          ha='left', va='bottom', fontsize=9, color='black', fontweight='normal',
          fontname='monospace')                       
    fig.tight_layout() 
    return fig


def generateVisuals(p_symbol, p_currency):
    df = downloader(p_symbol=p_symbol, p_period='40mo', p_interval='1d')
    
    if df.empty:
        print(f"No data found for {p_symbol} : skipping.....")
        return

    df = df.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
    #df = p_data
    #stop_loss = df['volatility_bbl'].iloc[-1]
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
    # if df['aroon_up'][i] > df['aroon_down'][i] and (i == 0 or df['aroon_up'][i-1] <= df['aroon_down'][i-1]):
    #         signals.append("Buy")  # Buy signal when Aroon Up crosses above Aroon Down
    #     elif df['aroon_up'][i] < df['aroon_down'][i] and (i == 0 or df['aroon_up'][i-1] >= df['aroon_down'][i-1]):
    #         signals.append("Sell")  # Sell signal when Aroon Down crosses above Aroon Up
    #     else:
    #         signals.append("Hold")  # Hold signal when no crossover occurs
    macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Line'] = macd.macd_signal()
    df['MACD_Histogram'] = macd.macd_diff()
    df['MACD_Signal'] = 0
    df['MACD_Signal'][(df['MACD'] < 0) & (df['MACD'] >= df['MACD_Line'])] = 1  # Buy signal
    df['MACD_Signal'][(df['MACD'] > 0) & (df['MACD'] <= df['MACD_Line'])] = -1 # Sell signal
    adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ADX'] = adx.adx()
    df['ADX_Positive'] = adx.adx_pos()
    df['ADX_Negative'] = adx.adx_neg()
    df['ADX_Signal'] = 0
    df['ADX_Signal'][(df['ADX'] > 20) & (df['ADX_Positive'] > df['ADX_Negative'])] = 1  # Buy signal
    df['ADX_Signal'][(df['ADX'] > 20) & (df['ADX_Negative'] > df['ADX_Positive'])] = -1  # Sell signal

    df = df.round(2)
    df = df.tail(calculate_days_from_prev_year_month_start())


    # Create two subplots: one for the adjusted close price and one for the volume
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]}) 

    # Plot Adjusted Close Price on the first subplot
    sma20, sma50, sma200 = df['SMA20'].iloc[-1], df['SMA50'].iloc[-1], df['SMA200'].iloc[-1]
    bb_bbh, bb_bbl = df['bb_bbh'].iloc[-1], df['bb_bbl'].iloc[-1]
    sma5, sma10 = df['SMA5'].iloc[-1], df['SMA10'].iloc[-1]
    ax1.plot(df.index, df['Close'], label=f'Daily Close', color='blue') 
    ax1.plot(df.index, df['SMA5'], label=f'SMA 5: {sma5}', color='grey', linewidth=0.5) 
    ax1.plot(df.index, df['SMA10'], label=f'SMA 10: {sma10}', color='grey', linewidth=0.5)
    ax1.plot(df.index, df['SMA20'], label=f'SMA 20: {sma20}', color='red') 
    ax1.plot(df.index, df['SMA50'], label=f'SMA 50: {sma50}', color='green')
    ax1.plot(df.index, df['SMA200'], label=f'SMA 200: {sma200}', color='brown')  
    ax1.plot(df.index, df['bb_bbh'], label=f'BBH: {bb_bbh}', color='black', linestyle=':') 
    ax1.plot(df.index, df['bb_bbl'], label=f'BBL: {bb_bbl}', color='black', linestyle=':') 
    #ax1.axhline(y=p_projected_value, color='b', linestyle='-', label=f'Projected Value: {p_projected_value:.2f}')
    #ax1.axhline(y=p_current_price, color='g', linestyle='-', label=f'Current Value: {p_current_price:.2f}')
    #ax1.axhline(y=stop_loss, color='r', linestyle='-', label=f'Stop Loss: {stop_loss}')
    #ax1.axhline(y=p_sma_200, color='tab:brown', linestyle=':', label=f'SMA 200: {p_sma_200:.2f}')
    #ax1.axhline(y=p_sma_50, color='tab:brown', linestyle=':', label=f'SMA 50: {p_sma_50:.2f}')
    #ax1.axhline(y=p_sma_20, color='tab:brown', linestyle=':', label=f'SMA 20: {p_sma_20:.2f}')
    ax1.set_ylabel(f'Daily Close Price ({p_currency})', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title(f'{p_symbol} Daily Stock Performance and Volume')
    ax1.grid(True)
    ax1.legend()
    
    # Add annotation example - this should work
    latest_price = df['Close'].iloc[-1]
    latest_date = df.index[-1]
    ax1.annotate(f'Close Price: ${latest_price:.2f}', 
                xy=(latest_date, latest_price), 
                xytext=(10, 10), 
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    


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

    ax3.plot(df.index, df['ADI'], label=f'Accumulation Distribution Index', color='blue') 
    ax3.set_ylabel('Volume', color='blue')
    ax3.tick_params(axis='y', labelcolor='blue')
    ax3.grid(True)
    ax3.legend()

    

    # Improve layout and display the plot
    fig.tight_layout()    
    #plt.savefig(f"images/{p_symbol}.png")
    df.to_csv(f"{p_symbol}.csv", index=True)

    summary = {
        'Current_Price': df['Close'].iloc[-1],
        'High_52_Week': df['52_week_high'].iloc[-1],
        'Low_52_Week': df['52_week_low'].iloc[-1],
        # 'SMA5': df['SMA5'].iloc[-1], 
        # 'SMA10': df['SMA10'].iloc[-1],
        # 'SMA20': df['SMA20'].iloc[-1], 
        # 'SMA50': df['SMA50'].iloc[-1],
        # 'SMA200': df['SMA200'].iloc[-1],
        'RSI': df['RSI'].iloc[-1],
        'RSI_Signal': "BUY" if df['RSI'].iloc[-1] < 30 else "SELL" if df['RSI'].iloc[-1] > 70 else "HOLD",
        'SMA_Signal': "BUY" if df['SMA20'].iloc[-1] > df['SMA50'].iloc[-1] and df['SMA50'].iloc[-1] > df['SMA200'].iloc[-1]  else "SELL" if df['SMA20'].iloc[-1] < df['SMA50'].iloc[-1] else "HOLD",
        'bb_signal': "BUY" if df['bb_bbhi'].iloc[-1] == 1 else "SELL" if df['bb_bbli'].iloc[-1] == 1 else "HOLD",
        #'StochRSI': df['StochRSI'].iloc[-1],
        # 'StochRSI_K': df['StochRSI_K'].iloc[-1],
        # 'StochRSI_D': df['StochRSI_D'].iloc[-1],
        'StochRSI_Signal': "BUY" if df['StochRSI'].iloc[-1] < 0.20 else "SELL" if df['StochRSI'].iloc[-1] > 0.80 else "HOLD",
        'VWAP': df['VWAP'].iloc[-1],
        'VWAP_Signal': "BUY" if df['Close'].iloc[-1] > df['VWAP'].iloc[-1] else "SELL" if df['Close'].iloc[-1] < df['VWAP'].iloc[-1] else "HOLD",
        'ATR': df['ATR'].iloc[-1],        
        'Aroon_Up': df['Aroon_Up'].iloc[-1],
        'Aroon_Down': df['Aroon_Down'].iloc[-1],
        'Aroon_Indicator': df['Aroon_Indicator'].iloc[-1],
        'MACD_Indicator': "BUY" if df['MACD_Signal'].iloc[-1] == 1 else "SELL" if df['MACD_Signal'].iloc[-1] == -1 else "HOLD",
        'ADX_Indicator': "BUY" if df['ADX_Signal'].iloc[-1] == 1 else "SELL" if df['ADX_Signal'].iloc[-1] == -1 else "HOLD",
        # 'bb_bbh': df['bb_bbh'].iloc[-1],
        # 'bb_bbl': df['bb_bbl'].iloc[-1],
        # 'bb_bbhi': df['bb_bbhi'].iloc[-1],
        # 'bb_bbli': df['bb_bbli'].iloc[-1],
        # 'bb_bbw': df['bb_bbw'].iloc[-1],
        # 'bb_bbp': df['bb_bbp'].iloc[-1]
    }

    return fig, summary

def main():

    st.markdown("""
<style>
.stTextInput input[aria-label="My colored input"] {
    background-color: #0066cc; /* Background color */
    color: #33ff33; /* Text color */
}
</style>
""", unsafe_allow_html=True)
    symbol_from_query_param = st.query_params.get("symbol")
    symbol = (st.text_input("Enter Stock Symbol", value=symbol_from_query_param if symbol_from_query_param  is not None else "SPX", max_chars=None, key=None, type="default", help=None, autocomplete=None, on_change=None, placeholder=None, disabled=False, label_visibility="visible", icon=None, width="stretch")).upper()
    symbol_valid, currency, market = validateSymbol(symbol)
    if symbol_valid:
        with st.spinner(f'Fetching stock details for {symbol}...'):
            with st.container():
                c1, c2 = st.columns((3, 1))
                with c1:    
                    time.sleep(1)
                    if symbol == "SPX":
                        symbol = "^SPX"
                    col1, spacer, col2 = st.columns((1, 2, 1))
                    with col1:
                        st.write(f"Symbol: {symbol}")
                    with col2:
                        st.write(f"Exchange Name: {market}")
                    #st.write(f"Symbol: {symbol} - Exchange Name: {market}")
                    fig, summary = generateVisuals(symbol, currency)
                    if fig is not None:
                        st.pyplot(fig)
                    monthly, weekly = st.columns(2)
                    with monthly:
                        st.write("Monthly Chart")
                        monthly_chart = generate_monthly_chart(symbol)
                        if monthly_chart is not None:
                            st.pyplot(monthly_chart)
                    with weekly:
                        st.write("Weekly Chart")
                        weekly_chart = generate_weekly_chart(symbol)
                        if weekly_chart is not None:
                            st.pyplot(weekly_chart)
                    
                    st.write(summary)
                with c2:
                    st.write("Latest News")
                    news_data = fetch_news(symbol, 10)
                    if news_data.get('stories'):
                        for article in news_data.get('stories'):
                            title = article.get('title', 'No Title')
                            url = article.get('url')
                            if url:
                                st.markdown(f"[{title}]({url})", unsafe_allow_html=True)
                            else:
                                st.markdown(f"{title}", unsafe_allow_html=True)
                            description = article.get('description', 'No description available')
                            if len(description) > 200:
                                description = description[:200] + "..."
                            st.markdown(f"<small>{description}</small>", unsafe_allow_html=True)
                            st.divider()
    else:
       st.write(f"Invalid Symbol: {symbol}")
       return

        

main()
