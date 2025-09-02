import pandas as pd
import yfinance as yf


# Step 1: Get S&P 500 companies from Wikipedia
# def sp():
#     headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
#     payload = pd.read_html(sp500_url, match='Symbol', header=0, 
#                              storage_options={'User-Agent': headers['User-Agent']})
#     df = payload[0]
#     df.to_csv("./data/sp500.csv", index=False)
#     return df

def get_tickers():
    sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

    # Read in the url and scrape ticker data with User-Agent header
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    data_table = pd.read_html(sp500_url, match='Symbol', header=0, 
                             storage_options={'User-Agent': headers['User-Agent']})
    
    # Get the first table which contains the S&P 500 component stocks
    sp500_table = data_table[0]    
    #print(sp500_table['Symbol'].str.replace('.', '-', regex=False)[40:80])
    sp500_table['Symbol'] = sp500_table['Symbol'].str.replace('.', '-', regex=False)
    sp500_table['Symbol'] = sp500_table['Symbol'].str.replace(' ', '', regex=False)
    sp500_table['Symbol'] = sp500_table['Symbol'].str.replace('\n', '', regex=False)
    #sp500_table.to_csv("./data/sp500.csv", index=False)
    sp500_table['DividendYield'] = 0.0
    sp500_table['PayoutRatio'] = 0.0
    sp500_table['Beta'] = 0.0   
    sp500_table['MarketCap'] = 0.0
    sp500_table['Industry'] = ''
    sp500_table['Sector'] = ''
    sp500_table['Exchange'] = ''
    sp500_table['Currency'] = ''
    sp500_table['TrailingPE'] = 0.0
    sp500_table['ForwardPE'] = 0.0
    sp500_table['DividendRate'] = 0.0
    sp500_table['TrailingEps'] = 0.0
    sp500_table['ForwardEps'] = 0.0


    for index, row in sp500_table.iterrows():
        ticker = row['Symbol']
        #print(f"Fetching data for {ticker}")
        try:
            ticker_data = enrich_ticker(ticker)
            if ticker_data is not None:
                sp500_table.loc[index, 'DividendYield'] = ticker_data['DividendYield']
                sp500_table.loc[index, 'PayoutRatio'] = ticker_data['PayoutRatio']
                sp500_table.loc[index, 'Beta'] = ticker_data['Beta']
                sp500_table.loc[index, 'MarketCap'] = ticker_data['MarketCap']
                sp500_table.loc[index, 'Industry'] = ticker_data['Industry']
                sp500_table.loc[index, 'Sector'] = ticker_data['Sector']
                sp500_table.loc[index, 'Exchange'] = ticker_data['Exchange']
                sp500_table.loc[index, 'Currency'] = ticker_data['Currency']
                sp500_table.loc[index, 'TrailingPE'] = ticker_data['TrailingPE']
                sp500_table.loc[index, 'ForwardPE'] = ticker_data['ForwardPE']
                sp500_table.loc[index, 'DividendRate'] = ticker_data['DividendRate']
                sp500_table.loc[index, 'TrailingEps'] = ticker_data['TrailingEps']
                sp500_table.loc[index, 'ForwardEps'] = ticker_data['ForwardEps']
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            pass

    #print(sp500_table.head())
    # tickers = sp500_table['Symbol'].values.tolist()
    # tickers = [s.replace('\n', '') for s in tickers]
    # tickers = [s.replace('.', '-') for s in tickers]
    # tickers = [s.replace(' ', '') for s in tickers]    
    # print(tickers)
    sp500_table.to_csv("./data/sp500e.csv", index=False)

def enrich_ticker(p_ticker):    
    try:        
        ticker = yf.Ticker(p_ticker)
        ticker_info = ticker.info        
        return {
            'DividendYield': ticker_info.get('dividendYield', 0),
            'PayoutRatio': ticker_info.get('payoutRatio', 0),
            'Beta': ticker_info.get('beta', 0),
            'MarketCap': ticker_info.get('marketCap', 0),
            'Industry': ticker_info.get('industry', ''),
            'Sector': ticker_info.get('sector', ''),
            'Exchange': ticker_info.get('exchange', ''),
            'Currency': ticker_info.get('currency', ''),
            'TrailingPE': ticker_info.get('trailingPE', ''),
            'ForwardPE': ticker_info.get('forwardPE', ''),
            'DividendRate': ticker_info.get('dividendRate', ''),
            'TrailingEps': ticker_info.get('trailingEps', ''),
            'ForwardEps': ticker_info.get('forwardEps', ''),
        }
    except Exception as e:
        print(f"Error fetching data for {p_ticker}: {e}")
        return None


get_tickers()

#sp500_companies = sp()
