import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st


st.set_page_config(
        page_title="S&P Performance Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
) 

# Step 1: Get S&P 500 companies from Wikipedia
def sp():
    df = pd.read_csv("data/portfolio.csv")
    return df
sp500_companies = sp()

def load_sp_master():
    df = pd.read_csv("data/sp500e.csv")
    return df
sp500_master = load_sp_master()  

portfolio = sp500_companies[(sp500_companies['Current_Price'] > sp500_companies['SMA5']) & (sp500_companies['Current_Price'] > sp500_companies['SMA10']) & (sp500_companies['Current_Price'] > sp500_companies['SMA20']) & (sp500_companies['Current_Price'] > sp500_companies['SMA50']) & (sp500_companies['Current_Price'] > sp500_companies['SMA200']) ]

refresh_date=""
with open('data/refresh_date.txt', 'r') as file:
    refresh_date = file.read()
st.title(f"Data As Of: {refresh_date.strip()} EST")
st.caption("List of stocks where 5 SMA > 10 SMA, 10 SMA > 20 SMA, 20 SMA > 50 SMA and 50 SMA > 200 SMA.")

tab_labels = portfolio["GICS_Sector"].unique().tolist()
sorted_tabs = sorted(tab_labels)
tabs = st.tabs(sorted_tabs)

for i, tab_label in enumerate(sorted_tabs):
    #print(tab)
    sector_ideas = portfolio[portfolio["GICS_Sector"]==tab_label]
    ##print(sector_ideas.size)

    with tabs[i]:
        st.header(tab_label)
    
        for stock in sector_ideas.iterrows():
            with st.expander(f"{stock[1].Symbol} - {stock[1].GICS_Sector} - {stock[1].GICS_Sub_Industry}", expanded=True):
                col1, col2 = st.columns(2)  # Create two columns within the expander
        
                with col1:
                    div_yield = sp500_master.loc[sp500_master['Symbol'] == stock[1].Symbol, 'DividendYield'].iloc[0]
                    st.write(f"[:blue[**{stock[1].Company}**]](%s)" % f'/Research?symbol={stock[1].Symbol}')
                    st.write(f"Current Price: :green[**{stock[1].Current_Price}**]")
                    st.write(f"Target Price: :blue[**{stock[1].Target_Price}**] (:blue[**{stock[1].Target_Pcnt}%**] :red[**↑**])")
                    #st.write(f"Target %: :blue[**{stock[1].Target_Pcnt}**]")
                    st.write(f"Stop Loss: :red[**{stock[1].Stop_Loss}**]")
                    st.write(f"52 Week: High - **{stock[1].High_52_Week}** Low - **{stock[1].Low_52_Week}**")
                    #st.write(f"52 Week: **Range:** {stock[1].Low_52_Week} - {stock[1].High_52_Week}")
                    #st.write(f"52 Week Low: {stock[1].Low_52_Week}")
                    st.write(f"Score:**{stock[1].Score:.2f}** Div. Yield%: **{div_yield:.2f}**")
                    if stock[1].High_52_Week - stock[1].Low_52_Week == 0:  # Avoid division by zero if low and high are the same
                        progress_percentage = 0
                    else:
                        progress_percentage = ((stock[1].Current_Price - stock[1].Low_52_Week) / (stock[1].High_52_Week - stock[1].Low_52_Week)) * 100
    
                    # Ensure the percentage is within 0-100
                    progress_percentage = max(0, min(100, progress_percentage))
    
                    st.progress(int(progress_percentage)) # st.progress expects an integer percentage
                with col2:
                    st.image(f'images/{stock[1].Symbol}.png')

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; padding: 1rem; color: #666;">
        <p>📊 Dashboard powered by yfinance & Streamlit | 🔄 Data updates daily</p>
        <p style="font-size: 0.8rem;">⚠️ For educational purposes only - not financial advice</p>
    </div>
    """, 
    unsafe_allow_html=True
)
