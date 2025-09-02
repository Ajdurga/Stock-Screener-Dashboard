import pandas as pd
import yfinance as yf
#from ta.momentum import RSIIndicator
#from ta.trend import SMAIndicator
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px

# Suppress downcasting warning
pd.set_option('future.no_silent_downcasting', True)


st.set_page_config(
        page_title="S&P Performance Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
) 

# Custom CSS for selectbox styling
st.markdown("""
<style>
    .stSelectbox > div > div > div {
        background: #F0F0F0;
        color: black;
        border-radius: 8px;
        border: none;
        font-weight: 500;
        font-size: 16px;
    }
    .stSelectbox > div > div > div:hover {
        background: #C0C0C0;
    }
    .stSelectbox > div > div > div > div {
        color: black !important;
        font-size: 18px !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        font-size: 18px !important;
    }
    .stSelectbox [role="combobox"] {
        font-size: 18px !important;
    }
    /* Fix dropdown arrow visibility in all themes */
    .stSelectbox svg,
    .stSelectbox > div > div > svg,
    .stSelectbox [data-testid="stSelectbox"] svg {
        color: #333 !important;
        fill: #333 !important;
        stroke: #333 !important;
        opacity: 1 !important;
    }
    /* Dark theme specific */
    .stApp[data-theme="dark"] .stSelectbox svg,
    [data-theme="dark"] .stSelectbox svg,
    .css-1d391kg .stSelectbox svg {
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
        opacity: 1 !important;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2px;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
        padding-bottom: 5px !important;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-title {
        font-size: 16px;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #34495e;
        margin: 10px 0;
    }
    .metric-delta-positive {
        color: #27ae60;
        font-size: 18px;
        font-weight: 600;
    }
    .metric-delta-negative {
        color: #e74c3c;
        font-size: 18px;
        font-weight: 600;
    }
    @media (max-width: 768px) {
        .metric-card {
            margin-bottom: 15px !important;
        }
        .stColumn {
            padding-bottom: 10px !important;
        }
    } 
</style>
""", unsafe_allow_html=True)

# blue shade for metric box: linear-gradient(135deg, #5DADE2 0%, #3498DB 100%);
# blue shade for select box (normal): linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
# blue shade for select box (hover): linear-gradient(135deg, #357ABD 0%, #2E6BA8 100%);
# gray background: #F0F0F0
# Step 1: Get S&P 500 companies from Wikipedia
def sp():
    df = pd.read_csv("data/sp500e.csv")
    return df
sp500_companies = sp()


def downloader(p_symbol, p_period='30mo', p_interval='1d'):
    df = yf.download(p_symbol, group_by="Ticker", period=p_period,
                     interval=p_interval, auto_adjust=False)
    return df

def calculate_months_from_beginning_of_the_year():
    # Get the current date
    current_date = datetime.now()
    # Get a specific date (e.g., August 28th, 2025)
    specific_date = datetime(current_date.year, 1, 1)
    months_diff = (current_date.year - specific_date.year) * 12 + (current_date.month - specific_date.month) + 1
    
    if months_diff == 0:
        #return "1mo"
        return 1
    else:
        return months_diff
        #return f"{months_diff}mo"

def get_monthly_performance():
    p_symbol = "^SPX"
    #months = calculate_months_from_beginning_of_the_year()
    #print(f'Months: {months}')
    df = downloader(p_symbol=p_symbol, p_period="14mo", p_interval='1mo')
    
    if df.empty:
        print(f"No data found for {p_symbol} : skipping.....")
        return

    df = df.stack(level=0, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index(level=1)
    df.drop(columns=['Ticker', 'Volume', 'Adj Close'], axis=1, inplace=True)
    df['Gain/Loss'] = df['Close'].diff()
    df['Gain/Loss%'] = (df['Gain/Loss'] / df['Close'].shift(1)) * 100
    # append empty row for future month
    next_month = df.index[-1] + pd.DateOffset(months=1)
    empty_row = pd.DataFrame(index=[next_month], columns=df.columns).fillna(0).infer_objects(copy=False)
    df = pd.concat([df, empty_row])

    df = df.round(2)    
    return df

def format_dataframe(val):
    return 'background-color: green' if val > 0 else 'background-color: red' if val < 0 else None

def format_negative_value(val):
    if isinstance(val, (int, float)) and val < 0:
        return f"{abs(val):.2f}"
    return val


def get_historic_performance():
    p_symbol = "^SPX"    
    #print(f'Months: {months}')
    df = yf.download(p_symbol, group_by="Ticker", start="2000-01-01", end="2025-01-01",
                     interval="1mo", auto_adjust=False)
    if df.empty:
        print(f"No data found for {p_symbol} : skipping.....")
        return

    df = df.stack(level=0, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index(level=1)
    df.drop(columns=['Ticker', 'Volume', 'Adj Close'], axis=1, inplace=True)
    df['Gain/Loss'] = df['Close'] - df['Open'] #.diff()
    df['Gain/Loss%'] = (df['Gain/Loss'] / df['Open']) * 100
    
       
    df = df.round(2)
    return df

historic_monthly_performance = get_historic_performance()
historic_monthly_performance.drop(columns=['Open', 'High', 'Low', 'Close', 'Gain/Loss'], axis=1, inplace=True)
# Add year and month columns
historic_monthly_performance['Year'] = historic_monthly_performance.index.year
historic_monthly_performance['Month'] = historic_monthly_performance.index.month_name()
# Transpose: Year as rows, Month as columns
transposed_data = historic_monthly_performance.pivot(index='Year', columns='Month', values='Gain/Loss%')


# Step 4: Save results to DataFrame
final_df = pd.read_csv("data/sp500_technicals.csv")

# # Calculate how many rows have Current_Price > SMA_5 and SMA_10
portfolio = final_df[(final_df['Current_Price'] > final_df['SMA_200']) ]

#portfolio.to_csv('portfolio.csv', index=False)
#print(portfolio)

good_list = final_df[(final_df['Current_Price'] > final_df['SMA_5']) & (final_df['Current_Price'] > final_df['SMA_10']) & (final_df['Current_Price'] > final_df['SMA_20']) & (final_df['Current_Price'] > final_df['SMA_50']) & (final_df['Current_Price'] > final_df['SMA_200']) ]

sma_200_count = len(portfolio)
sma_100_count = len(final_df[(final_df['Current_Price'] > final_df['SMA_100']) ])
sma_50_count = len(final_df[(final_df['Current_Price'] > final_df['SMA_50']) ])
sma_20_count = len(final_df[(final_df['Current_Price'] > final_df['SMA_20']) ])
sma_10_count = len(final_df[(final_df['Current_Price'] > final_df['SMA_10']) ])
sma_5_count = len(final_df[(final_df['Current_Price'] > final_df['SMA_5']) ])



def highlight_row_based_on_value(row):
    if row['Current_Price'] < row['SMA_5'] or row['Current_Price'] < row['SMA_10'] or row['Current_Price'] < row['SMA_20'] or row['Current_Price'] < row['SMA_50']:
        return ['background-color: lightblue; color:red'] * len(row)
    else:
        return [''] * len(row) # No styling

#st.dataframe(portfolio)
#st.title(f'Total Stocks above all the averages {len(good_list)} in S&P 500')

#st.dataframe(portfolio.style.apply(highlight_row_based_on_value, axis=1).format({'Current_Price': '{:.2f}', '52_Week_High': '{:.2f}', '52_Week_Low': '{:.2f}', 'SMA_5': '{:.2f}', 'SMA_10': '{:.2f}', 'SMA_20': '{:.2f}', 'SMA_50': '{:.2f}', 'SMA_100': '{:.2f}', 'SMA_125': '{:.2f}', 'SMA_200': '{:.2f}' }))

st.title("S&P Performance")
st.caption("These below metrics shows the monthly and year to date performance of the S&P 500 stocks. The metrics are updated daily as the month progress. The calculations are using closing values on the last day of the month, values may be off few points take it with a grain of salt! One year performance is always little off as it is using the extra partial month or even the entire month on the last day of the month.")
monthly_performance_temp = get_monthly_performance()
# print(monthly_performance_temp.index[-2])
# print(monthly_performance_temp.index[0])
one_year_performance = round((monthly_performance_temp['Close'].iloc[-2] - monthly_performance_temp['Close'].iloc[0]), 2)
one_year_performance_pcnt = round((monthly_performance_temp['Close'].iloc[-2] - monthly_performance_temp['Close'].iloc[0]) / monthly_performance_temp['Close'].iloc[0] * 100, 2)
ytd_performance = round((monthly_performance_temp['Close'].iloc[-2] - monthly_performance_temp['Close'].iloc[(-1*(calculate_months_from_beginning_of_the_year() + 2))]), 2)
ytd_performance_pcnt = ytd_performance / monthly_performance_temp['Close'].iloc[(-1*(calculate_months_from_beginning_of_the_year() + 2))] *100 #
#print(monthly_performance_temp.index([(-1*(calculate_months_from_beginning_of_the_year() + 2))]).month_name())
#print((-1*(calculate_months_from_beginning_of_the_year() + 2)))
#print(monthly_performance_temp.index[(-1*(calculate_months_from_beginning_of_the_year() + 2))].month_name())
monthly_performance_data = monthly_performance_temp.tail(calculate_months_from_beginning_of_the_year() + 1)
#yearly_performance = round((monthly_performance_data['Close'].iloc[-2] - monthly_performance_data['Open'].iloc[0]) / monthly_performance_data['Open'].iloc[0] * 100, 2)
#print(yearly_performance)
# styled_df = monthly_performance_data.style.map(format_dataframe, subset=['Gain/Loss', 'Gain/Loss%'])
# formated_df = styled_df.format({'Gain/Loss': format_negative_value, 'Gain/Loss%': format_negative_value})
monthly_performance_data.to_csv('monthly_performance_data1.csv', index=True)

vcol1, vcol2 = st.columns([4, 1])
with vcol1:
    with st.container(border=True):
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3, gap="large")

        mtd_delta = monthly_performance_data["Gain/Loss%"].iloc[-2]
        ytd_delta = ytd_performance_pcnt
        yr_delta = one_year_performance_pcnt

        with c1:
            delta_class = "metric-delta-positive" if mtd_delta >= 0 else "metric-delta-negative"
            st.markdown(f"""
            <div class="metric-card">
            <div class="metric-title">📈 Month to Date</div>
            <div class="metric-value">{monthly_performance_data["Gain/Loss"].iloc[-2]:.2f}</div>
            <div class="{delta_class}">{'▲' if mtd_delta >= 0 else '▼'} {abs(mtd_delta):.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            delta_class = "metric-delta-positive" if ytd_delta >= 0 else "metric-delta-negative"
            st.markdown(f"""
            <div class="metric-card">
            <div class="metric-title">📊 Year to Date</div>
            <div class="metric-value">{ytd_performance:.2f}</div>
            <div class="{delta_class}">{'▲' if ytd_delta >= 0 else '▼'} {abs(ytd_delta):.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            delta_class = "metric-delta-positive" if yr_delta >= 0 else "metric-delta-negative"
            st.markdown(f"""
            <div class="metric-card">
            <div class="metric-title">📅 One Year</div>
            <div class="metric-value">{one_year_performance:.2f}</div>
            <div class="{delta_class}">{'▲' if yr_delta >= 0 else '▼'} {abs(yr_delta):.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.header("Monthly Performance")

        st.markdown('''
        <div class="metric-container">
        ''', unsafe_allow_html=True)
        # Create 3 rows of 4 columns each
        for row in range(3):
            cols = st.columns(4)
            if row < 2:  # Add spacing between rows on mobile
                st.markdown('<div style="height: 10px; margin-bottom: 5px;"></div>', unsafe_allow_html=True)
            for col in range(4):
                metric_index = row * 4 + col
                if metric_index < len(monthly_performance_data):
                    month_data = monthly_performance_data.iloc[metric_index]
                    month_name = month_data.name.strftime('%b %Y')
                    gain_loss = month_data['Gain/Loss']
                    gain_loss_pct = month_data['Gain/Loss%']
                    
                    with cols[col]:
                        # st.metric(
                        #     label=month_name,
                        #     value=f"{int(gain_loss)}",
                        #     delta=f"{gain_loss_pct:.2f}%"
                        # )
                        delta_class = "metric-delta-positive" if gain_loss_pct >= 0 else "metric-delta-negative"
                        st.markdown(f"""
                        <div class="metric-card">
                        <div class="metric-title">{month_name}</div>
                        <div class="metric-value">{int(gain_loss)}</div>
                        <div class="{delta_class}">{'▲' if gain_loss_pct >= 0 else '▼'} {abs(gain_loss_pct):.2f}%</div>
                        </div>
                        """, unsafe_allow_html=True)    
                        # with st.popover("..."):
                        #     st.header(f"Historic {monthly_performance_data.index[metric_index].month_name()} Performance")                                
                        #     current_month = monthly_performance_data.index[metric_index].month_name()
                        #     chart_data = transposed_data[current_month].dropna()
                        #     positive_count = (chart_data > 0).sum()
                        #     negative_count = (chart_data < 0).sum()
                        #     st.write(f"Positive Months: {positive_count}, Negative Months: {negative_count} (from 2020)")
                        #     st.bar_chart(chart_data)
        st.markdown('</div>', unsafe_allow_html=True)           
with vcol2:
    with st.container(border=True):
        #st.header("^Above All Averages")
        st.metric("^Above All SMA", len(good_list), delta=None, delta_color="inverse", help=None, label_visibility="visible", border=False, width="stretch", height="content") # "Number of stocks with SMA 5 > 10, 10 > 20, 20 > 50 and 50 > 200"
        st.metric("^SMA 5", sma_5_count, delta=None, delta_color="normal", help="Number of stocks in S&P avoe 5 SMA", label_visibility="visible", border=False, width="stretch", height="content")
        st.metric("^SMA 5", sma_5_count, delta=None, delta_color="normal", help="Number of stocks in S&P avoe 5 SMA", label_visibility="visible", border=False, width="stretch", height="content")
        st.metric("^SMA 10", sma_10_count, delta=None, delta_color="normal", help="Number of stocks in S&P avoe 10 SMA", label_visibility="visible", border=False, width="stretch", height="content")
        st.metric("^SMA 20", sma_20_count, delta=None, delta_color="normal", help="Number of stocks in S&P avoe 20 SMA", label_visibility="visible", border=False, width="stretch", height="content")
        st.metric("^SMA 50", sma_50_count, delta=None, delta_color="normal", help="Number of stocks in S&P avoe 50 SMA", label_visibility="visible", border=False, width="stretch", height="content")
        st.metric("^SMA 200", sma_200_count, delta=None, delta_color="normal", help="Number of stocks in S&P avoe 200 SMA", label_visibility="visible", border=False, width="stretch", height="content")

with st.expander("📊 Historic Monthly Performance", expanded=True):
        # Get available months and order them chronologically
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                        'July', 'August', 'September', 'October', 'November', 'December']
        available_months = [month for month in month_order if month in transposed_data.columns and month is not None]
        default_month = monthly_performance_data.index[-2].month_name()
        default_index = available_months.index(default_month) if default_month in available_months else 0
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.write("**Select Month:**")
        with col2:
            selected_month = st.selectbox(
                "select_month",
                available_months,
                index=default_index,
                key="historic_month_selector",
                label_visibility="collapsed"
            )
        
        # selected_month = st.selectbox(
        #         "**Select Month:**",
        #         available_months,
        #         index=default_index,
        #         key="historic_month_selector",
        #         label_visibility="visible"
        #     )
        
        chart_data = transposed_data[selected_month].dropna()
        positive_count = (chart_data > 0).sum()
        negative_count = (chart_data < 0).sum()
        
        st.info(f"✅ Positive: {positive_count} | ❌ Negative: {negative_count} (since 2000)")
        
        # Beautiful plotly chart
        colors = ['#00CC96' if x > 0 else '#EF553B' for x in chart_data]
        fig = px.bar(
            x=chart_data.index, 
            y=chart_data.values, 
            color=colors,
            color_discrete_map={'#00CC96': '#00CC96', '#EF553B': '#EF553B'},
            title=f"{selected_month} Historical Performance"
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Year",
            yaxis_title="Percentage Change (%)"
        )
        st.plotly_chart(fig, use_container_width=True)


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

# st.metric(label="Revenue", value="500k", delta="20k")

# with st.popover("Click for details"):
#     st.write("Detailed revenue breakdown:")
#     st.line_chart({"Month": [1, 2, 3], "Revenue": [100, 200, 500]})
