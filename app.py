import streamlit as st
import pandas as pd
import yfinance as yf
import datetime

st.set_page_config(page_title="Portfolio Performance", layout="wide")
st.title("Portfolio Performance Dashboard")

# Initialize session state so your entries don't disappear when the page refreshes
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=["Ticker", "Amount", "Purchase Date"])

def get_benchmark(ticker):
    """Maps the entered ticker to its appropriate benchmark."""
    commodities = ['GLD', 'SLV', 'PDBC', 'IAU']
    intl_emerging = ['EEM', 'VWO', 'EPI', 'EFEIX']
    intl_developed = ['EFA', 'VEA', 'SHLD', 'CGW', 'BAESY']
    
    ticker_upper = ticker.upper()
    if ticker_upper in commodities:
        return 'AGG'
    elif ticker_upper in intl_emerging:
        return 'EEM'
    elif ticker_upper in intl_developed:
        return 'EFA'
    else:
        # Default to S&P 500 ETF for US-based equities
        return 'SPY' 

def calculate_return(ticker, start_date):
    """Pulls data from Yahoo Finance and calculates total return including dividends."""
    try:
        stock = yf.Ticker(ticker)
        # auto_adjust=True accounts for dividends and splits automatically
        hist = stock.history(start=start_date, auto_adjust=True)
        if hist.empty:
            return None
        
        initial_price = hist['Close'].iloc[0]
        current_price = hist['Close'].iloc[-1]
        return (current_price - initial_price) / initial_price
    except Exception as e:
        return None

def apply_color_logic(val):
    """Applies your +/- 2% conditional formatting rules."""
    if pd.isna(val) or val == "":
        return ''
    if val > 0.02:
        color = '#2ca02c' # Green for > +2%
    elif val < -0.02:
        color = '#d62728' # Red for < -2%
    else:
        color = '#7f7f7f' # Gray for in between
    return f'color: {color}; font-weight: bold;'

# --- Dashboard Interface ---
with st.form("add_position_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        new_ticker = st.text_input("Ticker Symbol").upper()
    with col2:
        new_amount = st.number_input("Amount ($)", min_value=0.0, step=100.0)
    with col3:
        new_date = st.date_input("Date of First Purchase")
        
    submitted = st.form_submit_button("Add Position")
    
    if submitted and new_ticker:
        new_row = pd.DataFrame({
            "Ticker": [new_ticker],
            "Amount": [new_amount],
            "Purchase Date": [new_date]
        })
        st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
        st.success(f"Successfully added {new_ticker}!")

# --- Data Processing & Display ---
if not st.session_state.portfolio.empty:
    st.subheader("Current Holdings")
    
    display_df = st.session_state.portfolio.copy()
    display_df['Benchmark'] = display_df['Ticker'].apply(get_benchmark)
    
    ticker_perfs, bench_perfs, differences = [], [], []
    
    with st.spinner('Fetching live market data...'):
        for index, row in display_df.iterrows():
            start_d = row['Purchase Date'].strftime('%Y-%m-%d')
            
            t_perf = calculate_return(row['Ticker'], start_d)
            b_perf = calculate_return(row['Benchmark'], start_d)
            
            if t_perf is not None and b_perf is not None:
                diff = t_perf - b_perf
                ticker_perfs.append(t_perf)
                bench_perfs.append(b_perf)
                differences.append(diff)
            else:
                ticker_perfs.extend([None])
                bench_perfs.extend([None])
                differences.extend([None])
                
    display_df['Ticker Return'] = ticker_perfs
    display_df['Benchmark Return'] = bench_perfs
    display_df['Difference'] = differences
    
    # Format columns for display
    format_dict = {
        'Amount': '${:,.2f}',
        'Ticker Return': '{:.2%}',
        'Benchmark Return': '{:.2%}',
        'Difference': '{:.2%}'
    }
    
    # Apply the color logic to the Difference column
    styled_df = display_df.style.map(
        apply_color_logic, subset=['Difference']
    ).format(format_dict, na_rep="Data Unavailable")
    
    st.dataframe(styled_df, use_container_width=True)
    
    if st.button("Clear Dashboard"):
        st.session_state.portfolio = pd.DataFrame(columns=["Ticker", "Amount", "Purchase Date"])
        st.rerun()