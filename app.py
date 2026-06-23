import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="Portfolio Performance", layout="wide")
st.title("Portfolio Performance Dashboard")

# Initialize persistent session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=["Ticker", "Amount", "Purchase Date", "Benchmark"])
if 'results_df' not in st.session_state:
    st.session_state.results_df = None

def get_benchmark(ticker):
    # (Existing benchmark logic remains the same)
    commodities = ['GLD', 'SLV', 'PDBC', 'IAU']
    intl_emerging = ['EEM', 'VWO', 'EPI', 'EFEIX']
    intl_developed = ['EFA', 'VEA', 'SHLD', 'CGW', 'BAESY', 'VEU', 'EFV']
    ticker_upper = str(ticker).upper()
    if ticker_upper in commodities: return 'AGG'
    elif ticker_upper in intl_emerging: return 'EEM'
    elif ticker_upper in intl_developed: return 'EFA'
    else: return 'SPY'

def calculate_return(ticker, start_date):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, auto_adjust=True)
        if hist.empty: return None
        return (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]
    except: return None

# --- SECTION 1 & 2: INPUT & MANAGEMENT ---
# (Keep your existing File Uploader and Data Editor logic here, ensuring it updates session_state.portfolio)

# --- SECTION 3: CALCULATION ENGINE ---
if st.button("Run Performance Calculation", type="primary"):
    with st.spinner('Calculating weighted returns...'):
        df = st.session_state.portfolio.copy()
        # Perform calculations and store in session state
        # ... (Insert your calculation loop here) ...
        st.session_state.results_df = display_df # Save the calculated table

# --- SECTION 4: VISUALS (Only if results exist) ---
if st.session_state.results_df is not None:
    res = st.session_state.results_df
    
    st.subheader("📊 Weighted Portfolio Comparison")
    
    # Mathematical aggregation for the Barchart
    total_val = res['Amount'].sum()
    port_wgt = (res['Amount'] * res['Ticker Return']).sum() / total_val
    bench_wgt = (res['Amount'] * res['Benchmark Return']).sum() / total_val
    
    summary_data = pd.DataFrame({
        "Return": [port_wgt, bench_wgt]
    }, index=["Portfolio (Weighted)", "Benchmark (Weighted)"])
    
    st.bar_chart(summary_data)
    
    st.divider()
    
    # Time Series (Now stays pinned!)
    selected = st.selectbox("Select Ticker for Time-Series:", res['Ticker'].unique())
    # ... (Insert your Time Series line chart logic here) ...
    
    st.dataframe(res.style.format(...), use_container_width=True)
