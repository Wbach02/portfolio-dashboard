import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="Portfolio Performance", layout="wide")
st.title("Portfolio Performance Dashboard")

# Initialize session state with the Benchmark column included
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=["Ticker", "Amount", "Purchase Date", "Benchmark"])

def get_benchmark(ticker):
    """Expanded benchmark mapping based on your critique."""
    commodities = ['GLD', 'SLV', 'PDBC', 'IAU']
    intl_emerging = ['EEM', 'VWO', 'EPI', 'EFEIX']
    # Added VEU, EFV, BAESY to developed markets
    intl_developed = ['EFA', 'VEA', 'SHLD', 'CGW', 'BAESY', 'VEU', 'EFV']
    
    ticker_upper = str(ticker).upper()
    if ticker_upper in commodities:
        return 'AGG'
    elif ticker_upper in intl_emerging:
        return 'EEM'
    elif ticker_upper in intl_developed:
        return 'EFA'
    else:
        return 'SPY' 

def calculate_return(ticker, start_date):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, auto_adjust=True)
        if hist.empty:
            return None
        
        initial_price = hist['Close'].iloc[0]
        current_price = hist['Close'].iloc[-1]
        return (current_price - initial_price) / initial_price
    except Exception as e:
        return None

def apply_color_logic(val):
    if pd.isna(val) or val == "":
        return ''
    if val > 0.02:
        return 'background-color: rgba(44, 160, 44, 0.3); font-weight: bold;' 
    elif val < -0.02:
        return 'background-color: rgba(214, 39, 40, 0.3); font-weight: bold;' 
    else:
        return 'background-color: rgba(127, 127, 127, 0.3); font-weight: bold;' 

# --- SECTION 1: ADD POSITIONS ---
st.header("1. Add Positions")
col_upload, col_manual = st.columns(2)

with col_upload:
    uploaded_file = st.file_uploader("Upload Excel File (.xlsx)", type=['xlsx'])
    if uploaded_file is not None and st.button("Process Excel File"):
        try:
            df = pd.read_excel(uploaded_file)
            
            # Smart mapper to catch your specific spreadsheet column names
            column_mapping = {
                'Position': 'Ticker',
                'Position Amount': 'Amount',
                'Date of First Purchase': 'Purchase Date'
            }
            df = df.rename(columns=column_mapping)
            
            # Keep only the needed columns and drop blank rows
            df = df[["Ticker", "Amount", "Purchase Date"]].dropna(subset=["Ticker"])
            df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])
            
            # Auto-assign the benchmark guess
            df['Benchmark'] = df['Ticker'].apply(get_benchmark)
            
            st.session_state.portfolio = pd.concat([st.session_state.portfolio, df], ignore_index=True)
            st.success("Spreadsheet uploaded successfully!")
        except Exception as e:
            st.error("Error processing file. Please ensure columns match: Ticker, Amount, Purchase Date.")

with col_manual:
    with st.form("add_position_form"):
        new_ticker = st.text_input("Ticker Symbol").upper()
        new_amount = st.number_input("Amount ($)", min_value=0.0, step=100.0)
        new_date = st.date_input("Date of First Purchase", format="MM/DD/YYYY")
        
        if st.form_submit_button("Add Single Position") and new_ticker:
            new_row = pd.DataFrame({
                "Ticker": [new_ticker],
                "Amount": [new_amount],
                "Purchase Date": [new_date],
                "Benchmark": [get_benchmark(new_ticker)]
            })
            st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
            st.success(f"Added {new_ticker}!")

# --- SECTION 2: MANAGE & EDIT PORTFOLIO ---
st.header("2. Manage Portfolio")
st.markdown("Check the box on the left of any row and press **Delete** to remove it. You can also manually correct benchmarks or dates here before running the report.")

# Interactive Data Editor
edited_portfolio = st.data_editor(
    st.session_state.portfolio,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Purchase Date": st.column_config.DateColumn("Purchase Date", format="MM/DD/YYYY"),
        "Amount": st.column_config.NumberColumn("Amount", format="$%.2f")
    }
)
# Save any edits (or deletions) back to the session state immediately
st.session_state.portfolio = edited_portfolio

if st.button("Clear Entire Portfolio"):
    st.session_state.portfolio = pd.DataFrame(columns=["Ticker", "Amount", "Purchase Date", "Benchmark"])
    st.rerun()

st.divider()

# --- SECTION 3: PERFORMANCE REPORT ---
st.header("3. Performance Report")

if not st.session_state.portfolio.empty:
    if st.button("Run Performance Calculation", type="primary"):
        display_df = st.session_state.portfolio.copy()
        ticker_perfs, bench_perfs, differences = [], [], []
        
        with st.spinner('Fetching live market data from Yahoo Finance...'):
            for index, row in display_df.iterrows():
                start_d = row['Purchase Date'].strftime('%Y-%m-%d')
                
                # Pull performance using the editable benchmark column
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
        
        # Bulletproof Date Format (Zero-width space trick)
        display_df['Purchase Date'] = pd.to_datetime(display_df['Purchase Date']).dt.strftime('%m/%d/%Y') + '\u200b'
        
        format_dict = {
            'Amount': '${:,.2f}',
            'Ticker Return': '{:.2%}',
            'Benchmark Return': '{:.2%}',
            'Difference': '{:.2%}'
        }
        
        styled_df = display_df.style.map(
            apply_color_logic, subset=['Difference']
        ).map(
            lambda _: 'font-weight: bold;', subset=['Ticker']
        ).set_properties(
            **{'font-size': '110%'}
        ).format(format_dict, na_rep="Data Unavailable")
        
        st.dataframe(styled_df, use_container_width=True)
