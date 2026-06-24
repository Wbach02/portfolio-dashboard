import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import numpy as np
from fpdf import FPDF
import tempfile
import os
import plotly.express as px

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

def fetch_risk_metrics(ticker, benchmark, start_date):
    """Fetches historical data to calculate Total Return and Risk Metrics."""
    try:
        data = yf.download([ticker, benchmark], start=start_date, progress=False)['Close']
        if data.empty or len(data.columns) < 2:
            return None
            
        t_data = data[ticker].dropna()
        b_data = data[benchmark].dropna()
        
        # Total Returns
        t_ret_total = (t_data.iloc[-1] - t_data.iloc[0]) / t_data.iloc[0]
        b_ret_total = (b_data.iloc[-1] - b_data.iloc[0]) / b_data.iloc[0]
        
        # Daily Returns for Risk Metrics
        t_ret_daily = t_data.pct_change().dropna()
        b_ret_daily = b_data.pct_change().dropna()
        
        # Align dates
        aligned = pd.concat([t_ret_daily, b_ret_daily], axis=1, join='inner').dropna()
        if aligned.empty:
            return None
            
        t_aligned = aligned.iloc[:, 0]
        b_aligned = aligned.iloc[:, 1]
        
        # Metrics Calculations
        std_dev = t_aligned.std() * np.sqrt(252)
        correlation = t_aligned.corr(b_aligned)
        cov = t_aligned.cov(b_aligned)
        var = b_aligned.var()
        beta = cov / var if var != 0 else 1.0
        alpha = (t_aligned.mean() - (beta * b_aligned.mean())) * 252
        sharpe = (t_aligned.mean() * 252) / std_dev if std_dev != 0 else 0
        
        return {
            't_ret': t_ret_total,
            'b_ret': b_ret_total,
            'alpha': alpha,
            'beta': beta,
            'sharpe': sharpe,
            'std_dev': std_dev,
            'correlation': correlation
        }
    except Exception as e:
        return None

def apply_color_logic(val):
    """Applies your +/- 2% conditional formatting rules with translucent backgrounds."""
    if pd.isna(val) or val == "":
        return ''
    if val > 0.02:
        return 'background-color: rgba(44, 160, 44, 0.3); font-weight: bold;' # Translucent Green
    elif val < -0.02:
        return 'background-color: rgba(214, 39, 40, 0.3); font-weight: bold;' # Translucent Red
    else:
        return 'background-color: rgba(127, 127, 127, 0.3); font-weight: bold;' # Translucent Gray


# --- SECTION 1: ADD POSITIONS ---
st.header("1. Add Positions")
col_upload, col_manual = st.columns(2)

with col_upload:
    uploaded_file = st.file_uploader("Upload Excel/CSV File", type=['xlsx', 'csv'])
    if uploaded_file is not None and st.button("Process File"):
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, low_memory=False)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Map the specific column names from your export
            column_mapping = {
                'Security Identifier': 'Ticker',
                'Market Value': 'Amount',
                'Trade Date': 'Purchase Date'
            }
            
            # Check if required columns exist before renaming
            existing_cols = [col for col in column_mapping.keys() if col in df.columns]
            if not existing_cols:
                st.error("Could not find the required columns (Security Identifier, Market Value, Trade Date).")
            else:
                df = df.rename(columns=column_mapping)
                
                # Keep only needed columns and drop rows without a Ticker
                df = df[["Ticker", "Amount", "Purchase Date"]].dropna(subset=["Ticker"])
                
                # Clean up data types (handles strings with commas or $ in Market Value)
                if df['Amount'].dtype == 'object':
                    df['Amount'] = df['Amount'].astype(str).str.replace(',', '').str.replace('$', '')
                df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
                
                # Clean up dates and drop rows with missing Trade Dates
                df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], errors='coerce')
                df = df.dropna(subset=["Purchase Date"]) 
                
                # Combine duplicates: Sum the total amount, and take the earliest date
                df = df.groupby('Ticker', as_index=False).agg({
                    'Amount': 'sum',
                    'Purchase Date': 'min'
                })
                
                # Auto-assign the benchmark guess
                df['Benchmark'] = df['Ticker'].apply(get_benchmark)
                
                st.session_state.portfolio = pd.concat([st.session_state.portfolio, df], ignore_index=True)
                st.success("File uploaded and consolidated successfully!")
        except Exception as e:
            st.error(f"Error processing file: {e}")

with col_manual:
    with st.form("add_position_form"):
        new_ticker = st.text_input("Ticker Symbol").upper()
        new_amount = st.number_input("Amount ($)", min_value=0.0, step=100.0)
        new_date = st.date_input("Date of First Purchase", format="MM/DD/YYYY")
        
        submitted = st.form_submit_button("Add Single Position")
        
        if submitted and new_ticker:
            new_row = pd.DataFrame({
                "Ticker": [new_ticker],
                "Amount": [new_amount],
                "Purchase Date": [new_date],
                "Benchmark": [get_benchmark(new_ticker)]
            })
            st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
            st.success(f"Successfully added {new_ticker}!")

# --- SECTION 2: MANAGE & EDIT PORTFOLIO ---
st.header("2. Manage Portfolio")
st.markdown("Check the box on the left of any row and press **Delete** to remove it. You can manually correct benchmarks or dates here before running the report.")

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
# Save edits back to session state immediately
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
        
        metrics_list = []
        
        with st.spinner('Fetching live market data and calculating risk metrics...'):
            for index, row in display_df.iterrows():
                start_d = row['Purchase Date'].strftime('%Y-%m-%d')
                
                metrics = fetch_risk_metrics(row['Ticker'], row['Benchmark'], start_d)
                
                if metrics is not None:
                    metrics_list.append({
                        'Ticker Return': metrics['t_ret'],
                        'Benchmark Return': metrics['b_ret'],
                        'Difference': metrics['t_ret'] - metrics['b_ret'],
                        'Alpha': metrics['alpha'],
                        'Beta': metrics['beta'],
                        'Sharpe': metrics['sharpe'],
                        'Std Dev': metrics['std_dev'],
                        'Correlation': metrics['correlation']
                    })
                else:
                    metrics_list.append({k: None for k in ['Ticker Return', 'Benchmark Return', 'Difference', 'Alpha', 'Beta', 'Sharpe', 'Std Dev', 'Correlation']})
                    
        # Append metrics to dataframe
        for col in metrics_list[0].keys():
            display_df[col] = [m[col] for m in metrics_list]
        
        # Save a clean version to session state with all data
        st.session_state.results_df = display_df.copy()
        
        # Format the display table without the extra metrics
        display_cols = ["Ticker", "Amount", "Purchase Date", "Benchmark", "Ticker Return", "Benchmark Return", "Difference"]
        table_df = display_df[display_cols].copy()
        
        # Format the table dates string safely
        table_df['Purchase Date'] = pd.to_datetime(table_df['Purchase Date']).dt.strftime('%m/%d/%Y') + '\u200b'
        
        format_dict = {
            'Amount': '${:,.2f}',
            'Ticker Return': '{:.2%}',
            'Benchmark Return': '{:.2%}',
            'Difference': '{:.2%}'
        }
        
        styled_df = table_df.style.map(
            apply_color_logic, subset=['Difference']
        ).map(
            lambda _: 'font-weight: bold;', subset=['Ticker']
        ).set_properties(
            **{'font-size': '110%'}
        ).format(format_dict, na_rep="Data Unavailable")
        
        st.dataframe(styled_df, use_container_width=True)

if 'results_df' in st.session_state and st.session_state.results_df is not None:
    res = st.session_state.results_df
    calc_df = res.dropna(subset=['Ticker Return', 'Benchmark Return']).copy()
    
    if not calc_df.empty:
        # --- SECTION 4: VISUAL SUMMARY & METRICS ---
        st.divider()
        st.subheader("📊 Portfolio Summary & Risk Metrics")
        
        total_value = calc_df['Amount'].sum()
        
        # Calculate KPIs
        weights = calc_df['Amount'] / total_value
        port_weighted_return = (calc_df['Ticker Return'] * weights).sum()
        bench_weighted_return = (calc_df['Benchmark Return'] * weights).sum()
        weighted_diff = port_weighted_return - bench_weighted_return
        
        # --- KPI Cards ---
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Portfolio Value", f"${total_value:,.2f}")
        with m2:
            st.metric("Weighted Portfolio Return", f"{port_weighted_return:.2%}", delta=f"{weighted_diff:.2%} vs Benchmark")
        with m3:
            st.metric("Weighted Benchmark Return", f"{bench_weighted_return:.2%}")
            
        st.write("") 

        # --- Render Performance Bar Chart ---
        st.markdown("**Asset vs Benchmark Performance**")
        chart_df = calc_df[['Ticker', 'Ticker Return', 'Benchmark Return']].set_index('Ticker')
        
        # Plot Bar Chart with Dark Green (Assets) vs Light Green (Benchmark)
        st.bar_chart(chart_df, height=400, color=["#136207", "#77DD77"])
        
        st.write("")
        st.divider()
        
        # --- Risk Metrics & Correlation Matrix Layout ---
        col_metrics, col_matrix = st.columns([1, 2])
        
        # 1. Weighted Averages
        with col_metrics:
            st.markdown("**Weighted Portfolio Risk Metrics**")
            w_alpha = (calc_df['Alpha'] * weights).sum()
            w_beta = (calc_df['Beta'] * weights).sum()
            w_sharpe = (calc_df['Sharpe'] * weights).sum()
            w_stddev = (calc_df['Std Dev'] * weights).sum()
            
            st.metric("Weighted Alpha", f"{w_alpha:.4f}", 
                      help="Measures the excess return of the portfolio relative to the benchmark. Positive alpha means outperformance.")
            st.metric("Weighted Beta", f"{w_beta:.2f}", 
                      help="Measures the portfolio's volatility relative to the benchmark. < 1.0 is less volatile, > 1.0 is more volatile.")
            st.metric("Weighted Sharpe Ratio", f"{w_sharpe:.2f}", 
                      help="Measures risk-adjusted return. Indicates how much excess return is received for the extra volatility. Higher is better.")
            st.metric("Weighted Standard Deviation", f"{w_stddev:.2%}", 
                      help="A measure of the portfolio's absolute volatility/risk over the period.")

        # 2. Correlation Matrix Heatmap
        with col_matrix:
            st.markdown("**Position Correlation Matrix**")
            with st.spinner("Calculating correlations..."):
                unique_tickers = calc_df['Ticker'].unique().tolist()
                min_date = st.session_state.portfolio['Purchase Date'].min().strftime('%Y-%m-%d')
                
                try:
                    # Fetch all historical data simultaneously for exact overlapping dates
                    if len(unique_tickers) > 1:
                        all_data = yf.download(unique_tickers, start=min_date, progress=False)['Close']
                    else:
                        all_data = pd.DataFrame(yf.download(unique_tickers[0], start=min_date, progress=False)['Close'])
                        all_data.columns = unique_tickers
                        
                    if not all_data.empty:
                        returns_df = all_data.pct_change()
                        corr_matrix = returns_df.corr().round(2)
                        
                        # Plotly Heatmap (Red=Positive, Blue=Negative)
                        fig = px.imshow(corr_matrix, 
                                        text_auto=".2f", 
                                        color_continuous_scale="RdBu_r", 
                                        zmin=-1, zmax=1,
                                        aspect="auto",
                                        labels=dict(color="Correlation"))
                        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning("Not enough data to build correlation matrix.")

        # --- SECTION 5: REPORT GENERATION ---
        st.divider()
        st.subheader("📄 Generate Client PDF Report")
        
        col_pdf_1, col_pdf_2 = st.columns(2)
        with col_pdf_1:
            client_name = st.text_input("Client Name", placeholder="e.g. Jane Doe")
            logo_upload = st.file_uploader("Upload Company Logo (PNG/JPG)", type=['png', 'jpg', 'jpeg'])
        with col_pdf_2:
            st.markdown("**Select Sections to Include:**")
            inc_holdings = st.checkbox("Include Holdings Table", value=True)
            inc_metrics = st.checkbox("Include Summary Risk Metrics", value=True)
            
        if st.button("Generate PDF", type="primary"):
            if not client_name:
                st.warning("Please enter a Client Name.")
            else:
                with st.spinner("Building PDF..."):
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_auto_page_break(auto=True, margin=15)
                    
                    # Header & Logo Integration Fix
                    if logo_upload is not None:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                            tmp_file.write(logo_upload.getvalue())
                            logo_path = tmp_file.name
                        try:
                            pdf.image(logo_path, x=10, y=8, w=30)
                        except Exception:
                            pass
                        finally:
                            os.remove(logo_path) 
                            
                    pdf.set_font("Arial", "B", 16)
                    pdf.cell(0, 10, f"Portfolio Performance Report", ln=True, align="C")
                    pdf.set_font("Arial", "", 12)
                    pdf.cell(0, 10, f"Prepared for: {client_name}", ln=True, align="C")
                    pdf.cell(0, 10, f"Date: {datetime.datetime.now().strftime('%B %d, %Y')}", ln=True, align="C")
                    pdf.ln(10)
                    
                    # KPI Summary
                    pdf.set_font("Arial", "B", 14)
                    pdf.cell(0, 10, "Portfolio Summary", ln=True)
                    pdf.set_font("Arial", "", 11)
                    pdf.cell(0, 8, f"Total Portfolio Value: ${total_value:,.2f}", ln=True)
                    pdf.cell(0, 8, f"Weighted Portfolio Return: {port_weighted_return:.2%}", ln=True)
                    pdf.cell(0, 8, f"Weighted Benchmark Return: {bench_weighted_return:.2%}", ln=True)
                    pdf.ln(10)
                    
                    # Holdings Table
                    if inc_holdings:
                        pdf.set_font("Arial", "B", 14)
                        pdf.cell(0, 10, "Current Holdings & Returns", ln=True)
                        
                        pdf.set_fill_color(19, 98, 7) # Dark Green Background
                        pdf.set_text_color(255, 255, 255) # White Text
                        pdf.set_font("Arial", "B", 10)
                        
                        col_widths = [25, 40, 35, 40, 40]
                        headers = ['Ticker', 'Amount', 'P. Date', 'Asset Ret.', 'Bench Ret.']
                        for i in range(len(headers)):
                            pdf.cell(col_widths[i], 10, headers[i], border=1, align='C', fill=True)
                        pdf.ln()
                        
                        pdf.set_text_color(0, 0, 0) # Back to Black text
                        pdf.set_font("Arial", "", 9)
                        for idx, row in calc_df.iterrows():
                            # Format date properly
                            date_str = row['Purchase Date'].strftime('%m/%d/%Y') if hasattr(row['Purchase Date'], 'strftime') else str(row['Purchase Date']).split(' ')[0]
                            pdf.cell(col_widths[0], 8, str(row['Ticker']), border=1)
                            pdf.cell(col_widths[1], 8, f"${row['Amount']:,.2f}", border=1, align='R')
                            pdf.cell(col_widths[2], 8, date_str, border=1, align='C')
                            pdf.cell(col_widths[3], 8, f"{row['Ticker Return']:.2%}", border=1, align='R')
                            pdf.cell(col_widths[4], 8, f"{row['Benchmark Return']:.2%}", border=1, align='R')
                            pdf.ln()
                        pdf.ln(10)
                        
                    # Summary Risk Metrics Table
                    if inc_metrics:
                        pdf.set_font("Arial", "B", 14)
                        pdf.cell(0, 10, "Weighted Portfolio Risk Summary", ln=True)
                        
                        pdf.set_fill_color(19, 98, 7) # Dark Green Background
                        pdf.set_text_color(255, 255, 255) # White Text
                        pdf.set_font("Arial", "B", 10)
                        
                        m_widths = [45, 45, 45, 45]
                        m_headers = ['Alpha', 'Beta', 'Sharpe Ratio', 'Std Dev']
                        for i in range(len(m_headers)):
                            pdf.cell(m_widths[i], 10, m_headers[i], border=1, align='C', fill=True)
                        pdf.ln()
                        
                        pdf.set_text_color(0, 0, 0) # Back to Black text
                        pdf.set_font("Arial", "", 10)
                        
                        pdf.cell(m_widths[0], 10, f"{w_alpha:.4f}", border=1, align='C')
                        pdf.cell(m_widths[1], 10, f"{w_beta:.2f}", border=1, align='C')
                        pdf.cell(m_widths[2], 10, f"{w_sharpe:.2f}", border=1, align='C')
                        pdf.cell(m_widths[3], 10, f"{w_stddev:.2%}", border=1, align='C')
                        pdf.ln()

                    # Output PDF bytes
                    pdf_output = pdf.output(dest='S')
                    pdf_bytes = pdf_output.encode('latin-1') if isinstance(pdf_output, str) else bytes(pdf_output)
                    
                    st.success("PDF generated successfully!")
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"{client_name.replace(' ', '_')}_Portfolio_Report.pdf",
                        mime="application/pdf"
                    )
