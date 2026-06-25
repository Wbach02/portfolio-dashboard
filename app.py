import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import numpy as np
from fpdf import FPDF
import tempfile
import os
import plotly.express as px
from PIL import Image
import textwrap

st.set_page_config(page_title="Portfolio Performance", layout="wide")
st.title("Portfolio Performance Dashboard")

# Initialize session state (Includes data migration for new Name & Type columns)
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=["Security Name", "Type", "Ticker", "Amount", "Purchase Date", "Benchmark"])
else:
    if "Security Name" not in st.session_state.portfolio.columns:
        st.session_state.portfolio.insert(0, "Security Name", "")
    if "Type" not in st.session_state.portfolio.columns:
        st.session_state.portfolio.insert(1, "Type", "")

def get_benchmark(ticker):
    commodities = ['GLD', 'SLV', 'PDBC', 'IAU']
    intl_emerging = ['EEM', 'VWO', 'EPI', 'EFEIX']
    intl_developed = ['EFA', 'VEA', 'SHLD', 'CGW', 'BAESY', 'VEU', 'EFV']
    
    ticker_upper = str(ticker).upper()
    if ticker_upper in commodities: return 'AGG'
    elif ticker_upper in intl_emerging: return 'EEM'
    elif ticker_upper in intl_developed: return 'EFA'
    else: return 'SPY' 

def standardize_type(raw_type):
    """Standardizes verbose asset types into clean, simple categories."""
    t_upper = str(raw_type).upper()
    if 'EXCHANGE-TRADED' in t_upper or 'ETF' in t_upper: return 'ETF'
    if 'MUTUAL' in t_upper: return 'Mutual Fund'
    if 'STOCK' in t_upper or 'EQUITY' in t_upper: return 'Stock'
    return str(raw_type)

@st.cache_data
def fetch_security_details(ticker):
    """Fetches the official company name and asset type from Yahoo Finance."""
    try:
        info = yf.Ticker(ticker).info
        name = info.get('shortName', info.get('longName', ticker))
        qtype = info.get('quoteType', 'Unknown')
        return name, standardize_type(qtype)
    except:
        return ticker, 'Unknown'

def fetch_risk_metrics(ticker, benchmark, start_date):
    """Fetches historical data to calculate True Total Return using Adj Close."""
    try:
        # STRICT ADJ CLOSE FETCH: Accounts for all splits and dividends mathematically.
        raw_data = yf.download([ticker, benchmark], start=start_date, progress=False, auto_adjust=False)
        
        if 'Adj Close' in raw_data:
            data = raw_data['Adj Close']
        elif 'Close' in raw_data:
            data = raw_data['Close']
        else:
            return None
            
        if data.empty or len(data.columns) < 2: return None
            
        t_data = data[ticker].dropna()
        b_data = data[benchmark].dropna()
        
        t_ret_total = (t_data.iloc[-1] - t_data.iloc[0]) / t_data.iloc[0]
        b_ret_total = (b_data.iloc[-1] - b_data.iloc[0]) / b_data.iloc[0]
        
        t_ret_daily = t_data.pct_change().dropna()
        b_ret_daily = b_data.pct_change().dropna()
        
        aligned = pd.concat([t_ret_daily, b_ret_daily], axis=1, join='inner').dropna()
        if aligned.empty: return None
            
        t_aligned, b_aligned = aligned.iloc[:, 0], aligned.iloc[:, 1]
        
        std_dev = t_aligned.std() * np.sqrt(252)
        correlation = t_aligned.corr(b_aligned)
        cov = t_aligned.cov(b_aligned)
        var = b_aligned.var()
        beta = cov / var if var != 0 else 1.0
        alpha = (t_aligned.mean() - (beta * b_aligned.mean())) * 252
        sharpe = (t_aligned.mean() * 252) / std_dev if std_dev != 0 else 0
        
        return {
            't_ret': t_ret_total, 'b_ret': b_ret_total,
            'alpha': alpha, 'beta': beta,
            'sharpe': sharpe, 'std_dev': std_dev,
            'correlation': correlation
        }
    except: return None

def apply_color_logic(val):
    if pd.isna(val) or val == "": return ''
    if val > 0.02: return 'background-color: rgba(44, 160, 44, 0.3); font-weight: bold;'
    elif val < -0.02: return 'background-color: rgba(214, 39, 40, 0.3); font-weight: bold;'
    else: return 'background-color: rgba(127, 127, 127, 0.3); font-weight: bold;'

# --- SECTION 1: ADD POSITIONS ---
st.header("1. Add Positions")
col_upload, col_manual = st.columns(2)

with col_upload:
    uploaded_file = st.file_uploader("Upload Excel/CSV File", type=['xlsx', 'csv'])
    if uploaded_file is not None and st.button("Process File"):
        try:
            df = pd.read_csv(uploaded_file, low_memory=False) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            
            column_mapping = {
                'Security Description': 'Security Name',
                'Security Type': 'Type',
                'Security Identifier': 'Ticker',
                'Market Value': 'Amount',
                'Trade Date': 'Purchase Date'
            }
            
            existing_cols = [col for col in column_mapping.keys() if col in df.columns]
            if not any(req in df.columns for req in ['Security Identifier', 'Market Value', 'Trade Date']):
                st.error("Could not find the required columns (Security Identifier, Market Value, Trade Date).")
            else:
                df = df.rename(columns=column_mapping)
                
                if 'Security Name' not in df.columns: df['Security Name'] = df['Ticker']
                if 'Type' not in df.columns: df['Type'] = "Unknown"
                
                df['Type'] = df['Type'].apply(standardize_type)
                df = df[["Security Name", "Type", "Ticker", "Amount", "Purchase Date"]].dropna(subset=["Ticker"])
                
                if df['Amount'].dtype == 'object':
                    df['Amount'] = df['Amount'].astype(str).str.replace(',', '').str.replace('$', '')
                df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
                
                df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], errors='coerce')
                df = df.dropna(subset=["Purchase Date"]) 
                
                df = df.groupby(['Security Name', 'Type', 'Ticker'], as_index=False).agg({'Amount': 'sum', 'Purchase Date': 'min'})
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
        
        if st.form_submit_button("Add Single Position") and new_ticker:
            name, qtype = fetch_security_details(new_ticker)
            new_row = pd.DataFrame({
                "Security Name": [name], "Type": [qtype],
                "Ticker": [new_ticker], "Amount": [new_amount],
                "Purchase Date": [new_date], "Benchmark": [get_benchmark(new_ticker)]
            })
            st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
            st.success(f"Successfully added {name}!")

# --- SECTION 2: MANAGE & EDIT PORTFOLIO ---
st.header("2. Manage Portfolio")
st.markdown("Check the box on the left of any row and press **Delete** to remove it.")

edited_portfolio = st.data_editor(
    st.session_state.portfolio,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Type": st.column_config.TextColumn("Type", help="Hover Info: Describes whether the asset is an Equity, ETF, Mutual Fund, etc."),
        "Purchase Date": st.column_config.DateColumn("Purchase Date", format="MM/DD/YYYY"),
        "Amount": st.column_config.NumberColumn("Amount", format="$%.2f")
    }
)
st.session_state.portfolio = edited_portfolio

if st.button("Clear Entire Portfolio"):
    st.session_state.portfolio = pd.DataFrame(columns=["Security Name", "Type", "Ticker", "Amount", "Purchase Date", "Benchmark"])
    st.rerun()

st.divider()

# --- SECTION 3: PERFORMANCE REPORT ---
st.header("3. Performance Report")

if not st.session_state.portfolio.empty:
    if st.button("Run Performance Calculation", type="primary"):
        display_df = st.session_state.portfolio.copy()
        metrics_list = []
        
        with st.spinner('Fetching live market data (Accurately Adjusted for Dividends & Splits)...'):
            for index, row in display_df.iterrows():
                start_d = row['Purchase Date'].strftime('%Y-%m-%d')
                metrics = fetch_risk_metrics(row['Ticker'], row['Benchmark'], start_d)
                
                if metrics:
                    metrics_list.append({
                        'Ticker Return': metrics['t_ret'], 'Benchmark Return': metrics['b_ret'],
                        'Difference': metrics['t_ret'] - metrics['b_ret'],
                        'Alpha': metrics['alpha'], 'Beta': metrics['beta'],
                        'Sharpe': metrics['sharpe'], 'Std Dev': metrics['std_dev'], 'Correlation': metrics['correlation']
                    })
                else:
                    metrics_list.append({k: None for k in ['Ticker Return', 'Benchmark Return', 'Difference', 'Alpha', 'Beta', 'Sharpe', 'Std Dev', 'Correlation']})
                    
        for col in metrics_list[0].keys():
            display_df[col] = [m[col] for m in metrics_list]
        
        st.session_state.results_df = display_df.copy()

if 'results_df' in st.session_state and st.session_state.results_df is not None:
    res = st.session_state.results_df
    calc_df = res.dropna(subset=['Ticker Return', 'Benchmark Return']).copy()
    
    if not calc_df.empty:
        
        display_cols = ["Security Name", "Type", "Ticker", "Benchmark", "Amount", "Purchase Date", "Ticker Return", "Benchmark Return", "Difference"]
        table_df = calc_df[display_cols].copy()
        table_df['Purchase Date'] = pd.to_datetime(table_df['Purchase Date']).dt.strftime('%m/%d/%Y') + '\u200b'
        
        format_dict = {'Amount': '${:,.2f}', 'Ticker Return': '{:.2%}', 'Benchmark Return': '{:.2%}', 'Difference': '{:.2%}'}
        
        styled_df = table_df.style.map(apply_color_logic, subset=['Difference']) \
            .map(lambda _: 'font-weight: bold;', subset=['Ticker']) \
            .set_properties(**{'font-size': '110%', 'white-space': 'normal'}).format(format_dict, na_rep="Data Unavailable")
            
        st.dataframe(styled_df, use_container_width=True, column_config={
            "Security Name": st.column_config.TextColumn("Security Name", width="large"),
            "Type": st.column_config.TextColumn("Type", help="Describes whether the asset is an Equity, ETF, Mutual Fund, etc.")
        })

        # --- SECTION 4: VISUAL SUMMARY & METRICS ---
        st.divider()
        st.subheader("📊 Portfolio Summary & Risk Metrics")
        
        total_value = calc_df['Amount'].sum()
        weights = calc_df['Amount'] / total_value
        port_weighted_return = (calc_df['Ticker Return'] * weights).sum()
        bench_weighted_return = (calc_df['Benchmark Return'] * weights).sum()
        weighted_diff = port_weighted_return - bench_weighted_return
        
        # DYNAMIC HIGHLIGHT LOGIC FOR THE DASHBOARD
        if port_weighted_return >= bench_weighted_return:
            bg_color = "#d4edda" # Bright Pastel Green
            text_color = "#155724" # Dark Green
        else:
            bg_color = "#f8d7da" # Bright Pastel Red
            text_color = "#721c24" # Dark Red

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"""
        <div style="background-color: #f1f3f5; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #dee2e6;">
            <p style="margin: 0; font-size: 1.2em; color: #495057;">Total Portfolio Value</p>
            <h1 style="margin: 0; font-size: 2.5em; font-weight: bold; color: #212529;">${total_value:,.2f}</h1>
        </div>
        """, unsafe_allow_html=True)

        c2.markdown(f"""
        <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid {text_color};">
            <p style="margin: 0; font-size: 1.2em; color: {text_color}; font-weight: bold;">Weighted Portfolio Return</p>
            <h1 style="margin: 0; font-size: 3em; font-weight: bold; color: {text_color};">{port_weighted_return:.2%}</h1>
        </div>
        """, unsafe_allow_html=True)

        c3.markdown(f"""
        <div style="background-color: #f1f3f5; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #dee2e6;">
            <p style="margin: 0; font-size: 1.2em; color: #495057;">Weighted Benchmark Return</p>
            <h1 style="margin: 0; font-size: 2.5em; font-weight: bold; color: #212529;">{bench_weighted_return:.2%}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        st.markdown("**Asset vs Benchmark Performance**")
        chart_df = calc_df[['Ticker', 'Ticker Return', 'Benchmark Return']].copy()
        chart_melt = chart_df.melt(id_vars='Ticker', var_name='Metric', value_name='Return')
        fig_bar = px.bar(chart_melt, x='Ticker', y='Return', color='Metric', barmode='group',
                         color_discrete_map={'Ticker Return': '#136207', 'Benchmark Return': '#77DD77'})
        fig_bar.update_layout(yaxis_tickformat='.2%', margin=dict(l=60, r=20, t=40, b=80), legend_title_text='')
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.divider()
        
        col_metrics, col_matrix = st.columns([1, 2])
        
        with col_metrics:
            st.markdown("**Weighted Portfolio Risk Metrics**")
            w_alpha = (calc_df['Alpha'] * weights).sum()
            w_beta = (calc_df['Beta'] * weights).sum()
            w_sharpe = (calc_df['Sharpe'] * weights).sum()
            w_stddev = (calc_df['Std Dev'] * weights).sum()
            
            st.metric("Weighted Alpha", f"{w_alpha:.4f}", help="Excess return of the portfolio relative to the benchmark. Positive alpha means outperformance.")
            st.metric("Weighted Beta", f"{w_beta:.2f}", help="Volatility relative to the benchmark. < 1.0 is less volatile, > 1.0 is more volatile.")
            st.metric("Weighted Sharpe Ratio", f"{w_sharpe:.2f}", help="Risk-adjusted return. How much excess return is received for the extra volatility. Higher is better.")
            st.metric("Weighted Standard Deviation", f"{w_stddev:.2%}", help="Absolute volatility/risk over the period.")

        with col_matrix:
            st.markdown("**Position Correlation Matrix**")
            fig_corr = None
            with st.spinner("Calculating correlations..."):
                unique_tickers = calc_df['Ticker'].unique().tolist()
                min_date = st.session_state.portfolio['Purchase Date'].min().strftime('%Y-%m-%d')
                try:
                    if len(unique_tickers) > 1:
                        raw_data = yf.download(unique_tickers, start=min_date, progress=False, auto_adjust=False)
                        all_data = raw_data['Adj Close'] if 'Adj Close' in raw_data else raw_data['Close']
                        
                        returns_df = all_data.pct_change()
                        corr_matrix = returns_df.corr().round(2)
                        
                        fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", 
                                             zmin=-1, zmax=1, aspect="auto", labels=dict(color="Correlation"))
                        fig_corr.update_layout(margin=dict(l=60, r=20, t=40, b=80))
                        st.plotly_chart(fig_corr, use_container_width=True)
                    else:
                        st.info("Add more than one position to generate a correlation matrix.")
                except:
                    st.warning("Not enough data to build correlation matrix.")

        # --- SECTION 5: REPORT GENERATION ---
        st.divider()
        st.subheader("📄 Generate Landscape Client PDF Report")
        
        col_pdf_1, col_pdf_2 = st.columns(2)
        with col_pdf_1:
            client_name = st.text_input("Client Name", placeholder="e.g. Jane Doe")
            logo_upload = st.file_uploader("Upload Company Logo", type=['png', 'jpg', 'jpeg'])
            
            st.markdown("**Select Columns for Holdings Table:**")
            # Added "Bench" to default columns
            available_cols = ['Security Name', 'Type', 'Ticker', 'Bench', 'Amount', 'P. Date', 'Asset Ret', 'Bench Ret', 'Difference']
            default_cols = ['Security Name', 'Ticker', 'Bench', 'Amount', 'P. Date', 'Asset Ret', 'Bench Ret', 'Difference']
            selected_pdf_cols = st.multiselect("Columns to include:", available_cols, default=default_cols)
            
        with col_pdf_2:
            st.markdown("**Select Sections to Include:**")
            inc_summary = st.checkbox("Portfolio Summary (After Holdings)", value=True)
            inc_holdings = st.checkbox("Performance Report Table", value=True)
            inc_bar = st.checkbox("Asset vs Benchmark Bar Chart", value=True)
            inc_risk = st.checkbox("Weighted Portfolio Risk Metrics", value=True)
            inc_corr = st.checkbox("Position Correlation Matrix", value=True)
            
        if st.button("Generate PDF", type="primary"):
            if not client_name:
                st.warning("Please enter a Client Name.")
            else:
                with st.spinner("Building Landscape PDF..."):
                    pdf = FPDF(orientation='L', unit='mm', format='A4')
                    pdf.set_margins(10, 10, 10) # Reduced margins for larger charts/tables
                    pdf.set_draw_color(200, 200, 200) # Light gray borders
                    
                    # --- PAGE 1: DEDICATED COVER PAGE ---
                    pdf.add_page()
                    pdf.ln(50) 
                    pdf.set_font("Arial", "B", 36)
                    pdf.cell(0, 15, "Portfolio Performance Report", ln=True, align="C")
                    
                    pdf.ln(15)
                    pdf.set_font("Arial", "", 20)
                    pdf.cell(0, 10, f"Prepared for: {client_name}", ln=True, align="C")
                    pdf.cell(0, 10, f"Date: {datetime.datetime.now().strftime('%B %d, %Y')}", ln=True, align="C")
                    
                    pdf.ln(35)
                    if logo_upload is not None:
                        try:
                            img = Image.open(logo_upload).convert("RGB")
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                                img.save(tmp_file.name, format="JPEG")
                                logo_path = tmp_file.name
                            
                            # Logo size increased 50% (from 70 to 105)
                            img_w = 105
                            x_pos = (297 - img_w) / 2
                            pdf.image(logo_path, x=x_pos, w=img_w)
                            os.remove(logo_path)
                        except Exception as e:
                            st.warning(f"Could not print logo: {e}")
                    
                    # --- PAGE 2: PERFORMANCE REPORT & SUMMARY ---
                    if inc_holdings and len(selected_pdf_cols) > 0:
                        pdf.add_page(orientation='L')
                        pdf.set_font("Arial", "B", 20)
                        pdf.cell(0, 15, "Performance Report", ln=True, align="C")
                        pdf.ln(5)
                        
                        pdf.set_fill_color(27, 79, 49) # Elegant Forest Green
                        pdf.set_text_color(255, 255, 255)
                        pdf.set_font("Arial", "B", 15) # 20% larger header font
                        
                        # Dynamically calculate total width to perfect center
                        col_width_map = {
                            'Security Name': 70, 'Type': 25, 'Ticker': 20, 'Bench': 20,
                            'Amount': 35, 'P. Date': 28, 'Asset Ret': 28, 
                            'Bench Ret': 28, 'Difference': 28
                        }
                        
                        total_table_width = sum([col_width_map[c] for c in selected_pdf_cols])
                        x_offset = (297 - total_table_width) / 2
                        
                        # Print Headers
                        pdf.set_x(x_offset)
                        for col in selected_pdf_cols:
                            pdf.cell(col_width_map[col], 12, col, border=1, align='C', fill=True)
                        pdf.ln()
                        
                        pdf.set_text_color(0, 0, 0)
                        pdf.set_font("Arial", "", 14) # 20% larger data font
                        
                        fill_row = False 
                        for idx, row in calc_df.iterrows():
                            # Zebra Striping
                            if fill_row: pdf.set_fill_color(242, 248, 242) 
                            else: pdf.set_fill_color(255, 255, 255)
                                
                            date_str = row['Purchase Date'].strftime('%m/%d/%Y') if hasattr(row['Purchase Date'], 'strftime') else str(row['Purchase Date']).split(' ')[0]
                            sec_name = str(row.get('Security Name', row['Ticker']))
                            
                            # Flawless Text Wrapping
                            wrapped_lines = textwrap.wrap(sec_name, width=28)
                            if len(wrapped_lines) == 0: wrapped_lines = [""]
                            
                            line_height = 9
                            row_height = line_height * len(wrapped_lines)
                            
                            # Boundary check - Prevents overlapping/broken bottom rows!
                            if pdf.get_y() + row_height > 190:
                                pdf.add_page(orientation='L')
                            
                            pdf.set_x(x_offset)
                            x_start = pdf.get_x()
                            y_start = pdf.get_y()
                            
                            for col in selected_pdf_cols:
                                w = col_width_map[col]
                                x_curr = pdf.get_x()
                                y_curr = pdf.get_y()
                                
                                if col == 'Security Name':
                                    pdf.multi_cell(w, line_height, '\n'.join(wrapped_lines), border=1, align='C', fill=True)
                                    pdf.set_xy(x_curr + w, y_start) # Hard reset coordinate to align row heights perfectly
                                elif col == 'Type':
                                    pdf.cell(w, row_height, str(row.get('Type', '')), border=1, align='C', fill=True)
                                elif col == 'Ticker':
                                    pdf.set_font("Arial", "B", 14)
                                    pdf.cell(w, row_height, str(row['Ticker']), border=1, align='C', fill=True)
                                    pdf.set_font("Arial", "", 14)
                                elif col == 'Bench':
                                    pdf.cell(w, row_height, str(row['Benchmark']), border=1, align='C', fill=True)
                                elif col == 'Amount':
                                    pdf.cell(w, row_height, f"${row['Amount']:,.2f}", border=1, align='R', fill=True)
                                elif col == 'P. Date':
                                    pdf.cell(w, row_height, date_str, border=1, align='C', fill=True)
                                elif col == 'Asset Ret':
                                    pdf.cell(w, row_height, f"{row['Ticker Return']:.2%}", border=1, align='R', fill=True)
                                elif col == 'Bench Ret':
                                    pdf.cell(w, row_height, f"{row['Benchmark Return']:.2%}", border=1, align='R', fill=True)
                                elif col == 'Difference':
                                    diff = row['Difference']
                                    pdf.set_font("Arial", "B", 14)
                                    if diff > 0.02: pdf.set_text_color(44, 160, 44) 
                                    elif diff < -0.02: pdf.set_text_color(214, 39, 40)
                                    else: pdf.set_text_color(127, 127, 127) 
                                    pdf.cell(w, row_height, f"{diff:.2%}", border=1, align='R', fill=True)
                                    pdf.set_text_color(0, 0, 0)
                                    pdf.set_font("Arial", "", 14)
                            
                            pdf.set_y(y_start + row_height)
                            fill_row = not fill_row 

                        pdf.ln(15)

                    # --- PORTFOLIO SUMMARY (Beneath Holdings) ---
                    if inc_summary:
                        pdf.set_font("Arial", "B", 20)
                        pdf.cell(0, 10, "Portfolio Summary", ln=True, align="C")
                        pdf.ln(5)
                        
                        # Dynamic POP Logic for PDF Colors
                        if port_weighted_return >= bench_weighted_return:
                            fill_r, fill_g, fill_b = 212, 237, 218 # Bright Light Green
                            text_r, text_g, text_b = 21, 87, 36    # Dark Green
                        else:
                            fill_r, fill_g, fill_b = 248, 215, 218 # Bright Light Red
                            text_r, text_g, text_b = 114, 28, 36   # Dark Red
                        
                        box_w = 85
                        total_summary_w = box_w * 3
                        x_offset_sum = (297 - total_summary_w) / 2
                        
                        # Header Row for Summary
                        pdf.set_x(x_offset_sum)
                        pdf.set_font("Arial", "B", 12)
                        pdf.set_fill_color(226, 227, 229)
                        pdf.set_text_color(56, 61, 65)
                        pdf.cell(box_w, 10, "Total Portfolio Value", border=1, align="C", fill=True)
                        
                        pdf.set_fill_color(fill_r, fill_g, fill_b)
                        pdf.set_text_color(text_r, text_g, text_b)
                        pdf.cell(box_w, 10, "Weighted Portfolio Return", border=1, align="C", fill=True)
                        
                        pdf.set_fill_color(226, 227, 229)
                        pdf.set_text_color(56, 61, 65)
                        pdf.cell(box_w, 10, "Weighted Benchmark Return", border=1, align="C", fill=True)
                        pdf.ln(10)
                        
                        # Value Row for Summary (Double size font, bright colors)
                        pdf.set_x(x_offset_sum)
                        pdf.set_font("Arial", "B", 26)
                        pdf.set_fill_color(226, 227, 229)
                        pdf.set_text_color(56, 61, 65)
                        pdf.cell(box_w, 20, f"${total_value:,.0f}", border=1, align="C", fill=True)
                        
                        pdf.set_fill_color(fill_r, fill_g, fill_b)
                        pdf.set_text_color(text_r, text_g, text_b)
                        pdf.cell(box_w, 20, f"{port_weighted_return:.2%}", border=1, align="C", fill=True)
                        
                        pdf.set_fill_color(226, 227, 229)
                        pdf.set_text_color(56, 61, 65)
                        pdf.cell(box_w, 20, f"{bench_weighted_return:.2%}", border=1, align="C", fill=True)
                        pdf.ln(25)

                    # --- PAGE 3: BAR CHART ---
                    if inc_bar:
                        pdf.add_page(orientation='L')
                        pdf.set_font("Arial", "B", 18)
                        pdf.cell(0, 15, "Asset vs Benchmark Performance", ln=True, align="C")
                        pdf.ln(10)
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f_bar:
                                fig_bar.write_image(f_bar.name, format="png", engine="kaleido", width=1200, height=550, scale=2)
                                
                                img_w = 260
                                x_pos = (297 - img_w) / 2
                                pdf.image(f_bar.name, x=x_pos, w=img_w)
                            os.remove(f_bar.name)
                        except Exception as e:
                            pdf.set_font("Arial", "", 12)
                            pdf.cell(0, 10, f"Chart could not be generated. Error details: {e}", ln=True, align="C")

                    # --- PAGE 4: RISK METRICS & CORRELATION MATRIX ---
                    if inc_risk or inc_corr:
                        pdf.add_page(orientation='L')
                        
                        if inc_risk:
                            pdf.set_font("Arial", "B", 18)
                            pdf.cell(0, 15, "Weighted Portfolio Risk Summary", ln=True, align="C")
                            
                            pdf.set_fill_color(27, 79, 49)
                            pdf.set_text_color(255, 255, 255)
                            pdf.set_font("Arial", "B", 14)
                            
                            m_widths = [50, 50, 50, 50]
                            m_headers = ['Weighted Alpha', 'Weighted Beta', 'Weighted Sharpe', 'Weighted Std Dev']
                            
                            pdf.set_x(48.5)
                            for i in range(len(m_headers)):
                                pdf.cell(m_widths[i], 12, m_headers[i], border=1, align='C', fill=True)
                            pdf.ln()
                            
                            pdf.set_x(48.5)
                            pdf.set_text_color(0, 0, 0)
                            pdf.set_fill_color(245, 247, 245)
                            pdf.set_font("Arial", "", 14)
                            pdf.cell(m_widths[0], 12, f"{w_alpha:.4f}", border=1, align='C', fill=True)
                            pdf.cell(m_widths[1], 12, f"{w_beta:.2f}", border=1, align='C', fill=True)
                            pdf.cell(m_widths[2], 12, f"{w_sharpe:.2f}", border=1, align='C', fill=True)
                            pdf.cell(m_widths[3], 12, f"{w_stddev:.2%}", border=1, align='C', fill=True)
                            pdf.ln(30) 

                        if inc_corr and fig_corr is not None:
                            pdf.set_font("Arial", "B", 18)
                            pdf.cell(0, 10, "Position Correlation Matrix", ln=True, align="C")
                            pdf.ln(5)
                            try:
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f_corr:
                                    fig_corr.write_image(f_corr.name, format="png", engine="kaleido", width=1000, height=550, scale=2)
                                    
                                    img_w = 200
                                    x_pos = (297 - img_w) / 2
                                    pdf.image(f_corr.name, x=x_pos, w=img_w)
                                os.remove(f_corr.name)
                            except Exception as e:
                                pdf.set_font("Arial", "", 12)
                                pdf.cell(0, 10, f"Chart could not be generated. Error details: {e}", ln=True, align="C")

                    pdf_output = pdf.output(dest='S')
                    pdf_bytes = pdf_output.encode('latin-1') if isinstance(pdf_output, str) else bytes(pdf_output)
                    
                    st.success("PDF generated successfully!")
                    st.download_button(
                        label="⬇️ Download Landscape PDF Report",
                        data=pdf_bytes,
                        file_name=f"{client_name.replace(' ', '_')}_Portfolio_Report.pdf",
                        mime="application/pdf"
                    )
