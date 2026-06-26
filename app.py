import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import numpy as np
from fpdf import FPDF
import tempfile
import os
import plotly.express as px
import matplotlib.pyplot as plt
from PIL import Image
import textwrap

st.set_page_config(page_title="Portfolio Performance", layout="wide")
st.title("Portfolio Performance Dashboard")

# Initialize session state (Includes data migration for Delete, Name, Type, Sector, and Yield columns)
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=["Delete", "Security Name", "Type", "Sector", "Yield", "Ticker", "Amount", "Purchase Date", "Benchmark"])
else:
    if "Delete" not in st.session_state.portfolio.columns:
        st.session_state.portfolio.insert(0, "Delete", False)
    if "Security Name" not in st.session_state.portfolio.columns:
        st.session_state.portfolio.insert(1, "Security Name", "")
    if "Type" not in st.session_state.portfolio.columns:
        st.session_state.portfolio.insert(2, "Type", "")
    if "Sector" not in st.session_state.portfolio.columns:
        st.session_state.portfolio.insert(3, "Sector", "Unknown")
    if "Yield" not in st.session_state.portfolio.columns:
        st.session_state.portfolio.insert(4, "Yield", 0.0)

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
    t_upper = str(raw_type).upper()
    if 'EXCHANGE-TRADED' in t_upper or 'ETF' in t_upper: return 'ETF'
    if 'MUTUAL' in t_upper: return 'Mutual Fund'
    if 'STOCK' in t_upper or 'EQUITY' in t_upper: return 'Stock'
    return str(raw_type)

def standardize_sector(sector, ticker):
    """Applies professional nicknames and specific overrides to sector/category strings."""
    ticker_upper = str(ticker).upper()
    
    # 1. Hardcoded Overrides based on Advisor Preference
    overrides = {
        'MLPX': 'Energy',
        'GLD': 'Commodities',
        'EFEIX': 'Emerging Markets',
        'DTCR': 'Real Estate',
        'EELV': 'Emerging Markets',
        'AMZN': 'Consumer Cyclical',
        'DECK': 'Consumer Cyclical'
    }
    if ticker_upper in overrides:
        return overrides[ticker_upper]
        
    if pd.isna(sector) or not sector:
        return 'Other'
        
    # 2. General Clean-up Mappings
    mapping = {
        "Communication Services": "Communication",
        "Commodities Focused": "Commodities",
        "Energy Limited Partnership": "Energy",
        "Consumer Defensive": "Consumer Cyclical" # Consolidates defensive into cyclical
    }
    return mapping.get(sector, str(sector))

@st.cache_data
def fetch_security_details(ticker):
    """Fetches the official company name, asset type, sector, and yield from Yahoo Finance."""
    try:
        info = yf.Ticker(ticker).info
        name = info.get('shortName', info.get('longName', ticker))
        qtype = info.get('quoteType', 'Unknown')
        
        # Deep extraction to minimize "Other"
        sector = info.get('sector')
        if not sector: sector = info.get('category')
        if not sector: sector = info.get('fundCategory')
        if not sector: sector = info.get('industry')
        if not sector: sector = info.get('family')
        if not sector: sector = 'Other'
        
        # Fetch Trailing Dividend Yield
        div_yield = info.get('trailingAnnualDividendYield', info.get('dividendYield', 0.0))
        if div_yield is None: div_yield = 0.0
        
        return name, standardize_type(qtype), standardize_sector(sector, ticker), float(div_yield)
    except:
        return ticker, 'Unknown', standardize_sector('Other', ticker), 0.0

def fetch_risk_metrics(ticker, benchmark, start_date):
    """Fetches historical data to calculate True Total Return using Adj Close."""
    try:
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

def save_plotly_as_jpg(fig, width, height):
    """Saves a Plotly figure to a flat JPEG, destroying the alpha layer that turns black in FPDF."""
    tmp_png = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    tmp_jpg = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
    fig.write_image(tmp_png, format="png", engine="kaleido", width=width, height=height, scale=2)
    img = Image.open(tmp_png).convert("RGBA")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    bg.save(tmp_jpg, format="JPEG")
    os.remove(tmp_png)
    return tmp_jpg

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
                
                unique_tickers = df['Ticker'].dropna().unique()
                details_dict = {t: fetch_security_details(t) for t in unique_tickers}
                
                # Assign default Delete status
                df['Delete'] = False 
                
                if 'Security Name' not in df.columns: 
                    df['Security Name'] = df['Ticker'].map(lambda x: details_dict.get(x, (x, '', '', 0.0))[0])
                if 'Type' not in df.columns: 
                    df['Type'] = df['Ticker'].map(lambda x: details_dict.get(x, ('', 'Unknown', '', 0.0))[1])
                if 'Sector' not in df.columns: 
                    df['Sector'] = df['Ticker'].map(lambda x: details_dict.get(x, ('', '', 'Other', 0.0))[2])
                if 'Yield' not in df.columns: 
                    df['Yield'] = df['Ticker'].map(lambda x: details_dict.get(x, ('', '', '', 0.0))[3])
                
                df['Type'] = df['Type'].apply(standardize_type)
                # Re-apply standardize_sector across all rows to catch custom overrides
                df['Sector'] = df.apply(lambda row: standardize_sector(row['Sector'], row['Ticker']), axis=1)
                
                df = df[["Delete", "Security Name", "Type", "Sector", "Yield", "Ticker", "Amount", "Purchase Date"]].dropna(subset=["Ticker"])
                
                if df['Amount'].dtype == 'object':
                    df['Amount'] = df['Amount'].astype(str).str.replace(',', '').str.replace('$', '')
                df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
                
                df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], errors='coerce')
                df = df.dropna(subset=["Purchase Date"]) 
                
                df = df.groupby(['Delete', 'Security Name', 'Type', 'Sector', 'Yield', 'Ticker'], as_index=False).agg({'Amount': 'sum', 'Purchase Date': 'min'})
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
            name, qtype, sector, div_yield = fetch_security_details(new_ticker)
            new_row = pd.DataFrame({
                "Delete": [False],
                "Security Name": [name], "Type": [qtype], "Sector": [sector], "Yield": [div_yield],
                "Ticker": [new_ticker], "Amount": [new_amount],
                "Purchase Date": [new_date], "Benchmark": [get_benchmark(new_ticker)]
            })
            st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
            st.success(f"Successfully added {name}!")

# --- SECTION 2: MANAGE & EDIT PORTFOLIO ---
st.header("2. Manage Portfolio")
st.markdown("Check the **Delete** box on the left of any row and click the **Delete Selected Rows** button below to remove it.")

edited_portfolio = st.data_editor(
    st.session_state.portfolio,
    key="portfolio_table",
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Delete": st.column_config.CheckboxColumn("Delete?", default=False),
        "Type": st.column_config.TextColumn("Type", help="Hover Info: Describes whether the asset is an Equity, ETF, Mutual Fund, etc."),
        "Sector": st.column_config.TextColumn("Sector", help="The industry category of the asset."),
        "Yield": st.column_config.NumberColumn("Yield", format="%.4f"),
        "Purchase Date": st.column_config.DateColumn("Purchase Date", format="MM/DD/YYYY"),
        "Amount": st.column_config.NumberColumn("Amount", format="$%.2f")
    }
)
st.session_state.portfolio = edited_portfolio

# Custom visible buttons for deleting rows or clearing the portfolio
col_del1, col_del2 = st.columns(2)
with col_del1:
    if st.button("🗑️ Delete Selected Rows", type="primary"):
        # Keep only the rows where Delete is False
        st.session_state.portfolio = st.session_state.portfolio[st.session_state.portfolio["Delete"] == False].reset_index(drop=True)
        st.rerun()

with col_del2:
    if st.button("⚠️ Clear Entire Portfolio"):
        st.session_state.portfolio = pd.DataFrame(columns=["Delete", "Security Name", "Type", "Sector", "Yield", "Ticker", "Amount", "Purchase Date", "Benchmark"])
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
        
        display_cols = ["Security Name", "Type", "Sector", "Ticker", "Benchmark", "Amount", "Purchase Date", "Ticker Return", "Benchmark Return", "Difference"]
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
        st.subheader("📊 Portfolio Summary & Sector Allocation")
        
        total_value = calc_df['Amount'].sum()
        weights = calc_df['Amount'] / total_value
        port_weighted_return = (calc_df['Ticker Return'] * weights).sum()
        bench_weighted_return = (calc_df['Benchmark Return'] * weights).sum()
        weighted_diff = port_weighted_return - bench_weighted_return
        
        # Calculate the pure Excess Value generated/lost compared to benchmark
        excess_value = total_value * weighted_diff
        excess_str = f"+${excess_value:,.2f}" if excess_value >= 0 else f"-${abs(excess_value):,.2f}"
        
        if port_weighted_return >= bench_weighted_return:
            port_bg, port_txt = "#d4edda", "#155724" 
            bench_bg, bench_txt = "#f8d7da", "#721c24" 
        else:
            port_bg, port_txt = "#f8d7da", "#721c24" 
            bench_bg, bench_txt = "#d4edda", "#155724" 

        # Dashboard Summary Layout (Numbers on Left, Pie on Right)
        col_kpi, col_pie = st.columns([1, 1.2])
        
        with col_kpi:
            # Box 1: Total Value (Gray)
            st.markdown(f"""
            <div style="background-color: #f1f3f5; padding: 15px; border-radius: 10px; margin-bottom: 15px; border: 1px solid #dee2e6;">
                <p style="margin: 0; font-size: 1.1em; color: #495057; font-weight: bold; text-align: center;">Total Portfolio Value</p>
                <h1 style="margin: 0; font-size: 2.2em; font-weight: bold; color: #212529; text-align: center;">${total_value:,.2f}</h1>
            </div>
            """, unsafe_allow_html=True)

            # Box 2: Portfolio Return
            st.markdown(f"""
            <div style="background-color: {port_bg}; padding: 15px; border-radius: 10px; margin-bottom: 15px; border: 1px solid {port_txt};">
                <p style="margin: 0; font-size: 1.1em; color: {port_txt}; font-weight: bold; text-align: center;">Weighted Portfolio Return</p>
                <h1 style="margin: 0; font-size: 2.5em; font-weight: bold; color: {port_txt}; text-align: center;">{port_weighted_return:.2%}</h1>
            </div>
            """, unsafe_allow_html=True)

            # Box 3: Benchmark Return
            st.markdown(f"""
            <div style="background-color: {bench_bg}; padding: 15px; border-radius: 10px; margin-bottom: 15px; border: 1px solid {bench_txt};">
                <p style="margin: 0; font-size: 1.1em; color: {bench_txt}; font-weight: bold; text-align: center;">Weighted Benchmark Return</p>
                <h1 style="margin: 0; font-size: 2.2em; font-weight: bold; color: {bench_txt}; text-align: center;">{bench_weighted_return:.2%}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            # Box 4: Excess Value
            st.markdown(f"""
            <div style="background-color: {port_bg}; padding: 15px; border-radius: 10px; border: 1px solid {port_txt};">
                <p style="margin: 0; font-size: 1.1em; color: {port_txt}; font-weight: bold; text-align: center;">Excess Value Created</p>
                <h1 style="margin: 0; font-size: 2.2em; font-weight: bold; color: {port_txt}; text-align: center;">{excess_str}</h1>
            </div>
            """, unsafe_allow_html=True)
            
        with col_pie:
            # Aggregating tickers and amounts for the hover tooltip logic
            hover_data = []
            for sector, group in calc_df.groupby('Sector'):
                group = group.sort_values('Amount', ascending=False)
                lines = [f"{row['Ticker']}: ${row['Amount']:,.0f}" for _, row in group.iterrows()]
                hover_data.append({'Sector': sector, 'HoverText': '<br>'.join(lines)})
            
            hover_df = pd.DataFrame(hover_data)
            sector_df = calc_df.groupby('Sector', as_index=False)['Amount'].sum()
            sector_df = pd.merge(sector_df, hover_df, on='Sector')

            fig_pie = px.pie(sector_df, values='Amount', names='Sector', hole=0.4, custom_data=['HoverText'])
            fig_pie.update_traces(
                textposition='inside', 
                textinfo='percent+label', 
                textfont_size=16,
                hovertemplate="<b>%{label}</b><br><br>%{customdata[0]}<extra></extra>"
            )
            fig_pie.update_layout(
                margin=dict(l=20, r=20, t=20, b=20), 
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                height=500
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.write("")
        st.markdown("**Asset vs Benchmark Performance**")
        chart_df = calc_df[['Ticker', 'Ticker Return', 'Benchmark Return']].copy()
        chart_melt = chart_df.melt(id_vars='Ticker', var_name='Metric', value_name='Return')
        
        fig_bar = px.bar(chart_melt, x='Ticker', y='Return', color='Metric', barmode='group',
                         color_discrete_map={'Ticker Return': '#136207', 'Benchmark Return': '#77DD77'})
        fig_bar.update_layout(
            yaxis_tickformat='.2%', 
            margin=dict(l=80, r=20, t=20, b=40), 
            legend_title_text='',
            font=dict(size=16),
            xaxis=dict(title=""),
            yaxis=dict(title="")
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.divider()
        
        # Calculate 5th Metric (Yield)
        w_alpha = (calc_df['Alpha'] * weights).sum()
        w_beta = (calc_df['Beta'] * weights).sum()
        w_sharpe = (calc_df['Sharpe'] * weights).sum()
        w_stddev = (calc_df['Std Dev'] * weights).sum()
        w_yield = (calc_df['Yield'] * weights).sum()

        col_metrics, col_matrix = st.columns([1, 2])
        
        with col_metrics:
            st.markdown("**Risk Summary**")
            st.metric("Weighted Alpha", f"{w_alpha:.4f}")
            st.metric("Weighted Beta", f"{w_beta:.2f}")
            st.metric("Weighted Sharpe Ratio", f"{w_sharpe:.2f}")
            st.metric("Weighted Standard Deviation", f"{w_stddev:.2%}")
            st.metric("Weighted Dividend Yield", f"{w_yield:.2%}")

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
                        fig_corr.update_layout(margin=dict(l=60, r=20, t=20, b=80), font=dict(size=14))
                        st.plotly_chart(fig_corr, use_container_width=True)
                    else:
                        st.info("Add more than one position to generate a correlation matrix.")
                except:
                    st.warning("Not enough data to build correlation matrix.")

        # --- SECTION 5: REPORT GENERATION ---
        st.divider()
        st.subheader("📄 Generate Professional Landscape PDF")
        
        col_pdf_1, col_pdf_2 = st.columns(2)
        with col_pdf_1:
            client_name = st.text_input("Client Name", placeholder="e.g. Jane Doe")
            logo_upload = st.file_uploader("Upload Company Logo", type=['png', 'jpg', 'jpeg'])
            
            st.markdown("**Select Columns for Holdings Table:**")
            available_cols = ['Security Name', 'Type', 'Sector', 'Ticker', 'Bench', 'Amount', 'P. Date', 'Asset Ret', 'Bench Ret', 'Difference']
            default_cols = ['Security Name', 'Ticker', 'Bench', 'Amount', 'P. Date', 'Asset Ret', 'Bench Ret', 'Difference']
            selected_pdf_cols = st.multiselect("Columns to include:", available_cols, default=default_cols)
            
        with col_pdf_2:
            st.markdown("**Select Sections to Include:**")
            inc_summary = st.checkbox("Portfolio Summary & Sector Pie", value=True)
            inc_holdings = st.checkbox("Performance Report Table", value=True)
            inc_bar = st.checkbox("Asset vs Benchmark Bar Chart", value=True)
            inc_risk = st.checkbox("Risk Summary & Correlation Matrix", value=True)
            
        if st.button("Generate PDF", type="primary"):
            if not client_name:
                st.warning("Please enter a Client Name.")
            else:
                with st.spinner("Building Professional PDF..."):
                    
                    # Custom PDF class for automatic headers/footers
                    class ProfessionalPDF(FPDF):
                        def __init__(self, logo_path, client_name):
                            super().__init__(orientation='L', unit='mm', format='A4')
                            self.logo_path = logo_path
                            self.client_name = client_name
                            self.set_auto_page_break(auto=True, margin=15)
                            
                        def header(self):
                            if self.page_no() > 1:
                                self.set_font("Arial", "I", 10)
                                self.set_text_color(150, 150, 150)
                                self.cell(0, 8, f"Portfolio Performance Report - {self.client_name}", ln=True, align="L")
                                self.ln(2)
                                
                        def footer(self):
                            if self.page_no() > 1:
                                self.set_y(-18)
                                self.set_font("Arial", "I", 9)
                                self.set_text_color(150, 150, 150)
                                self.cell(0, 10, f"Page {self.page_no()}", align="C")
                                if self.logo_path and os.path.exists(self.logo_path):
                                    self.image(self.logo_path, x=262, y=182, w=25)

                    logo_path = None
                    if logo_upload is not None:
                        try:
                            img = Image.open(logo_upload).convert("RGB")
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                                img.save(tmp_file.name, format="JPEG")
                                logo_path = tmp_file.name
                        except Exception:
                            pass

                    pdf = ProfessionalPDF(logo_path, client_name)
                    pdf.set_margins(10, 10, 10) 
                    
                    # --- PAGE 1: DEDICATED COVER PAGE ---
                    pdf.add_page()
                    pdf.ln(50) 
                    pdf.set_font("Arial", "B", 42)
                    pdf.set_text_color(27, 79, 49) 
                    pdf.cell(0, 15, "Portfolio Performance Report", ln=True, align="C")
                    
                    pdf.ln(15)
                    pdf.set_font("Arial", "", 24)
                    pdf.set_text_color(0, 0, 0)
                    pdf.cell(0, 10, f"Prepared for: {client_name}", ln=True, align="C")
                    pdf.cell(0, 10, f"Date: {datetime.datetime.now().strftime('%B %d, %Y')}", ln=True, align="C")
                    
                    if logo_path:
                        img_w = 90
                        x_pos = (297 - img_w) / 2
                        pdf.image(logo_path, x=x_pos, y=140, w=img_w)

                    # --- PAGE 2: PORTFOLIO SUMMARY & PIE CHART ---
                    if inc_summary:
                        pdf.add_page(orientation='L')
                        pdf.set_font("Arial", "B", 26)
                        pdf.set_text_color(0, 0, 0)
                        pdf.cell(0, 15, "Portfolio Summary", ln=True, align="L")
                        pdf.ln(5)
                        
                        f_pie = None
                        try:
                            # MATPLOTLIB: The ultimate fix for headless Linux Plotly Pie bugs
                            fig, ax = plt.subplots(figsize=(9, 9), facecolor='white')
                            colors = plt.cm.Pastel1.colors
                            
                            wedges, texts, autotexts = ax.pie(
                                sector_df['Amount'], labels=sector_df['Sector'], 
                                autopct='%1.1f%%', startangle=140, colors=colors, 
                                textprops={'fontsize': 16, 'weight': 'bold', 'color': '#333333'},
                                wedgeprops={'edgecolor': 'white', 'linewidth': 2}
                            )
                            ax.axis('equal')
                            
                            tmp_pie = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
                            fig.savefig(tmp_pie, format='jpg', bbox_inches='tight', dpi=300)
                            plt.close(fig)
                            f_pie = tmp_pie
                        except Exception:
                            pass

                        if port_weighted_return >= bench_weighted_return:
                            p_fill_r, p_fill_g, p_fill_b = 212, 237, 218 
                            p_txt_r, p_txt_g, p_txt_b = 21, 87, 36
                            b_fill_r, b_fill_g, b_fill_b = 248, 215, 218 
                            b_txt_r, b_txt_g, b_txt_b = 114, 28, 36
                        else:
                            p_fill_r, p_fill_g, p_fill_b = 248, 215, 218 
                            p_txt_r, p_txt_g, p_txt_b = 114, 28, 36
                            b_fill_r, b_fill_g, b_fill_b = 212, 237, 218 
                            b_txt_r, b_txt_g, b_txt_b = 21, 87, 36
                        
                        y_start_summary = pdf.get_y()
                        pdf.set_x(15)
                        box_w = 110 
                        
                        # Box 1: Total Value
                        pdf.set_font("Arial", "B", 14)
                        pdf.set_fill_color(226, 227, 229)
                        pdf.set_text_color(56, 61, 65)
                        pdf.cell(box_w, 10, "Total Portfolio Value", border=1, align="C", fill=True)
                        pdf.ln()
                        pdf.set_x(15)
                        pdf.set_font("Arial", "B", 24)
                        pdf.cell(box_w, 20, f"${total_value:,.0f}", border=1, align="C", fill=True)
                        pdf.ln(20)
                        
                        # Box 2: Portfolio Return
                        pdf.set_x(15)
                        pdf.set_font("Arial", "B", 14)
                        pdf.set_fill_color(p_fill_r, p_fill_g, p_fill_b)
                        pdf.set_text_color(p_txt_r, p_txt_g, p_txt_b)
                        pdf.cell(box_w, 10, "Weighted Portfolio Return", border=1, align="C", fill=True)
                        pdf.ln()
                        pdf.set_x(15)
                        pdf.set_font("Arial", "B", 28)
                        pdf.cell(box_w, 22, f"{port_weighted_return:.2%}", border=1, align="C", fill=True)
                        pdf.ln(20)
                        
                        # Box 3: Benchmark Return
                        pdf.set_x(15)
                        pdf.set_font("Arial", "B", 14)
                        pdf.set_fill_color(b_fill_r, b_fill_g, b_fill_b)
                        pdf.set_text_color(b_txt_r, b_txt_g, b_txt_b)
                        pdf.cell(box_w, 10, "Weighted Benchmark Return", border=1, align="C", fill=True)
                        pdf.ln()
                        pdf.set_x(15)
                        pdf.set_font("Arial", "B", 24)
                        pdf.cell(box_w, 20, f"{bench_weighted_return:.2%}", border=1, align="C", fill=True)
                        pdf.ln(20)

                        # Box 4: Excess Value
                        pdf.set_x(15)
                        pdf.set_font("Arial", "B", 14)
                        pdf.set_fill_color(p_fill_r, p_fill_g, p_fill_b) 
                        pdf.set_text_color(p_txt_r, p_txt_g, p_txt_b)
                        pdf.cell(box_w, 10, "Excess Value Created", border=1, align="C", fill=True)
                        pdf.ln()
                        pdf.set_x(15)
                        pdf.set_font("Arial", "B", 24)
                        pdf.cell(box_w, 20, excess_str, border=1, align="C", fill=True)
                        
                        if f_pie and os.path.exists(f_pie):
                            pdf.image(f_pie, x=135, y=y_start_summary, w=145)
                            os.remove(f_pie)

                    # --- PAGE 3: PERFORMANCE REPORT HOLDINGS ---
                    if inc_holdings and len(selected_pdf_cols) > 0:
                        pdf.add_page(orientation='L')
                        pdf.set_font("Arial", "B", 26)
                        pdf.set_text_color(0, 0, 0)
                        pdf.cell(0, 15, "Performance Report", ln=True, align="L")
                        pdf.ln(5)
                        
                        base_widths = {
                            'Security Name': 65, 'Type': 22, 'Sector': 25, 'Ticker': 18, 'Bench': 18,
                            'Amount': 30, 'P. Date': 25, 'Asset Ret': 24, 
                            'Bench Ret': 24, 'Difference': 26
                        }
                        
                        col_width_map = {k: base_widths[k] for k in selected_pdf_cols}
                        x_offset = (297 - sum(col_width_map.values())) / 2
                        
                        def draw_table_headers():
                            pdf.set_fill_color(27, 79, 49) 
                            pdf.set_text_color(255, 255, 255)
                            pdf.set_font("Arial", "B", 13)
                            pdf.set_x(x_offset)
                            for col in selected_pdf_cols:
                                pdf.cell(col_width_map[col], 12, col, border=1, align='C', fill=True)
                            pdf.ln()
                            pdf.set_text_color(0, 0, 0)
                            pdf.set_font("Arial", "", 12)

                        draw_table_headers()
                        
                        fill_row = False 
                        for idx, row in calc_df.iterrows():
                            if fill_row: pdf.set_fill_color(242, 248, 242) 
                            else: pdf.set_fill_color(255, 255, 255)
                                
                            date_str = row['Purchase Date'].strftime('%m/%d/%Y') if hasattr(row['Purchase Date'], 'strftime') else str(row['Purchase Date']).split(' ')[0]
                            sec_name = str(row.get('Security Name', row['Ticker']))
                            
                            wrapped_lines = textwrap.wrap(sec_name, width=22, break_long_words=True)
                            if len(wrapped_lines) == 0: wrapped_lines = [""]
                            
                            line_height = 8
                            row_height = line_height * len(wrapped_lines)
                            
                            if pdf.get_y() + row_height > 185:
                                pdf.add_page(orientation='L')
                                draw_table_headers()
                                if fill_row: pdf.set_fill_color(242, 248, 242) 
                                else: pdf.set_fill_color(255, 255, 255)
                            
                            y_start = pdf.get_y()
                            
                            x_curr = x_offset
                            for col in selected_pdf_cols:
                                w = col_width_map[col]
                                pdf.rect(x_curr, y_start, w, row_height, 'DF')
                                x_curr += w
                                
                            x_curr = x_offset
                            for col in selected_pdf_cols:
                                w = col_width_map[col]
                                pdf.set_xy(x_curr, y_start)
                                
                                if col == 'Security Name':
                                    pdf.multi_cell(w, line_height, '\n'.join(wrapped_lines), align='C')
                                elif col == 'Type':
                                    pdf.cell(w, row_height, str(row.get('Type', '')), align='C')
                                elif col == 'Sector':
                                    pdf.cell(w, row_height, str(row.get('Sector', 'Other'))[:15], align='C')
                                elif col == 'Ticker':
                                    pdf.set_font("Arial", "B", 12)
                                    pdf.cell(w, row_height, str(row['Ticker']), align='C')
                                    pdf.set_font("Arial", "", 12)
                                elif col == 'Bench':
                                    pdf.cell(w, row_height, str(row['Benchmark']), align='C')
                                elif col == 'Amount':
                                    pdf.cell(w, row_height, f"${row['Amount']:,.2f}", align='R')
                                elif col == 'P. Date':
                                    pdf.cell(w, row_height, date_str, align='C')
                                elif col == 'Asset Ret':
                                    pdf.cell(w, row_height, f"{row['Ticker Return']:.2%}", align='R')
                                elif col == 'Bench Ret':
                                    pdf.cell(w, row_height, f"{row['Benchmark Return']:.2%}", align='R')
                                elif col == 'Difference':
                                    diff = row['Difference']
                                    pdf.set_font("Arial", "B", 12)
                                    if diff > 0.02: pdf.set_text_color(44, 160, 44) 
                                    elif diff < -0.02: pdf.set_text_color(214, 39, 40)
                                    else: pdf.set_text_color(127, 127, 127) 
                                    pdf.cell(w, row_height, f"{diff:.2%}", align='R')
                                    pdf.set_text_color(0, 0, 0)
                                    pdf.set_font("Arial", "", 12)
                                x_curr += w
                            
                            pdf.set_y(y_start + row_height)
                            fill_row = not fill_row 
                        pdf.ln(15)

                    # --- PAGE 4: BAR CHART ---
                    if inc_bar:
                        pdf.add_page(orientation='L')
                        pdf.set_font("Arial", "B", 26)
                        pdf.cell(0, 15, "Asset vs Benchmark Performance", ln=True, align="L")
                        pdf.ln(5)
                        try:
                            fig_bar_pdf = px.bar(chart_melt, x='Ticker', y='Return', color='Metric', barmode='group',
                                                 color_discrete_map={'Ticker Return': '#136207', 'Benchmark Return': '#77DD77'})
                            fig_bar_pdf.update_layout(
                                template="plotly_white",
                                yaxis_tickformat='.2%', 
                                margin=dict(l=140, r=20, t=20, b=50), 
                                legend_title_text='',
                                font=dict(size=26), 
                                xaxis=dict(title="", tickfont=dict(size=26)),
                                yaxis=dict(title="", tickfont=dict(size=26)),
                                legend=dict(font=dict(size=26), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                paper_bgcolor='rgba(255,255,255,1)',
                                plot_bgcolor='rgba(255,255,255,1)'
                            )
                            f_bar_jpg = save_plotly_as_jpg(fig_bar_pdf, 1400, 650)
                            
                            img_w = 277 
                            x_pos = 10
                            pdf.image(f_bar_jpg, x=x_pos, w=img_w)
                            os.remove(f_bar_jpg)
                        except Exception as e:
                            pdf.set_font("Arial", "", 12)
                            pdf.cell(0, 10, f"Chart could not be generated. Error details: {e}", ln=True, align="L")

                    # --- PAGE 5: RISK METRICS & CORRELATION MATRIX ---
                    if inc_risk:
                        pdf.add_page(orientation='L')
                        
                        pdf.set_font("Arial", "B", 26) 
                        pdf.cell(0, 10, "Risk Summary", ln=True, align="L")
                        pdf.ln(5)
                        
                        # Boxes exactly fitting 5 metrics
                        r_box_w = 38 
                        spacing = 4
                        total_r_w = (r_box_w * 5) + (spacing * 4)
                        x_r_start = (297 - total_r_w) / 2
                        
                        m_data = [
                            ("Weighted Alpha", f"{w_alpha:.4f}"),
                            ("Weighted Beta", f"{w_beta:.2f}"),
                            ("Weighted Sharpe", f"{w_sharpe:.2f}"),
                            ("Weighted Std Dev", f"{w_stddev:.2%}"),
                            ("Dividend Yield", f"{w_yield:.2%}")
                        ]
                        
                        y_boxes_start = pdf.get_y()
                        for title, val in m_data:
                            pdf.set_x(x_r_start)
                            pdf.set_fill_color(27, 79, 49)
                            pdf.set_text_color(255, 255, 255)
                            pdf.set_font("Arial", "B", 9) 
                            pdf.cell(r_box_w, 8, title, border=1, align='C', fill=True)
                            
                            pdf.set_xy(x_r_start, y_boxes_start + 8)
                            pdf.set_fill_color(245, 247, 245)
                            pdf.set_text_color(0, 0, 0)
                            pdf.set_font("Arial", "B", 12)
                            pdf.cell(r_box_w, 10, val, border=1, align='C', fill=True)
                            
                            x_r_start += r_box_w + spacing
                            pdf.set_y(y_boxes_start) 
                            
                        pdf.set_y(y_boxes_start + 18)
                        pdf.ln(2) # Flush padding

                        if fig_corr is not None:
                            try:
                                fig_corr_pdf = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", 
                                                         zmin=-1, zmax=1, aspect="auto", labels=dict(color="Correlation"))
                                # Expanded margins, shrunk fonts to prevent any axis label overlap
                                fig_corr_pdf.update_layout(
                                    template="plotly_white",
                                    margin=dict(l=100, r=20, t=10, b=100), 
                                    font=dict(size=12),
                                    xaxis_tickangle=-45,
                                    paper_bgcolor='rgba(255,255,255,1)',
                                    plot_bgcolor='rgba(255,255,255,1)'
                                )
                                f_corr_jpg = save_plotly_as_jpg(fig_corr_pdf, 1100, 500)
                                
                                # Massive 240mm width centered
                                img_w = 240
                                x_pos = (297 - img_w) / 2
                                current_y = pdf.get_y()
                                pdf.image(f_corr_jpg, x=x_pos, y=current_y, w=img_w)
                                os.remove(f_corr_jpg)
                            except Exception as e:
                                pdf.set_font("Arial", "", 12)
                                pdf.cell(0, 10, f"Chart could not be generated. Error details: {e}", ln=True, align="L")

                    if logo_path and os.path.exists(logo_path):
                        os.remove(logo_path)

                    pdf_output = pdf.output(dest='S')
                    pdf_bytes = pdf_output.encode('latin-1') if isinstance(pdf_output, str) else bytes(pdf_output)
                    
                    st.success("PDF generated successfully!")
                    st.download_button(
                        label="⬇️ Download Professional PDF Report",
                        data=pdf_bytes,
                        file_name=f"{client_name.replace(' ', '_')}_Portfolio_Report.pdf",
                        mime="application/pdf"
                    )
