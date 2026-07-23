import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import numpy as np
from fpdf import FPDF
import tempfile
import os
import time
import plotly.express as px
from PIL import Image
import textwrap

st.set_page_config(page_title="Portfolio Performance", layout="wide")
st.title("Portfolio Performance Dashboard")

# Portfolio schema (no dead "Lots"/"Delete" columns)
PORTFOLIO_COLS = ["Security Name", "Type", "Sector", "Yield", "Ticker",
                  "Amount", "Total Cost", "Purchase Date", "Benchmark"]

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=PORTFOLIO_COLS)

# Per-ticker trade lots {ticker: [{'Purchase Date','Adjusted Cost','Original Cost','Current Cost','Amount'}, ...]}
# captured from the Excel report so returns respect each lot's own purchase date.
# Storing all three cost fields lets us coalesce (adjusted -> original -> current)
# so pre-2011/spinoff-reset lots with "Adjusted Cost = 0" are no longer dropped.
if 'lots' not in st.session_state:
    st.session_state.lots = {}

def get_benchmark(ticker):
    commodities = ['GLD', 'SLV', 'PDBC', 'IAU']
    intl_emerging = ['EEM', 'VWO', 'EPI', 'EFEIX', 'EELV']
    intl_developed = ['EFA', 'VEA', 'SHLD', 'CGW', 'BAESY', 'VEU', 'EFV', 'FNORX', 'EUAD']

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

    if pd.isna(sector) or not str(sector).strip():
        return 'Other'

    # 2. General Clean-up Mappings (label nicknames only -- no re-categorizing)
    mapping = {
        "Communication Services": "Communication",
        "Commodities Focused": "Commodities",
        "Energy Limited Partnership": "Energy",
    }
    return mapping.get(str(sector).strip(), str(sector).strip())

def _coalesce_cost(lot):
    """Return the best available cost basis for a lot: adjusted -> original -> current."""
    for key in ('Adjusted Cost', 'Original Cost', 'Current Cost', 'Total Cost'):
        v = pd.to_numeric(lot.get(key), errors='coerce')
        if pd.notna(v) and v > 0:
            return float(v)
    return 0.0

@st.cache_data(ttl=3600)
def fetch_security_details(ticker):
    """Fallback lookup (Yahoo Finance) used ONLY when the uploaded file is missing a field."""
    try:
        info = yf.Ticker(ticker).info
        name = info.get('shortName', info.get('longName', ticker))
        qtype = info.get('quoteType', 'Unknown')

        sector = info.get('sector')
        if not sector: sector = info.get('category')
        if not sector: sector = info.get('fundCategory')
        if not sector: sector = info.get('industry')
        if not sector: sector = info.get('family')
        if not sector: sector = 'Other'

        div_yield = info.get('trailingAnnualDividendYield', info.get('dividendYield', 0.0))
        if div_yield is None: div_yield = 0.0
        div_yield = float(div_yield)
        # Yahoo usually returns a decimal fraction (0.0234). Normalize to a
        # percentage NUMBER (2.34) so it matches the Excel "Current Yield" column.
        if 0 < div_yield < 1:
            div_yield *= 100.0

        return name, standardize_type(qtype), standardize_sector(sector, ticker), div_yield
    except Exception:
        return ticker, 'Unknown', standardize_sector('Other', ticker), 0.0

@st.cache_data(ttl=3600)
def get_risk_free_rate():
    """Current US 10-Year Treasury Note yield (^TNX) from Yahoo Finance, returned as a decimal.
    Defensively flattens the MultiIndex that recent yfinance versions return and sanity-checks
    the magnitude so a bad feed can never silently poison Alpha/Sharpe with a 0% rate."""
    try:
        data = yf.download("^TNX", period="5d", progress=False, auto_adjust=False)
        if data is None or data.empty:
            return 0.0
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        close = data.get('Close')
        if close is None:
            return 0.0
        series = close.dropna()
        if series.empty:
            return 0.0
        latest = float(series.iloc[-1])
        # ^TNX is quoted in percent (e.g. 4.25 = 4.25%); convert to a decimal fraction.
        # Sanity guard against feeds that return a scaled integer.
        if latest > 20:
            latest /= 10.0
        return latest / 100.0
    except Exception:
        return 0.0

def fetch_risk_metrics(ticker, benchmark, start_date, risk_free_rate=0.0, lots=None):
    try:
        # Handle the case where the ticker IS its own benchmark (e.g. SPY -> SPY).
        # yf.download deduplicates identical tickers into a single column, which
        # would otherwise trip the `< 2 columns` guard and drop the position.
        same = str(ticker).upper() == str(benchmark).upper()

        if same:
            raw_data = yf.download(ticker, start=start_date, progress=False, auto_adjust=False)
            if raw_data is None or raw_data.empty:
                return None
            if isinstance(raw_data.columns, pd.MultiIndex):
                raw_data.columns = raw_data.columns.get_level_values(0)
            px_series = raw_data.get('Adj Close', raw_data.get('Close'))
            if px_series is None:
                return None
            t_data = b_data = px_series.dropna()
        else:
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

        if t_data.empty or b_data.empty: return None

        # Total return since purchase (dividend/split/cap-gain adjusted via Adj Close).
        # When the Excel report has multiple trade lots for this ticker, compute a
        # COST-WEIGHTED return where each lot is measured from its OWN purchase date,
        # and value the benchmark as if the same dollars were invested on the same dates.
        if lots:
            t_cost_wsum = 0.0
            b_cost_wsum = 0.0
            tot_cost = 0.0
            for lot in lots:
                c = _coalesce_cost(lot)   # adjusted -> original -> current
                d = pd.to_datetime(lot.get('Purchase Date'), errors='coerce')
                if c <= 0 or pd.isna(d):
                    continue
                t_lot = t_data.loc[t_data.index >= d]
                b_lot = b_data.loc[b_data.index >= d]
                if t_lot.empty or b_lot.empty:
                    continue
                r_t = (t_lot.iloc[-1] - t_lot.iloc[0]) / t_lot.iloc[0]
                r_b = (b_lot.iloc[-1] - b_lot.iloc[0]) / b_lot.iloc[0]
                t_cost_wsum += c * r_t
                b_cost_wsum += c * r_b
                tot_cost += c

            if tot_cost > 0:
                t_ret_total = t_cost_wsum / tot_cost
                b_ret_total = b_cost_wsum / tot_cost
            else:
                t_ret_total = (t_data.iloc[-1] - t_data.iloc[0]) / t_data.iloc[0]
                b_ret_total = (b_data.iloc[-1] - b_data.iloc[0]) / b_data.iloc[0]
        else:
            # Single-lot / manual entry: measure from the one purchase date.
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
        # Jensen's alpha (annualized), risk-free rate included for consistency with Sharpe
        rf_daily = risk_free_rate / 252.0
        alpha = ((t_aligned.mean() - rf_daily) - beta * (b_aligned.mean() - rf_daily)) * 252
        # Annualized Sharpe using the US 10-Year Treasury as the risk-free rate
        sharpe = ((t_aligned.mean() * 252) - risk_free_rate) / std_dev if std_dev != 0 else 0

        return {
            't_ret': t_ret_total, 'b_ret': b_ret_total,
            'alpha': alpha, 'beta': beta,
            'sharpe': sharpe, 'std_dev': std_dev,
            'correlation': correlation
        }
    except Exception:
        return None

def apply_color_logic(val):
    if pd.isna(val) or val == "": return ''
    if val > 0.02: return 'background-color: rgba(44, 160, 44, 0.3); font-weight: bold;'
    elif val < -0.02: return 'background-color: rgba(214, 39, 40, 0.3); font-weight: bold;'
    else: return 'background-color: rgba(127, 127, 127, 0.3); font-weight: bold;'

def save_plotly_as_jpg(fig, width, height):
    """Saves a Plotly figure to a flat JPEG, with robust retry logic for Kaleido engine crashes."""
    tmp_png = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    tmp_jpg = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name

    success = False
    last_error = None

    for attempt in range(3):
        try:
            fig.write_image(tmp_png, format="png", width=width, height=height, scale=2)
            success = True
            break
        except Exception as e:
            last_error = e
            time.sleep(1.5)

    if not success:
        raise last_error

    img = Image.open(tmp_png).convert("RGBA")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    bg.save(tmp_jpg, format="JPEG", quality=95)
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

            # Keep the three Pershing cost fields distinct so we can coalesce later.
            # Original Total Cost is the *acquisition* dollar amount (never zero for a real lot);
            # Original Adjusted Cost may be zero on noncovered/pre-2011/spinoff-reset lots;
            # Current Total Cost is the running adjusted basis after partial sales.
            column_mapping = {
                'Security Description':   'Security Name',
                'Security Type':          'Type',
                'Security Identifier':    'Ticker',
                'Market Value':           'Amount',
                'Original Total Cost':    'Original Cost',
                'Original Adjusted Cost': 'Adjusted Cost',
                'Current Total Cost':     'Current Cost',
                'Trade Date':             'Purchase Date',
                'Asset Category':         'Sector',       # Column Z -> Sector
                'Current Yield':          'Yield',        # Column V -> Yield (dividends)
                'Yield':                  'Yield'
            }

            if not any(req in df.columns for req in ['Security Identifier', 'Market Value', 'Trade Date']):
                st.error("Could not find the required columns (Security Identifier, Market Value, Trade Date).")
            else:
                df = df.rename(columns=column_mapping)

                # Drop Pershing's "Multiple" aggregate rows so we don't double-count
                # a multi-lot ticker (once as an aggregate, again as its per-lot rows).
                if 'Taxlot Category' in df.columns:
                    df = df[df['Taxlot Category'].astype(str).str.upper() != 'MULTIPLE']

                unique_tickers = df['Ticker'].dropna().unique()

                # EFFICIENCY: only call Yahoo Finance when the Excel is missing a metadata column.
                # Sector comes from "Asset Category" and Yield from "Current Yield" in the file itself.
                meta_cols = ['Security Name', 'Type', 'Sector', 'Yield']
                have_all_meta = all(c in df.columns for c in meta_cols)
                details_dict = {} if have_all_meta else {t: fetch_security_details(t) for t in unique_tickers}

                if 'Security Name' not in df.columns:
                    df['Security Name'] = df['Ticker'].map(lambda x: details_dict.get(x, (x, '', '', 0.0))[0])
                if 'Type' not in df.columns:
                    df['Type'] = df['Ticker'].map(lambda x: details_dict.get(x, ('', 'Unknown', '', 0.0))[1])
                if 'Sector' not in df.columns:
                    df['Sector'] = df['Ticker'].map(lambda x: details_dict.get(x, ('', '', 'Other', 0.0))[2])
                if 'Yield' not in df.columns:
                    df['Yield'] = df['Ticker'].map(lambda x: details_dict.get(x, ('', '', '', 0.0))[3])

                # Ensure every cost field exists so downstream coalescing is safe.
                for c in ('Adjusted Cost', 'Original Cost', 'Current Cost'):
                    if c not in df.columns:
                        df[c] = 0.0

                df['Type'] = df['Type'].apply(standardize_type)
                df['Sector'] = df.apply(lambda row: standardize_sector(row['Sector'], row['Ticker']), axis=1)

                df = df[["Security Name", "Type", "Sector", "Yield", "Ticker", "Amount",
                        "Adjusted Cost", "Original Cost", "Current Cost", "Purchase Date"]].dropna(subset=["Ticker"])

                if df['Amount'].dtype == 'object':
                    df['Amount'] = df['Amount'].astype(str).str.replace(',', '').str.replace('$', '')
                df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)

                for cost_col in ('Adjusted Cost', 'Original Cost', 'Current Cost'):
                    if df[cost_col].dtype == 'object':
                        df[cost_col] = df[cost_col].astype(str).str.replace(',', '').str.replace('$', '')
                    df[cost_col] = pd.to_numeric(df[cost_col], errors='coerce').fillna(0.0)

                if df['Yield'].dtype == 'object':
                    df['Yield'] = df['Yield'].astype(str).str.replace('%', '').str.replace(',', '')
                df['Yield'] = pd.to_numeric(df['Yield'], errors='coerce').fillna(0.0)

                df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], errors='coerce')
                df = df.dropna(subset=["Purchase Date"])

                # --- CONSOLIDATE BY TICKER ONLY ---
                # Guarantees each position shows up exactly once, sums every lot's
                # Amount and Total Cost, and returns the single EARLIEST trade date.
                df['Ticker'] = df['Ticker'].astype(str).str.strip().str.upper()

                # Row-level coalesced cost for portfolio-level display and consolidation.
                df['Total Cost'] = df.apply(lambda r: _coalesce_cost(r.to_dict()), axis=1)

                # Preserve each individual trade lot (date + costs) BEFORE consolidating,
                # so return calculations can weight each lot from its own purchase date.
                # Reset any prior lot list for tickers in this upload to prevent
                # duplicate-lot corruption on re-uploads of the same file.
                for tkr in df['Ticker'].unique():
                    st.session_state.lots[tkr] = []
                for tkr, grp in df.groupby('Ticker'):
                    lot_list = st.session_state.lots[tkr]
                    for _, r in grp.iterrows():
                        lot_list.append({
                            'Purchase Date': pd.to_datetime(r['Purchase Date'], errors='coerce'),
                            'Adjusted Cost': float(r['Adjusted Cost']) if pd.notna(r['Adjusted Cost']) else 0.0,
                            'Original Cost': float(r['Original Cost']) if pd.notna(r['Original Cost']) else 0.0,
                            'Current Cost':  float(r['Current Cost'])  if pd.notna(r['Current Cost'])  else 0.0,
                            'Amount':        float(r['Amount'])        if pd.notna(r['Amount'])        else 0.0,
                        })

                df = df.groupby('Ticker', as_index=False).agg(
                    **{
                        'Security Name': ('Security Name', 'first'),
                        'Type':          ('Type', 'first'),
                        'Sector':        ('Sector', 'first'),
                        'Yield':         ('Yield', 'first'),
                        'Amount':        ('Amount', 'sum'),         # total of all trades
                        'Total Cost':    ('Total Cost', 'sum'),     # coalesced cost across lots
                        'Purchase Date': ('Purchase Date', 'min'),  # earliest purchase date
                    }
                )
                df = df[["Security Name", "Type", "Sector", "Yield", "Ticker", "Amount", "Total Cost", "Purchase Date"]]
                df['Benchmark'] = df['Ticker'].apply(get_benchmark)

                # Merge with anything already in the portfolio, then re-consolidate so
                # a re-upload of the same ticker still shows only one row.
                combined = pd.concat([st.session_state.portfolio, df], ignore_index=True)
                combined['Ticker'] = combined['Ticker'].astype(str).str.strip().str.upper()
                combined['Purchase Date'] = pd.to_datetime(combined['Purchase Date'], errors='coerce')
                combined = combined.dropna(subset=["Ticker"])
                combined = combined.groupby('Ticker', as_index=False).agg(
                    **{
                        'Security Name': ('Security Name', 'first'),
                        'Type':          ('Type', 'first'),
                        'Sector':        ('Sector', 'first'),
                        'Yield':         ('Yield', 'first'),
                        'Amount':        ('Amount', 'sum'),
                        'Total Cost':    ('Total Cost', 'sum'),
                        'Purchase Date': ('Purchase Date', 'min'),
                        'Benchmark':     ('Benchmark', 'first'),
                    }
                )
                st.session_state.portfolio = combined[PORTFOLIO_COLS]
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
            tkr_norm = str(new_ticker).strip().upper()
            st.session_state.lots.setdefault(tkr_norm, []).append({
                'Purchase Date': pd.to_datetime(new_date),
                'Adjusted Cost': float(new_amount),
                'Original Cost': float(new_amount),
                'Current Cost':  float(new_amount),
                'Amount':        float(new_amount),
            })
            new_row = pd.DataFrame({
                "Security Name": [name], "Type": [qtype], "Sector": [sector], "Yield": [div_yield],
                "Ticker": [new_ticker], "Amount": [new_amount], "Total Cost": [new_amount],
                "Purchase Date": [pd.to_datetime(new_date)], "Benchmark": [get_benchmark(new_ticker)]
            })
            st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
            st.success(f"Successfully added {name}!")

# --- SECTION 2: MANAGE & EDIT PORTFOLIO ---
st.header("2. Manage Portfolio")
st.markdown("To delete a row, click the box on the far left edge of the table to select it, then press the **Delete** key on your keyboard or click the trash can icon in the top right of the table.")

edited_portfolio = st.data_editor(
    st.session_state.portfolio,
    key="portfolio_table",
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Type": st.column_config.TextColumn("Type", help="Hover Info: Describes whether the asset is an Equity, ETF, Mutual Fund, etc."),
        "Sector": st.column_config.TextColumn("Sector", help="The industry category of the asset."),
        "Yield": st.column_config.NumberColumn("Yield", format="%.4f"),
        "Purchase Date": st.column_config.DateColumn("Purchase Date", format="MM/DD/YYYY"),
        "Amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
        "Total Cost": st.column_config.NumberColumn("Total Cost", format="$%.2f")
    }
)
st.session_state.portfolio = edited_portfolio

if st.button("⚠️ Clear Entire Portfolio"):
    st.session_state.portfolio = pd.DataFrame(columns=PORTFOLIO_COLS)
    st.session_state.lots = {}
    st.rerun()

st.divider()

# --- SECTION 3: PERFORMANCE REPORT ---
st.header("3. Performance Report")

if not st.session_state.portfolio.empty:
    if st.button("Run Performance Calculation", type="primary"):
        display_df = st.session_state.portfolio.copy()
        display_df['Purchase Date'] = pd.to_datetime(display_df['Purchase Date'], errors='coerce')
        metrics_list = []

        risk_free_rate = get_risk_free_rate()

        with st.spinner('Fetching live market data (Accurately Adjusted for Dividends & Splits)...'):
            for index, row in display_df.iterrows():
                start_d = row['Purchase Date'].strftime('%Y-%m-%d')
                lots = st.session_state.lots.get(str(row['Ticker']).strip().upper())
                metrics = fetch_risk_metrics(row['Ticker'], row['Benchmark'], start_d, risk_free_rate=risk_free_rate, lots=lots)

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

        # Ensure all strictly numeric columns are strictly typed to prevent object dtype errors like nlargest failures
        numeric_cols = ['Amount', 'Total Cost', 'Ticker Return', 'Benchmark Return', 'Difference', 'Alpha', 'Beta', 'Sharpe', 'Std Dev', 'Correlation', 'Yield']
        for col in numeric_cols:
            if col in calc_df.columns:
                calc_df[col] = pd.to_numeric(calc_df[col], errors='coerce')

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

        # --- SECTION: TOP CONTRIBUTORS & DETRACTORS ---
        st.write("")
        st.subheader("Top Contributors and Detractors")

        # Excess Value uses cost basis (not current market value) so the sum reconciles
        # with what the client actually put in vs. what a same-dollar benchmark investment
        # would have grown to.
        cost_for_excess = calc_df['Total Cost'].where(calc_df['Total Cost'] > 0, calc_df['Amount'])
        calc_df['Excess Value'] = cost_for_excess * calc_df['Difference']
        top_contribs = calc_df[calc_df['Excess Value'] > 0].nlargest(3, 'Excess Value')
        top_detracts = calc_df[calc_df['Excess Value'] < 0].nsmallest(3, 'Excess Value')

        col_c, col_d = st.columns(2)

        with col_c:
            st.markdown('**Top Contributors (Positive Excess Value)**')
            if not top_contribs.empty:
                for _, row in top_contribs.iterrows():
                    st.markdown(f"""
                    <div style="background-color: rgba(44, 160, 44, 0.1); padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 6px solid #2ca02c;">
                        <span style="font-size: 1.1em; font-weight: bold; color: #212529;">{row['Ticker']}</span> <span style="color: #495057;">- {row['Security Name']}</span><br>
                        <div style="margin-top: 5px; font-size: 1.05em;">
                            Excess Return: <span style="color: #155724; font-weight: bold;">{row['Difference']:.2%}</span> |
                            Excess Value: <span style="color: #155724; font-weight: bold;">+${row['Excess Value']:,.2f}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No positive contributors found.")

        with col_d:
            st.markdown('**Top Detractors (Negative Excess Value)**')
            if not top_detracts.empty:
                for _, row in top_detracts.iterrows():
                    st.markdown(f"""
                    <div style="background-color: rgba(214, 39, 40, 0.1); padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 6px solid #d62728;">
                        <span style="font-size: 1.1em; font-weight: bold; color: #212529;">{row['Ticker']}</span> <span style="color: #495057;">- {row['Security Name']}</span><br>
                        <div style="margin-top: 5px; font-size: 1.05em;">
                            Excess Return: <span style="color: #721c24; font-weight: bold;">{row['Difference']:.2%}</span> |
                            Excess Value: <span style="color: #721c24; font-weight: bold;">-${abs(row['Excess Value']):,.2f}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No negative detractors found.")

        # --- SECTION 4: VISUAL SUMMARY & METRICS ---
        st.divider()
        st.subheader("📊 Portfolio Summary & Sector Allocation")

        total_value = calc_df['Amount'].sum()
        # Cost-basis weights for the return figures so the weighted portfolio return
        # reconciles with Excess Value / Total Cost. Falls back to market value when a
        # position has no cost basis (e.g. manually added positions with 0 cost).
        cost_basis = calc_df['Total Cost'].where(calc_df['Total Cost'] > 0, calc_df['Amount'])
        total_cost = cost_basis.sum()
        w_cost = cost_basis / total_cost if total_cost > 0 else calc_df['Amount'] / total_value
        # Market-value weights for allocation/risk figures (accurate current exposure).
        weights = calc_df['Amount'] / total_value

        port_weighted_return = (calc_df['Ticker Return'] * w_cost).sum()
        bench_weighted_return = (calc_df['Benchmark Return'] * w_cost).sum()

        excess_value = calc_df['Excess Value'].sum()
        excess_str = f"+${excess_value:,.2f}" if excess_value >= 0 else f"-${abs(excess_value):,.2f}"

        if port_weighted_return >= bench_weighted_return:
            port_bg, port_txt = "#d4edda", "#155724"
            bench_bg, bench_txt = "#f8d7da", "#721c24"
        else:
            port_bg, port_txt = "#f8d7da", "#721c24"
            bench_bg, bench_txt = "#d4edda", "#155724"

        col_kpi, col_pie = st.columns([1, 1.2])

        with col_kpi:
            st.markdown(f"""
            <div style="background-color: #f1f3f5; padding: 15px; border-radius: 10px; margin-bottom: 15px; border: 1px solid #dee2e6;">
                <p style="margin: 0; font-size: 1.1em; color: #495057; font-weight: bold; text-align: center;">Total Portfolio Value</p>
                <h1 style="margin: 0; font-size: 2.2em; font-weight: bold; color: #212529; text-align: center;">${total_value:,.2f}</h1>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background-color: {port_bg}; padding: 15px; border-radius: 10px; margin-bottom: 15px; border: 1px solid {port_txt};">
                <p style="margin: 0; font-size: 1.1em; color: {port_txt}; font-weight: bold; text-align: center;">Weighted Portfolio Return</p>
                <h1 style="margin: 0; font-size: 2.5em; font-weight: bold; color: {port_txt}; text-align: center;">{port_weighted_return:.2%}</h1>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background-color: {bench_bg}; padding: 15px; border-radius: 10px; margin-bottom: 15px; border: 1px solid {bench_txt};">
                <p style="margin: 0; font-size: 1.1em; color: {bench_txt}; font-weight: bold; text-align: center;">Weighted Benchmark Return</p>
                <h1 style="margin: 0; font-size: 2.2em; font-weight: bold; color: {bench_txt}; text-align: center;">{bench_weighted_return:.2%}</h1>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background-color: {port_bg}; padding: 15px; border-radius: 10px; border: 1px solid {port_txt};">
                <p style="margin: 0; font-size: 1.1em; color: {port_txt}; font-weight: bold; text-align: center;">Excess Value Created <span title="How much total excess value have my currently held assets generated since I bought each of them?" style="cursor: help; font-size: 0.9em; opacity: 0.8;">&#9432;</span></p>
                <h1 style="margin: 0; font-size: 2.2em; font-weight: bold; color: {port_txt}; text-align: center;">{excess_str}</h1>
            </div>
            """, unsafe_allow_html=True)

        with col_pie:
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
        n_bars_dash = chart_melt['Ticker'].nunique()
        dash_tick_font = 16 if n_bars_dash <= 15 else 12
        dash_angle = 0 if n_bars_dash <= 12 else -45
        fig_bar.update_layout(
            yaxis_tickformat='.2%',
            margin=dict(l=80, r=20, t=20, b=40),
            legend_title_text='',
            font=dict(size=16),
            xaxis=dict(title="", tickfont=dict(size=dash_tick_font), tickangle=dash_angle),
            yaxis=dict(title="")
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()

        w_alpha = (calc_df['Alpha'] * weights).sum()
        w_beta = (calc_df['Beta'] * weights).sum()
        w_yield = (calc_df['Yield'] * weights).sum()

        # --- TRUE portfolio-level risk via the covariance matrix (sigma_p = sqrt(wT * Cov * w)).
        #     This accounts for diversification/correlation between holdings instead of
        #     just averaging each position's individual volatility. ---
        rf_rate = get_risk_free_rate()
        portfolio_returns_df = None            # reused for the correlation matrix below
        w_stddev = (calc_df['Std Dev'] * weights).sum()   # fallback if the download fails
        w_sharpe = (calc_df['Sharpe'] * weights).sum()    # fallback if the download fails
        try:
            risk_tickers = calc_df['Ticker'].tolist()
            risk_weights = calc_df.set_index('Ticker')['Amount'] / calc_df['Amount'].sum()
            risk_min_date = pd.to_datetime(st.session_state.portfolio['Purchase Date'], errors='coerce').min().strftime('%Y-%m-%d')
            if len(risk_tickers) > 1:
                raw_px = yf.download(risk_tickers, start=risk_min_date, progress=False, auto_adjust=False)
                px_data = raw_px['Adj Close'] if 'Adj Close' in raw_px else raw_px['Close']
                portfolio_returns_df = px_data.pct_change()
                # Common window where every current holding has data -> weights sum to 1
                daily = portfolio_returns_df.reindex(columns=risk_tickers).dropna()
                wv = risk_weights.reindex(risk_tickers).fillna(0.0)
                if wv.sum() > 0:
                    wv = wv / wv.sum()
                if len(daily) >= 2:
                    cov_annual = daily.cov() * 252
                    wvec = wv.reindex(cov_annual.columns).fillna(0.0).values
                    port_var = float(wvec @ cov_annual.values @ wvec)
                    if port_var > 0:
                        w_stddev = np.sqrt(port_var)
                    port_ann_ret = float((daily.mean() * 252).reindex(cov_annual.columns).fillna(0.0).values @ wvec)
                    w_sharpe = (port_ann_ret - rf_rate) / w_stddev if w_stddev != 0 else 0
            else:
                # Single holding: portfolio risk == that position's risk
                w_stddev = float(calc_df['Std Dev'].iloc[0])
                w_sharpe = float(calc_df['Sharpe'].iloc[0])
        except Exception:
            pass

        col_metrics, col_matrix = st.columns([1, 2])

        with col_metrics:
            st.markdown("**Risk Summary**")
            st.metric("Weighted Alpha", f"{w_alpha:.2%}", help="Jensen's alpha vs. the benchmark, net of the risk-free rate. Positive means outperformance.")
            st.metric("Weighted Beta", f"{w_beta:.2f}", help="Volatility relative to the benchmark. < 1.0 is less volatile, > 1.0 is more volatile.")
            st.metric("Weighted Sharpe Ratio", f"{w_sharpe:.2f}", help="Portfolio-level risk-adjusted return using the covariance matrix and the US 10-Year Treasury (^TNX) as the risk-free rate. Higher is better.")
            st.metric("Portfolio Standard Deviation", f"{w_stddev:.2%}", help="True portfolio volatility from the covariance matrix (accounts for diversification across holdings), not a simple average of each position's risk.")
            st.metric("Weighted Dividend Yield", f"{w_yield / 100.0:.2%}", help="The weighted average trailing 12-month dividend yield of the portfolio.")
            st.caption(f"Risk-free rate (US 10-Yr Treasury, ^TNX): {rf_rate:.2%}")

        with col_matrix:
            st.markdown("**Position Correlation Matrix**")
            fig_corr = None
            corr_matrix = None
            with st.spinner("Calculating correlations..."):
                unique_tickers = calc_df['Ticker'].unique().tolist()
                min_date = pd.to_datetime(st.session_state.portfolio['Purchase Date'], errors='coerce').min().strftime('%Y-%m-%d')
                try:
                    if len(unique_tickers) > 1:
                        if portfolio_returns_df is not None:
                            returns_df = portfolio_returns_df
                        else:
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
                except Exception:
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
                            fig_pie_pdf = px.pie(sector_df, values='Amount', names='Sector', color_discrete_sequence=px.colors.qualitative.Plotly)
                            fig_pie_pdf.update_traces(textposition='inside', textinfo='percent+label', textfont_size=24, marker=dict(line=dict(color='#FFFFFF', width=2)))
                            fig_pie_pdf.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor='white', plot_bgcolor='white')
                            f_pie = save_plotly_as_jpg(fig_pie_pdf, 800, 800)
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

                        pdf.set_font("Arial", "B", 14)
                        pdf.set_fill_color(226, 227, 229)
                        pdf.set_text_color(56, 61, 65)
                        pdf.cell(box_w, 10, "Total Portfolio Value", border=1, align="C", fill=True)
                        pdf.ln()
                        pdf.set_x(15)
                        pdf.set_font("Arial", "B", 24)
                        pdf.cell(box_w, 20, f"${total_value:,.0f}", border=1, align="C", fill=True)
                        pdf.ln(20)

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

                    # --- TOP CONTRIBUTORS & DETRACTORS ---
                    pdf.add_page(orientation='L')
                    pdf.set_font("Arial", "B", 26)
                    pdf.set_text_color(0, 0, 0)
                    pdf.cell(0, 15, "Top Contributors and Detractors", ln=True, align="L")
                    pdf.ln(5)

                    top_contribs_pdf = calc_df[calc_df['Excess Value'] > 0].nlargest(3, 'Excess Value')
                    top_detracts_pdf = calc_df[calc_df['Excess Value'] < 0].nsmallest(3, 'Excess Value')

                    box_w = 130
                    x_start_c = 15
                    x_start_d = 150

                    y_start = pdf.get_y()

                    # Contributors
                    pdf.set_xy(x_start_c, y_start)
                    pdf.set_font("Arial", "B", 16)
                    pdf.set_text_color(21, 87, 36)
                    pdf.cell(box_w, 10, "Top Contributors (Positive Excess Value)", ln=False, align="L")

                    y_curr_c = y_start + 15
                    if not top_contribs_pdf.empty:
                        for _, row in top_contribs_pdf.iterrows():
                            pdf.set_xy(x_start_c, y_curr_c)
                            pdf.set_fill_color(242, 248, 242)
                            pdf.rect(x_start_c, y_curr_c, box_w, 20, 'F')
                            pdf.set_fill_color(44, 160, 44)
                            pdf.rect(x_start_c, y_curr_c, 3, 20, 'F')

                            pdf.set_xy(x_start_c + 5, y_curr_c + 3)
                            pdf.set_font("Arial", "B", 12)
                            pdf.set_text_color(0, 0, 0)
                            sec_name = str(row.get('Security Name', ''))[:35]
                            pdf.cell(box_w - 5, 6, f"{row['Ticker']} - {sec_name}", ln=True)

                            pdf.set_xy(x_start_c + 5, y_curr_c + 10)
                            pdf.set_font("Arial", "", 11)
                            pdf.set_text_color(21, 87, 36)
                            pdf.cell(box_w - 5, 6, f"Excess Return: {row['Difference']:.2%}  |  Excess Value: +${row['Excess Value']:,.2f}", ln=True)
                            y_curr_c += 25
                    else:
                        pdf.set_xy(x_start_c, y_curr_c)
                        pdf.set_font("Arial", "I", 12)
                        pdf.set_text_color(100, 100, 100)
                        pdf.cell(box_w, 10, "No positive contributors found.", ln=True)

                    # Detractors
                    pdf.set_xy(x_start_d, y_start)
                    pdf.set_font("Arial", "B", 16)
                    pdf.set_text_color(114, 28, 36)
                    pdf.cell(box_w, 10, "Top Detractors (Negative Excess Value)", ln=False, align="L")

                    y_curr_d = y_start + 15
            
