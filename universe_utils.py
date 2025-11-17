"""
Universe utility functions for filtering stocks by market cap and liquidity.
Separated from config.py to avoid circular imports.
"""

import pandas as pd
import yfinance as yf
from data_pipeline import load_price_data
from factors import load_fundamentals
from config import MIN_MARKET_CAP, MIN_AVG_VOLUME


# S&P 500 ticker list (comprehensive list of major S&P 500 constituents)
# Used as fallback if dynamic filtering fails
_SP500_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
    'V', 'JNJ', 'WMT', 'JPM', 'MA', 'PG', 'UNH', 'HD', 'DIS', 'BAC',
    'ADBE', 'NFLX', 'CRM', 'PYPL', 'INTC', 'CMCSA', 'PEP', 'TMO', 'COST',
    'AVGO', 'CSCO', 'ABT', 'NKE', 'MRK', 'TXN', 'QCOM', 'ACN', 'DHR',
    'VZ', 'LIN', 'NEE', 'PM', 'HON', 'UNP', 'RTX', 'UPS', 'BMY', 'LOW',
    'AMGN', 'SPGI', 'INTU', 'T', 'DE', 'BKNG', 'AXP', 'SBUX', 'ADP',
    'GILD', 'MDT', 'ZTS', 'C', 'TJX', 'ISRG', 'SYK', 'CL', 'GE', 'MMC',
    'CAT', 'GS', 'MS', 'BLK', 'SCHW', 'CVX', 'XOM', 'COP', 'SLB', 'EOG',
    'MO', 'SO', 'DUK', 'AEP', 'SRE', 'PCG', 'EXC', 'XEL', 'WEC', 'ES',
    'ETR', 'PEG', 'ED', 'FE', 'AEE', 'CNP', 'CMS', 'LNT', 'ATO', 'NI',
    'OKE', 'KMI', 'WMB', 'MPLX', 'EPD', 'ET', 'PAA', 'PSX', 'VLO', 'MPC',
    'HES', 'FANG', 'MRO', 'DVN', 'CTRA', 'OVV', 'MTCH', 'FTNT', 'ANET', 'CRWD',
    'ZS', 'NET', 'DDOG', 'MDB', 'SNOW', 'PLTR', 'RBLX', 'U', 'BILL', 'DOCN',
    'NOW', 'TEAM', 'ZM', 'DOCU', 'ESTC', 'SPLK', 'WDAY', 'VEEV', 'OKTA', 'QLYS',
    'PANW', 'CHKP', 'FFIV', 'AKAM', 'MCHP', 'SWKS', 'QRVO', 'MRVL', 'AMD', 'NXPI',
    'ON', 'MPWR', 'ALGM', 'DIOD', 'SLAB', 'AOSL', 'AMBA', 'OLED', 'POWI', 'SITM',
    'SILX', 'ALRM', 'AMAT', 'LRCX', 'KLAC', 'ASML', 'TER', 'ONTO', 'ACLS', 'UCTT',
    'FORM', 'PLAB', 'COHU', 'ICHR', 'AEIS', 'MU', 'WDC', 'STX', 'HPE', 'HPQ',
    'NTAP', 'DDD', 'SSYS', 'PRLB', 'XONE', 'DM', 'NNDM', 'IBM', 'JNPR', 'CIEN',
    'COMM', 'CALX', 'EXTR', 'INFN', 'NTGR', 'UI', 'UIS', 'VG', 'VSH', 'WSO',
    'WWD', 'ORCL', 'TENB', 'VRNS', 'RDWR', 'PFPT', 'TMUS', 'LUMN', 'ATUS', 'CNSL',
    'SHEN', 'USM', 'IDT', 'GOGO', 'WFC', 'TROW', 'BEN', 'IVZ', 'ETFC', 'AMTD',
    'HOOD', 'SOFI', 'LC', 'UPST', 'AFRM', 'NU', 'PAG', 'ABBV', 'REGN', 'VRTX',
    'BIIB', 'ALKS', 'ALNY', 'IONS', 'SGMO', 'CRISPR', 'EDIT', 'NTLA', 'BEAM', 'VERV',
    'PRME', 'RGNX', 'RARE', 'BLUE', 'FOLD', 'PFE', 'BDX', 'EW', 'BSX', 'ZBH',
    'BAX', 'HOLX', 'ALGN', 'XRAY', 'COO', 'NVST', 'OMCL', 'PODD', 'BA', 'LMT',
    'NOC', 'GD', 'HWM', 'TXT', 'EMR', 'ETN', 'IR', 'AOS', 'DOV', 'PH',
    'AME', 'GGG', 'RBC', 'ITT', 'AGCO', 'CNHI', 'TTC', 'MTZ', 'ASTE', 'AWI',
    'AZEK', 'BLD', 'NSC', 'KSU', 'JBHT', 'ODFL', 'XPO', 'KNX', 'WERN', 'ARCB',
    'CHRW', 'RXO', 'YELL', 'HTLD', 'MTRX', 'PTSI', 'USAK', 'WNC', 'FDX', 'GXO',
    'ZTO', 'JD', 'BABA', 'PDD', 'VIPS', 'WB', 'BIDU', 'TME', 'NTES', 'YY',
    'HUYA', 'DOYU', 'TAL', 'EDU', 'GSX', 'COE', 'LAIX', 'TGT', 'ROST', 'DG',
    'DLTR', 'FIVE', 'BBY', 'GME', 'AMC', 'BBBY', 'RH', 'WSM', 'W', 'ETSY',
    'SHOP', 'CMG', 'YUM', 'DPZ', 'WEN', 'JACK', 'CAKE', 'BLMN', 'DIN', 'TXRH',
    'CHUY', 'BJRI', 'RUTH', 'STKS', 'LOCO', 'TAST', 'FRGI', 'NDLS', 'BOJA', 'UAA',
    'UA', 'DKS', 'HIBB', 'ASO', 'BGS', 'FL', 'CAL', 'SHOO', 'VSTO', 'SWBI',
    'RGR', 'AOBC', 'OLN', 'CLW', 'AXTA', 'PPG', 'SHW', 'VAL', 'APD', 'ECL',
    'IFF', 'FMC', 'CF', 'MOS', 'NTR', 'CTVA', 'ADM', 'BG'
]
# Remove duplicates while preserving order
_SP500_TICKERS = list(dict.fromkeys(_SP500_TICKERS))


def get_filtered_universe():
    """
    Returns the top 50 most liquid S&P 500 stocks by market cap.
    
    Filters:
    - Excludes microcaps (market cap < $5 billion)
    - Excludes stocks with missing price data
    - Excludes stocks with average daily volume < 1 million
    
    Returns:
        List of ticker symbols (strings), sorted by market cap (descending)
    """
    # Load fundamentals to get market cap
    fundamentals_df = load_fundamentals(use_cache=True)
    
    if fundamentals_df.empty or 'market_cap' not in fundamentals_df.columns:
        # Fallback to hardcoded list if fundamentals not available
        print("Warning: Fundamentals not available, using fallback universe")
        return _SP500_TICKERS[:50]
    
    # Filter by market cap
    valid_tickers = fundamentals_df[
        (fundamentals_df['market_cap'].notna()) &
        (fundamentals_df['market_cap'] >= MIN_MARKET_CAP)
    ].copy()
    
    if valid_tickers.empty:
        print("Warning: No tickers meet market cap criteria, using fallback")
        return _SP500_TICKERS[:50]
    
    # Check price data availability and calculate average volume
    ticker_scores = []
    
    print("Filtering universe by liquidity and data quality...")
    
    for idx, row in valid_tickers.iterrows():
        ticker = row['ticker']
        market_cap = row['market_cap']
        
        # Load cached price data to check availability
        price_data = load_price_data(ticker)
        
        if price_data.empty:
            continue  # Skip if no price data
        
        # Check for missing data (more than 10% missing)
        if 'Adj Close' in price_data.columns:
            price_series = price_data['Adj Close']
            missing_pct = price_series.isna().sum() / len(price_series)
            if missing_pct > 0.1:
                continue  # Skip if too much missing data
            
            if len(price_series) < 100:
                continue  # Skip if insufficient data
        else:
            continue  # Skip if no usable data
        
        # Download volume data from yfinance (one-time check)
        try:
            stock = yf.Ticker(ticker)
            hist_data = stock.history(period="1y")  # Get last year of data
            
            if not hist_data.empty and 'Volume' in hist_data.columns:
                # Calculate average daily volume (last 252 trading days)
                recent_volume = hist_data['Volume'].tail(252)
                avg_volume = recent_volume.mean()
                
                if pd.isna(avg_volume) or avg_volume < MIN_AVG_VOLUME:
                    continue  # Skip if volume too low
            else:
                # If no volume data available, skip this ticker
                continue
        except Exception:
            # If we can't get volume data, skip this ticker
            continue
        
        ticker_scores.append({
            'ticker': ticker,
            'market_cap': market_cap
        })
    
    if not ticker_scores:
        print("Warning: No tickers meet all criteria, using fallback")
        return _SP500_TICKERS[:50]
    
    # Sort by market cap and return top 50
    ticker_df = pd.DataFrame(ticker_scores)
    ticker_df = ticker_df.sort_values('market_cap', ascending=False)
    top_50 = ticker_df.head(50)['ticker'].tolist()
    
    print(f"Universe: {len(top_50)} stocks (top 50 by market cap, filtered for liquidity)")
    
    return top_50

