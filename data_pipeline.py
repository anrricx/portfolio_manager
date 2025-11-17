"""
Data pipeline for downloading and caching historical price and fundamental data.
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from config import (
    PRICES_DIR, FUNDAMENTALS_DIR, START_DATE, 
    get_universe, get_benchmark_ticker
)


def ensure_directories():
    """Ensure data directories exist."""
    os.makedirs(PRICES_DIR, exist_ok=True)
    os.makedirs(FUNDAMENTALS_DIR, exist_ok=True)


def is_recent_data(cache_file, days_threshold=3):
    """
    Check if cached data is recent (within N trading days).
    
    Args:
        cache_file: Path to cached CSV file
        days_threshold: Number of trading days threshold (default 3)
        
    Returns:
        True if data is recent, False otherwise
    """
    if not os.path.exists(cache_file):
        return False
    
    try:
        cached_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        if cached_data.empty:
            return False
        
        # Get the last date in the cached data
        last_date = cached_data.index[-1]
        
        # Calculate today's date (market close time)
        today = datetime.now().date()
        
        # Check if last date is within threshold trading days
        # Approximate: 3 trading days ≈ 5 calendar days (accounting for weekends)
        days_diff = (today - last_date.date()).days
        
        # If within threshold calendar days, consider it recent
        # (More precise would be to count actual trading days, but this is simpler)
        return days_diff <= (days_threshold + 2)  # Add buffer for weekends
    except Exception as e:
        print(f"Error checking cache for {cache_file}: {e}")
        return False


def download_price_data(ticker, start_date, end_date=None):
    """
    Download daily adjusted close prices for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD), or None for today
        
    Returns:
        Series with adjusted close prices, indexed by date
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Download historical data
        if end_date:
            data = stock.history(start=start_date, end=end_date)
        else:
            # Use today as end date
            data = stock.history(start=start_date)
        
        if data.empty:
            print(f"Warning: No data found for {ticker}")
            return pd.Series(dtype=float)
        
        # Extract adjusted close prices
        # Use 'Adj Close' if available, otherwise use 'Close'
        if 'Adj Close' in data.columns:
            adj_close = data['Adj Close']
        else:
            adj_close = data['Close']
        
        return adj_close
    except Exception as e:
        print(f"Error downloading price data for {ticker}: {e}")
        return pd.Series(dtype=float)


def save_price_data(ticker, price_series):
    """
    Save price data to CSV file.
    
    Args:
        ticker: Stock ticker symbol
        price_series: Series with prices indexed by date
    """
    if price_series.empty:
        return
    
    cache_file = os.path.join(PRICES_DIR, f"{ticker}.csv")
    
    # Convert to DataFrame for easier CSV handling
    df = pd.DataFrame({'Adj Close': price_series})
    df.index.name = 'Date'
    
    try:
        df.to_csv(cache_file)
    except Exception as e:
        print(f"Error saving price data for {ticker}: {e}")


def download_fundamentals(ticker):
    """
    Download basic fundamental data for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with pe_ratio, pb_ratio, market_cap, or None if error
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract key fundamental metrics
        fundamentals = {
            'ticker': ticker,
            'pe_ratio': info.get('trailingPE', None),
            'pb_ratio': info.get('priceToBook', None),
            'market_cap': info.get('marketCap', None),
            # Quality metrics
            'roe': info.get('returnOnEquity', None),  # Return on equity
            'gross_margin': info.get('grossMargins', None),  # Gross margin
            'debt_to_equity': info.get('debtToEquity', None),  # Debt-to-equity ratio
            'earnings_growth': info.get('earningsGrowth', None),  # Earnings growth (for stability proxy)
        }
        
        # Check if we got any valid data
        if all(v is None for k, v in fundamentals.items() if k != 'ticker'):
            return None
        
        return fundamentals
    except Exception as e:
        print(f"Error downloading fundamentals for {ticker}: {e}")
        return None


def download_all_prices(tickers, start_date, end_date=None):
    """
    Download price data for all tickers, skipping recent cached data.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD), or None for today
        
    Returns:
        Dictionary mapping ticker to price Series
    """
    ensure_directories()
    
    price_data = {}
    skipped_count = 0
    downloaded_count = 0
    error_count = 0
    
    print(f"Downloading price data for {len(tickers)} tickers...")
    print(f"Date range: {start_date} to {end_date or 'today'}")
    print("-" * 60)
    
    for i, ticker in enumerate(tickers, 1):
        cache_file = os.path.join(PRICES_DIR, f"{ticker}.csv")
        
        # Check if we should skip this ticker
        if is_recent_data(cache_file, days_threshold=3):
            print(f"[{i}/{len(tickers)}] {ticker}: Using cached data (recent)")
            skipped_count += 1
            # Load cached data
            try:
                cached_df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                if 'Adj Close' in cached_df.columns:
                    price_data[ticker] = cached_df['Adj Close']
                else:
                    # Fallback if column name is different
                    price_data[ticker] = cached_df.iloc[:, 0]
            except Exception as e:
                print(f"  Warning: Error loading cached data for {ticker}: {e}")
                error_count += 1
            continue
        
        # Download new data
        print(f"[{i}/{len(tickers)}] {ticker}: Downloading...", end=" ")
        price_series = download_price_data(ticker, start_date, end_date)
        
        if price_series.empty:
            print("FAILED (no data)")
            error_count += 1
            continue
        
        # Save to cache
        save_price_data(ticker, price_series)
        price_data[ticker] = price_series
        downloaded_count += 1
        print(f"OK ({len(price_series)} days)")
    
    print("-" * 60)
    print(f"Summary: {downloaded_count} downloaded, {skipped_count} skipped (cached), {error_count} errors")
    
    return price_data


def download_all_fundamentals(tickers):
    """
    Download fundamental data for all tickers and save to a single CSV.
    
    Args:
        tickers: List of ticker symbols
        
    Returns:
        DataFrame with fundamentals
    """
    ensure_directories()
    
    fundamentals_list = []
    success_count = 0
    error_count = 0
    
    print(f"\nDownloading fundamental data for {len(tickers)} tickers...")
    print("-" * 60)
    
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}...", end=" ")
        fundamentals = download_fundamentals(ticker)
        
        if fundamentals is None:
            print("FAILED (no data)")
            error_count += 1
            continue
        
        fundamentals_list.append(fundamentals)
        success_count += 1
        print("OK")
    
    print("-" * 60)
    print(f"Summary: {success_count} successful, {error_count} errors")
    
    if not fundamentals_list:
        print("Warning: No fundamental data downloaded")
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(fundamentals_list)
    
    # Ensure columns are in the right order
    columns = ['ticker', 'pe_ratio', 'pb_ratio', 'market_cap']
    for col in columns:
        if col not in df.columns:
            df[col] = None
    df = df[columns]
    
    # Save to CSV
    fundamentals_file = os.path.join(FUNDAMENTALS_DIR, "fundamentals.csv")
    df.to_csv(fundamentals_file, index=False)
    print(f"\nFundamentals saved to {fundamentals_file}")
    
    return df


def load_price_data(ticker):
    """
    Load price data from cache file.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        DataFrame with price data (Date index, Adj Close column), or empty DataFrame if not found
    """
    cache_file = os.path.join(PRICES_DIR, f"{ticker}.csv")
    
    if not os.path.exists(cache_file):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df
    except Exception as e:
        print(f"Error loading price data for {ticker}: {e}")
        return pd.DataFrame()


def main():
    """
    Main entry point: Download all price and fundamental data.
    """
    print("=" * 60)
    print("STOCK INTELLIGENCE SYSTEM - DATA PIPELINE")
    print("=" * 60)
    
    # Get configuration
    universe = get_universe()
    benchmark = get_benchmark_ticker()
    start_date = START_DATE
    
    # Combine universe and benchmark
    all_tickers = list(set(universe + [benchmark]))
    
    print(f"\nUniverse: {len(universe)} tickers")
    print(f"Benchmark: {benchmark}")
    print(f"Total tickers to process: {len(all_tickers)}")
    
    # Download price data
    print("\n" + "=" * 60)
    print("STEP 1: Downloading Price Data")
    print("=" * 60)
    price_data = download_all_prices(all_tickers, start_date, end_date=None)
    
    # Download fundamental data
    print("\n" + "=" * 60)
    print("STEP 2: Downloading Fundamental Data")
    print("=" * 60)
    fundamentals_df = download_all_fundamentals(all_tickers)
    
    # Final summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Price data: {len(price_data)} tickers")
    print(f"Fundamental data: {len(fundamentals_df)} tickers")
    print(f"\nData saved to:")
    print(f"  Prices: {PRICES_DIR}/")
    print(f"  Fundamentals: {FUNDAMENTALS_DIR}/fundamentals.csv")


if __name__ == "__main__":
    main()
