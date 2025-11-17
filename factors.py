"""
Factor calculation module for computing momentum and value factor scores.
"""

import os
import pandas as pd
import numpy as np
from config import PRICES_DIR, FUNDAMENTALS_DIR, get_universe

# Global in-memory cache for price data
_PRICE_CACHE = {}
_FUNDAMENTALS_CACHE = None


def load_all_price_data(tickers, price_cache=None, max_missing_pct=0.1):
    """
    Load price data for all tickers from CSV files with caching.
    
    Args:
        tickers: List of ticker symbols
        price_cache: Optional dictionary to use as cache (if None, uses global cache)
        max_missing_pct: Maximum percentage of missing values allowed (default 0.1 = 10%)
        
    Returns:
        Dictionary mapping ticker to DataFrame with 'Adj Close' column and clean Date index
    """
    # Use provided cache or global cache
    cache = price_cache if price_cache is not None else _PRICE_CACHE
    
    price_data = {}
    
    for ticker in tickers:
        # Check cache first
        if ticker in cache:
            price_data[ticker] = cache[ticker]
            continue
        
        csv_file = os.path.join(PRICES_DIR, f"{ticker}.csv")
        
        if not os.path.exists(csv_file):
            continue
        
        try:
            # Read CSV - try with index_col=0 first (standard format)
            try:
                df = pd.read_csv(csv_file, index_col=0)
                # Date is the index, convert it using the specified method
                df.index = pd.to_datetime(df.index, utc=True, errors='coerce').dt.tz_convert(None)
            except:
                # Fallback: read without index_col and check for Date column
                df = pd.read_csv(csv_file)
                if 'Date' in df.columns:
                    # Date is a column, parse and use as index
                    df['Date'] = pd.to_datetime(df['Date'], utc=True, errors='coerce').dt.tz_convert(None)
                    df = df.set_index('Date')
                else:
                    # Try to parse first column as date
                    first_col = df.columns[0]
                    df[first_col] = pd.to_datetime(df[first_col], utc=True, errors='coerce').dt.tz_convert(None)
                    df = df.set_index(first_col)
            
            # Drop any rows with invalid dates (NaT in index)
            df = df[df.index.notna()]
            
            if df.empty:
                continue
            
            # Extract Adj Close column
            if 'Adj Close' in df.columns:
                price_series = df['Adj Close'].copy()
            elif len(df.columns) > 0:
                # Fallback: use first column
                price_series = df.iloc[:, 0].copy()
            else:
                continue
            
            # Remove missing values and check if too many are missing
            total_rows = len(price_series)
            missing_count = price_series.isna().sum()
            missing_pct = missing_count / total_rows if total_rows > 0 else 1.0
            
            if missing_pct > max_missing_pct:
                # Skip tickers with too many missing values
                continue
            
            # Forward fill then backward fill to handle gaps
            price_series = price_series.ffill().bfill()
            
            # Drop any remaining NaN values
            price_series = price_series.dropna()
            
            if price_series.empty:
                continue
            
            # Create clean DataFrame with Date index
            price_df = pd.DataFrame({'Adj Close': price_series})
            price_df.index.name = 'Date'
            
            # Sort by date
            price_df = price_df.sort_index()
            
            # Store in cache and return dict
            cache[ticker] = price_df
            price_data[ticker] = price_df
            
        except Exception as e:
            # Silently skip problematic files to avoid spam
            continue
    
    return price_data


def load_fundamentals(use_cache=True):
    """
    Load fundamental data from CSV file with caching.
    
    Args:
        use_cache: If True, use cached fundamentals if available
        
    Returns:
        DataFrame with columns: ticker, pe_ratio, pb_ratio, market_cap
    """
    global _FUNDAMENTALS_CACHE
    
    # Return cached data if available
    if use_cache and _FUNDAMENTALS_CACHE is not None:
        return _FUNDAMENTALS_CACHE
    
    fundamentals_file = os.path.join(FUNDAMENTALS_DIR, "fundamentals.csv")
    
    if not os.path.exists(fundamentals_file):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(fundamentals_file)
        # Cache the result
        if use_cache:
            _FUNDAMENTALS_CACHE = df
        return df
    except Exception as e:
        return pd.DataFrame()


def calculate_momentum_12m(price_series, as_of_date):
    """
    Calculate 12-month momentum: return over last 252 trading days excluding last 21 days.
    
    Args:
        price_series: Series with prices indexed by date
        as_of_date: Date string (YYYY-MM-DD) - only use data prior to this date
        
    Returns:
        Momentum value (float) or np.nan if insufficient data
    """
    # Convert as_of_date to Timestamp for comparison
    if isinstance(as_of_date, str):
        as_of_date = pd.to_datetime(as_of_date)
    elif not isinstance(as_of_date, pd.Timestamp):
        as_of_date = pd.to_datetime(as_of_date)
    
    # Filter data up to (but not including) as_of_date to avoid look-ahead bias
    price_data = price_series[price_series.index < as_of_date].copy()
    
    if len(price_data) < 252 + 21:  # Need at least 273 days
        return np.nan
    
    # Get prices: 21 days ago and 21+252 days ago
    price_21d_ago = price_data.iloc[-21]
    price_273d_ago = price_data.iloc[-(21 + 252)]
    
    # Calculate return
    momentum_12m = (price_21d_ago / price_273d_ago) - 1.0
    
    return momentum_12m


def calculate_momentum_6m(price_series, as_of_date):
    """
    Calculate 6-month momentum: return over last 126 trading days excluding last 21 days.
    
    Args:
        price_series: Series with prices indexed by date
        as_of_date: Date string (YYYY-MM-DD) - only use data prior to this date
        
    Returns:
        Momentum value (float) or np.nan if insufficient data
    """
    # Convert as_of_date to Timestamp for comparison
    if isinstance(as_of_date, str):
        as_of_date = pd.to_datetime(as_of_date)
    elif not isinstance(as_of_date, pd.Timestamp):
        as_of_date = pd.to_datetime(as_of_date)
    
    # Filter data up to (but not including) as_of_date to avoid look-ahead bias
    price_data = price_series[price_series.index < as_of_date].copy()
    
    if len(price_data) < 126 + 21:  # Need at least 147 days
        return np.nan
    
    # Get prices: 21 days ago and 21+126 days ago
    price_21d_ago = price_data.iloc[-21]
    price_147d_ago = price_data.iloc[-(21 + 126)]
    
    # Calculate return
    momentum_6m = (price_21d_ago / price_147d_ago) - 1.0
    
    return momentum_6m


def winsorize(series, lower=0.05, upper=0.95):
    """
    Winsorize a series at specified percentiles.
    
    Args:
        series: Series to winsorize
        lower: Lower percentile (default 0.05)
        upper: Upper percentile (default 0.95)
        
    Returns:
        Winsorized series
    """
    valid_series = series.dropna()
    if len(valid_series) == 0:
        return series
    
    lower_bound = valid_series.quantile(lower)
    upper_bound = valid_series.quantile(upper)
    
    winsorized = series.copy()
    winsorized[winsorized < lower_bound] = lower_bound
    winsorized[winsorized > upper_bound] = upper_bound
    
    return winsorized


def calculate_value_score(fundamentals_df, ticker):
    """
    Calculate value score using inverse of PE and PB ratios.
    
    Args:
        fundamentals_df: DataFrame with fundamental data
        ticker: Ticker symbol
        
    Returns:
        Dictionary with inv_pe and inv_pb, or None if data unavailable
    """
    ticker_data = fundamentals_df[fundamentals_df['ticker'] == ticker]
    
    if ticker_data.empty:
        return None
    
    row = ticker_data.iloc[0]
    pe_ratio = row.get('pe_ratio')
    pb_ratio = row.get('pb_ratio')
    
    # Calculate inverse ratios (higher inverse = better value)
    inv_pe = None
    inv_pb = None
    
    if pd.notna(pe_ratio) and pe_ratio > 0:
        inv_pe = 1.0 / pe_ratio
    
    if pd.notna(pb_ratio) and pb_ratio > 0:
        inv_pb = 1.0 / pb_ratio
    
    if inv_pe is None and inv_pb is None:
        return None
    
    return {'inv_pe': inv_pe, 'inv_pb': inv_pb}


def calculate_quality_score(fundamentals_df, ticker, price_series=None, as_of_date=None):
    """
    Calculate quality score using ROE, gross margin, debt-to-equity, and earnings stability.
    
    Args:
        fundamentals_df: DataFrame with fundamental data
        ticker: Ticker symbol
        price_series: Optional price series for earnings stability calculation
        as_of_date: Optional date for earnings stability calculation
        
    Returns:
        Dictionary with quality components, or None if data unavailable
    """
    ticker_data = fundamentals_df[fundamentals_df['ticker'] == ticker]
    
    if ticker_data.empty:
        return None
    
    row = ticker_data.iloc[0]
    
    # Extract quality metrics
    roe = row.get('roe')  # Return on equity (higher is better)
    gross_margin = row.get('gross_margin')  # Gross margin (higher is better)
    debt_to_equity = row.get('debt_to_equity')  # Debt-to-equity (lower is better, so we'll use inverse)
    earnings_growth = row.get('earnings_growth')  # Earnings growth (for stability proxy)
    
    # Calculate earnings stability from price volatility (proxy)
    # Lower volatility in returns = more stable earnings
    earnings_stability = None
    if price_series is not None and as_of_date is not None and len(price_series) > 252:
        # Calculate rolling 1-year return volatility (inverse = stability)
        try:
            # Get data up to as_of_date
            available_data = price_series[price_series.index <= as_of_date]
            if len(available_data) >= 252:
                # Calculate 1-year rolling volatility
                returns = available_data.pct_change().dropna()
                if len(returns) >= 252:
                    # Use last 252 days volatility
                    recent_returns = returns.tail(252)
                    volatility = recent_returns.std()
                    # Inverse volatility = stability (higher is better)
                    if volatility > 0:
                        earnings_stability = 1.0 / (1.0 + volatility)  # Normalize to 0-1 range
        except Exception:
            pass
    
    # If earnings_stability not calculated, use earnings_growth as proxy
    if earnings_stability is None and pd.notna(earnings_growth):
        # Use absolute earnings growth as stability proxy (more stable = less negative growth)
        earnings_stability = max(0, earnings_growth) if earnings_growth is not None else None
    
    return {
        'roe': roe,
        'gross_margin': gross_margin,
        'inv_debt_to_equity': 1.0 / (1.0 + debt_to_equity) if pd.notna(debt_to_equity) and debt_to_equity >= 0 else None,  # Inverse (lower debt = better)
        'earnings_stability': earnings_stability,
    }


def normalize_scores(scores, method='zscore'):
    """
    Normalize scores using z-score or rank.
    
    Args:
        scores: Series or array of scores
        method: 'zscore' or 'rank'
        
    Returns:
        Normalized scores
    """
    scores = pd.Series(scores)
    valid_scores = scores.dropna()
    
    if len(valid_scores) == 0:
        return pd.Series(np.nan, index=scores.index)
    
    if method == 'zscore':
        mean = valid_scores.mean()
        std = valid_scores.std()
        if std == 0:
            return pd.Series(0.0, index=scores.index)
        normalized = (scores - mean) / std
    elif method == 'rank':
        # Rank from 0 to 1 (percentile rank)
        ranks = valid_scores.rank(method='min', pct=True)
        normalized = pd.Series(np.nan, index=scores.index)
        normalized[valid_scores.index] = ranks
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


def compute_factor_scores(as_of_date, price_cache=None):
    """
    Compute factor scores for all tickers as of a given date.
    
    This function only uses data prior to as_of_date to avoid look-ahead bias.
    
    Args:
        as_of_date: Date string (YYYY-MM-DD) or datetime object
        price_cache: Optional dictionary of pre-loaded price data (if None, loads from files)
        
    Returns:
        DataFrame with columns: ["ticker", "date", "momentum_score", 
                                 "value_score", "quality_score", "combined_score"]
    """
    # Convert as_of_date to datetime
    if isinstance(as_of_date, str):
        as_of_date_dt = pd.to_datetime(as_of_date)
    else:
        as_of_date_dt = pd.to_datetime(as_of_date)
    
    # Get universe
    tickers = get_universe()
    
    # Load price data (use cache if provided, otherwise load from files)
    if price_cache is not None:
        # Use provided cache, only load missing tickers
        missing_tickers = [t for t in tickers if t not in price_cache]
        if missing_tickers:
            load_all_price_data(missing_tickers, price_cache=price_cache)
        price_data = {t: price_cache[t] for t in tickers if t in price_cache}
    else:
        # Load from files (will use global cache)
        price_data = load_all_price_data(tickers)
    
    if not price_data:
        return pd.DataFrame()
    
    # Load fundamentals (uses global cache)
    fundamentals_df = load_fundamentals(use_cache=True)
    
    # Filter out microcaps (market cap < $5 billion)
    MIN_MARKET_CAP = 5_000_000_000  # $5 billion
    if not fundamentals_df.empty and 'market_cap' in fundamentals_df.columns:
        valid_tickers = fundamentals_df[
            (fundamentals_df['market_cap'].notna()) & 
            (fundamentals_df['market_cap'] >= MIN_MARKET_CAP)
        ]['ticker'].tolist()
        # Only process tickers that pass market cap filter
        tickers = [t for t in tickers if t in valid_tickers]
    
    # Calculate raw factor scores for each ticker
    results = []
    
    for ticker in tickers:
        if ticker not in price_data:
            continue
        
        price_series = price_data[ticker]['Adj Close']
        
        # Calculate momentum factors (excluding last 21 days)
        momentum_12m = calculate_momentum_12m(price_series, as_of_date_dt)
        momentum_6m = calculate_momentum_6m(price_series, as_of_date_dt)
        
        # Calculate value components
        value_components = calculate_value_score(fundamentals_df, ticker)
        
        # Calculate quality components
        quality_components = calculate_quality_score(fundamentals_df, ticker, price_series, as_of_date_dt)
        
        results.append({
            'ticker': ticker,
            'momentum_12m': momentum_12m,
            'momentum_6m': momentum_6m,
            'inv_pe': value_components['inv_pe'] if value_components else None,
            'inv_pb': value_components['inv_pb'] if value_components else None,
            'roe': quality_components['roe'] if quality_components else None,
            'gross_margin': quality_components['gross_margin'] if quality_components else None,
            'inv_debt_to_equity': quality_components['inv_debt_to_equity'] if quality_components else None,
            'earnings_stability': quality_components['earnings_stability'] if quality_components else None,
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    if df.empty:
        return pd.DataFrame()
    
    # MOMENTUM: Convert to cross-sectional ranks (0-1 range)
    momentum_12m_rank = normalize_scores(df['momentum_12m'], method='rank')
    momentum_6m_rank = normalize_scores(df['momentum_6m'], method='rank')
    
    # Combine momentum: 0.7 * rank_12m + 0.3 * rank_6m
    momentum_score = 0.7 * momentum_12m_rank + 0.3 * momentum_6m_rank
    
    # VALUE: Process inverse PE and PB
    # Collect all value components for winsorization
    all_inv_pe = df['inv_pe'].dropna()
    all_inv_pb = df['inv_pb'].dropna()
    
    # Winsorize at 5th and 95th percentile
    if len(all_inv_pe) > 0:
        df['inv_pe_winsorized'] = winsorize(df['inv_pe'], lower=0.05, upper=0.95)
    else:
        df['inv_pe_winsorized'] = df['inv_pe']
    
    if len(all_inv_pb) > 0:
        df['inv_pb_winsorized'] = winsorize(df['inv_pb'], lower=0.05, upper=0.95)
    else:
        df['inv_pb_winsorized'] = df['inv_pb']
    
    # Average winsorized inverse ratios (use available components)
    value_raw = pd.Series(index=df.index, dtype=float)
    for idx in df.index:
        inv_pe = df.loc[idx, 'inv_pe_winsorized']
        inv_pb = df.loc[idx, 'inv_pb_winsorized']
        
        if pd.notna(inv_pe) and pd.notna(inv_pb):
            # Both available: average
            value_raw.loc[idx] = (inv_pe + inv_pb) / 2
        elif pd.notna(inv_pe):
            # Only PE available
            value_raw.loc[idx] = inv_pe
        elif pd.notna(inv_pb):
            # Only PB available
            value_raw.loc[idx] = inv_pb
        else:
            # Both missing
            value_raw.loc[idx] = np.nan
    
    # Convert to Z-scores, then to ranks (0-1)
    value_zscore = normalize_scores(value_raw, method='zscore')
    value_score = normalize_scores(value_zscore, method='rank')
    
    # QUALITY: Process quality components
    # Collect all quality components
    quality_components = ['roe', 'gross_margin', 'inv_debt_to_equity', 'earnings_stability']
    quality_scores_list = []
    
    for component in quality_components:
        if component in df.columns:
            # Normalize each component as z-score
            component_zscore = normalize_scores(df[component], method='zscore')
            quality_scores_list.append(component_zscore)
    
    # Average all available quality components (z-scores)
    if len(quality_scores_list) > 0:
        quality_df = pd.DataFrame(quality_scores_list).T
        # Average across columns (components), handling NaN values
        quality_raw = quality_df.mean(axis=1, skipna=True)
        # Convert final quality score to z-score, then rank
        quality_zscore = normalize_scores(quality_raw, method='zscore')
        quality_score = normalize_scores(quality_zscore, method='rank')
    else:
        # No quality data available
        quality_score = pd.Series(0.0, index=df.index)
    
    # COMBINED SCORE: 0.4 * momentum_score + 0.3 * value_score + 0.3 * quality_score
    combined_score = 0.4 * momentum_score + 0.3 * value_score + 0.3 * quality_score
    
    # Build final DataFrame with cleaner output
    result_df = pd.DataFrame({
        'ticker': df['ticker'],
        'date': as_of_date_dt,
        'momentum_score': momentum_score,
        'value_score': value_score,
        'quality_score': quality_score,
        'combined_score': combined_score,
    })
    
    # Sort by combined score (descending)
    result_df = result_df.sort_values('combined_score', ascending=False, na_position='last')
    
    return result_df


def compute_factor_scores_for_dates(dates, price_cache=None):
    """
    Compute factor scores for multiple dates.
    
    This is a convenience function for backtesting that computes factor scores
    for a list of dates and returns them in a format compatible with the old interface.
    
    Args:
        dates: List of date strings (YYYY-MM-DD) or datetime objects
        price_cache: Optional dictionary of pre-loaded price data
        
    Returns:
        DataFrame indexed by date with columns: ticker, momentum_score, 
        value_score, quality_score, combined_score
    """
    all_scores = []
    
    for date in dates:
        scores_df = compute_factor_scores(date, price_cache=price_cache)
        if not scores_df.empty:
            # Set date as index for compatibility with old format
            scores_df = scores_df.set_index('date')
            all_scores.append(scores_df)
    
    if not all_scores:
        return pd.DataFrame()
    
    # Combine all dates
    result = pd.concat(all_scores)
    
    return result


def get_top_stocks(factor_scores, date=None, top_n=20, factor='combined_score'):
    """
    Get top N stocks by factor score on a given date.
    
    This function works with the DataFrame returned by compute_factor_scores().
    For backward compatibility, it also handles the old format.
    
    Args:
        factor_scores: DataFrame from compute_factor_scores() or old format
        date: Date to filter (if None, uses most recent date in DataFrame)
        top_n: Number of top stocks to return
        factor: Factor to rank by (default 'combined_score', or 'combined' for old format)
        
    Returns:
        DataFrame with top N stocks and their scores
    """
    if factor_scores.empty:
        return pd.DataFrame()
    
    # Convert date to datetime if string
    if date is not None and isinstance(date, str):
        date = pd.to_datetime(date)
    
    # Handle format indexed by date (from compute_factor_scores_for_dates or old format)
    if isinstance(factor_scores.index, pd.DatetimeIndex) or 'date' not in factor_scores.columns:
        # Old format or date-indexed format
        if date is not None:
            factor_scores = factor_scores[factor_scores.index == date].copy()
        else:
            # Use most recent date
            date = factor_scores.index.max()
            factor_scores = factor_scores[factor_scores.index == date].copy()
        
        if factor_scores.empty:
            return pd.DataFrame()
        
        # Handle old 'combined' column name
        if factor == 'combined' and 'combined' in factor_scores.columns:
            factor = 'combined_score'
            if 'combined_score' not in factor_scores.columns:
                factor_scores = factor_scores.rename(columns={'combined': 'combined_score'})
    
    # Handle new format (with 'date' column)
    elif 'date' in factor_scores.columns:
        if date is not None:
            factor_scores = factor_scores[factor_scores['date'] == date].copy()
        # If no date specified, use the most recent date
        else:
            if not factor_scores.empty:
                date = factor_scores['date'].max()
                factor_scores = factor_scores[factor_scores['date'] == date].copy()
    
    # Also handle old 'as_of_date' column for backward compatibility
    elif 'as_of_date' in factor_scores.columns:
        if date is not None:
            factor_scores = factor_scores[factor_scores['as_of_date'] == date].copy()
        else:
            if not factor_scores.empty:
                date = factor_scores['as_of_date'].max()
                factor_scores = factor_scores[factor_scores['as_of_date'] == date].copy()
    
    if factor_scores.empty:
        return pd.DataFrame()
    
    # Sort by factor score
    if factor not in factor_scores.columns:
        print(f"Warning: Factor '{factor}' not found in DataFrame. Available columns: {factor_scores.columns.tolist()}")
        return pd.DataFrame()
    
    factor_scores = factor_scores.sort_values(by=factor, ascending=False, na_position='last')
    
    # Return top N
    return factor_scores.head(top_n)


if __name__ == "__main__":
    # Test the factor calculation
    from datetime import datetime
    
    print("Testing factor calculations...")
    
    # Use today's date
    test_date = datetime.now().strftime('%Y-%m-%d')
    print(f"Computing factor scores as of {test_date}...")
    
    factor_scores = compute_factor_scores(test_date)
    
    if not factor_scores.empty:
        print(f"\nFactor scores computed for {len(factor_scores)} tickers")
        print("\nTop 10 by combined score:")
        print(factor_scores.head(10)[['ticker', 'momentum_12m', 'momentum_6m', 'value_score', 'combined_score']])
    else:
        print("No factor scores computed. Make sure price data is downloaded first.")
