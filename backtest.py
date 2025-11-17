"""
Backtesting module for testing factor strategy against benchmark.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from config import (
    INITIAL_CAPITAL, BENCHMARK, TRANSACTION_COST_BPS, PLOTS_DIR, TOP_N,
    START_DATE, get_universe, get_benchmark_ticker, REBALANCE_FREQUENCY
)
from data_pipeline import load_price_data
from factors import compute_factor_scores, load_all_price_data
import os


def load_price_matrix(tickers, price_cache=None):
    """
    Load price data for all tickers and create a price matrix.
    Uses optimized caching to avoid re-reading files.
    
    Args:
        tickers: List of ticker symbols
        price_cache: Optional dictionary to store/retrieve price data
        
    Returns:
        Tuple of (price_matrix DataFrame, price_cache dictionary)
    """
    # Load all price data at once using optimized loader
    if price_cache is None:
        price_cache = {}
    
    # Load all tickers at once (will use cache)
    price_data_dict = load_all_price_data(tickers, price_cache=price_cache)
    
    # Build price matrix from cached data
    price_matrix = pd.DataFrame()
    
    for ticker in tickers:
        if ticker in price_cache:
            price_df = price_cache[ticker]
            if 'Adj Close' in price_df.columns:
                price_series = price_df['Adj Close'].copy()
                # Ensure timezone-naive index
                if price_series.index.tz is not None:
                    price_series.index = price_series.index.tz_localize(None)
                price_matrix[ticker] = price_series
    
    # Sort by date and ensure timezone-naive
    if not price_matrix.empty:
        price_matrix = price_matrix.sort_index()
        if price_matrix.index.tz is not None:
            price_matrix.index = price_matrix.index.tz_localize(None)
    
    return price_matrix, price_cache


def get_monthly_rebalance_dates(price_matrix):
    """
    Get monthly rebalance dates (last trading day of each month).
    
    Args:
        price_matrix: DataFrame with dates as index
        
    Returns:
        List of rebalance dates (last trading day of each month)
    """
    # Group by year-month and get last date in each group
    rebalance_dates = []
    for year_month, group in price_matrix.groupby([price_matrix.index.year, price_matrix.index.month]):
        last_date = group.index.max()
        rebalance_dates.append(last_date)
    
    return sorted(rebalance_dates)


def calculate_cagr(equity_curve):
    """
    Calculate Compound Annual Growth Rate (CAGR).
    
    Args:
        equity_curve: Series of portfolio values
        
    Returns:
        CAGR as decimal (e.g., 0.10 for 10%)
    """
    if len(equity_curve) < 2:
        return 0.0
    
    start_value = equity_curve.iloc[0]
    end_value = equity_curve.iloc[-1]
    
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    years = days / 365.25
    
    if years <= 0 or start_value <= 0:
        return 0.0
    
    cagr = (end_value / start_value) ** (1 / years) - 1
    
    return cagr


def calculate_annualized_volatility(equity_curve):
    """
    Calculate annualized volatility.
    
    Args:
        equity_curve: Series of portfolio values
        
    Returns:
        Annualized volatility as decimal
    """
    returns = equity_curve.pct_change().dropna()
    
    if len(returns) == 0:
        return 0.0
    
    # Annualize daily volatility
    volatility = returns.std() * np.sqrt(252)
    
    return volatility


def calculate_sharpe_ratio(equity_curve, risk_free_rate=0.0):
    """
    Calculate Sharpe ratio.
    
    Args:
        equity_curve: Series of portfolio values
        risk_free_rate: Risk-free rate (default 0.0)
        
    Returns:
        Sharpe ratio
    """
    cagr = calculate_cagr(equity_curve)
    volatility = calculate_annualized_volatility(equity_curve)
    
    if volatility == 0:
        return 0.0
    
    sharpe = (cagr - risk_free_rate) / volatility
    
    return sharpe


def calculate_max_drawdown(equity_curve):
    """
    Calculate maximum drawdown.
    
    Args:
        equity_curve: Series of portfolio values
        
    Returns:
        Maximum drawdown as decimal (negative value, e.g., -0.20 for 20% drawdown)
    """
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    
    max_drawdown = drawdown.min()
    
    return max_drawdown


def run_backtest():
    """
    Run a long-only, monthly-rebalanced backtest using factor scores.
    
    Returns:
        Dictionary with:
        - 'equity_curve': DataFrame with strategy and benchmark equity curves
        - 'metrics': Dictionary with performance metrics
        - 'cagr': CAGR of strategy
        - 'benchmark_cagr': CAGR of benchmark
        - 'volatility': Annualized volatility
        - 'benchmark_volatility': Benchmark annualized volatility
        - 'sharpe_ratio': Sharpe ratio
        - 'benchmark_sharpe_ratio': Benchmark Sharpe ratio
        - 'max_drawdown': Maximum drawdown
        - 'benchmark_max_drawdown': Benchmark maximum drawdown
    """
    print("=" * 60)
    print("RUNNING BACKTEST")
    print("=" * 60)
    
    # Get configuration
    universe = get_universe()
    benchmark_ticker = get_benchmark_ticker()
    initial_capital = INITIAL_CAPITAL
    top_n = TOP_N
    transaction_cost_bps = TRANSACTION_COST_BPS
    transaction_cost = transaction_cost_bps / 10000  # Convert to decimal
    
    print(f"Universe: {len(universe)} tickers")
    print(f"Top N: {top_n}")
    print(f"Initial Capital: ${initial_capital:,.0f}")
    print(f"Transaction Cost: {transaction_cost_bps} bps ({transaction_cost*100:.4f}%)")
    print(f"Rebalance Frequency: Monthly")
    
    # Load ALL price data ONCE at the start (optimized with caching)
    print("\nLoading all price data (one-time load)...")
    all_tickers = list(set(universe + [benchmark_ticker]))
    
    # Create price cache dictionary
    price_cache = {}
    
    # Load all price data into cache
    price_matrix, price_cache = load_price_matrix(all_tickers, price_cache=price_cache)
    
    if price_matrix.empty:
        print("Error: No price data loaded")
        return {}
    
    # Filter to dates with data
    price_matrix = price_matrix.dropna(how='all')
    
    if price_matrix.empty:
        print("Error: No valid price data")
        return {}
    
    start_date = price_matrix.index[0]
    end_date = price_matrix.index[-1]
    
    print(f"Price data range: {start_date.date()} to {end_date.date()}")
    print(f"Trading days: {len(price_matrix)}")
    print(f"Loaded {len(price_cache)} tickers into cache")
    
    # Get monthly rebalance dates (last trading day of each month)
    rebalance_dates = get_monthly_rebalance_dates(price_matrix)
    
    if not rebalance_dates:
        # Add first date if no monthly dates found
        rebalance_dates = [start_date]
    
    print(f"Rebalance dates: {len(rebalance_dates)} (last trading day of each month)")
    
    # Load SPY data for 200-day SMA check
    print("Loading SPY data for market regime filter...")
    spy_data = load_price_data(benchmark_ticker)
    spy_prices = None
    spy_sma_200 = None
    
    if not spy_data.empty and 'Adj Close' in spy_data.columns:
        spy_prices = spy_data['Adj Close'].copy()
        # Safe timezone conversion: check if timezone-aware first
        try:
            # Check if index has timezone info
            if hasattr(spy_prices.index, 'tz') and spy_prices.index.tz is not None:
                # Timezone-aware: convert to UTC then remove timezone
                spy_prices.index = pd.to_datetime(spy_prices.index, utc=True, errors='coerce')
                spy_prices.index = spy_prices.index.tz_convert(None)
            else:
                # Already timezone-naive, just ensure it's datetime
                spy_prices.index = pd.to_datetime(spy_prices.index, errors='coerce')
        except (AttributeError, TypeError):
            # Fallback: try to convert, then check and remove timezone if needed
            spy_prices.index = pd.to_datetime(spy_prices.index, errors='coerce')
            if hasattr(spy_prices.index, 'tz') and spy_prices.index.tz is not None:
                spy_prices.index = spy_prices.index.tz_convert(None)
        
        # Drop any NaT values from failed conversions
        spy_prices = spy_prices.dropna()
        spy_prices = spy_prices.sort_index()
        
        # Calculate 200-day SMA
        spy_sma_200 = spy_prices.rolling(window=200, min_periods=1).mean()
        print(f"  SPY data loaded: {len(spy_prices)} dates from {spy_prices.index[0].date()} to {spy_prices.index[-1].date()}")
        if len(spy_prices) >= 5:
            print(f"  SPY last 5 dates: {[d.date() for d in spy_prices.tail(5).index]}")
            print(f"  SPY last 5 prices: {spy_prices.tail(5).values}")
            if len(spy_prices) >= 200:
                print(f"  SPY SMA200 (last): ${spy_sma_200.iloc[-1]:.2f}")
    else:
        print(f"  WARNING: Could not load SPY data")
    
    # Initialize portfolio tracking
    # Ensure index is unique (drop duplicates if any)
    portfolio_index = price_matrix.index
    if portfolio_index.duplicated().any():
        print(f"Warning: Found {portfolio_index.duplicated().sum()} duplicate dates in price matrix, removing duplicates")
        portfolio_index = portfolio_index.drop_duplicates()
        price_matrix = price_matrix.loc[portfolio_index]
    
    portfolio_value = pd.Series(index=portfolio_index, dtype=float)
    portfolio_value.iloc[0] = initial_capital
    
    # Track holdings (dictionary: ticker -> number of shares)
    current_holdings = {}
    cash = initial_capital  # Track cash separately
    MIN_WEIGHT_CHANGE = 0.01  # 1% minimum weight change threshold
    
    # Backtest loop
    print("\nRunning backtest...")
    rebalance_count = 0
    debug_count = 0  # Track first 5 rebalances for debugging
    
    for i, date in enumerate(price_matrix.index):
        # Calculate current portfolio value (stocks + cash)
        stock_value = sum(
            shares * price_matrix.loc[date, ticker]
            for ticker, shares in current_holdings.items()
            if ticker in price_matrix.columns and pd.notna(price_matrix.loc[date, ticker])
        )
        current_portfolio_value = stock_value + cash
        
        # Check if we should rebalance (only on last trading day of month)
        should_rebalance = (date in rebalance_dates or i == 0)
        
        if should_rebalance:
            date_str = date.strftime('%Y-%m-%d')
            
            # Find the last available trading day BEFORE this date for factor calculation
            # This ensures we use data that was actually available at that time
            available_dates = price_matrix.index[price_matrix.index <= date]
            if len(available_dates) == 0:
                continue
            
            # Use the previous trading day (or current if it's the first day)
            factor_calc_date = available_dates[-2] if len(available_dates) > 1 else available_dates[-1]
            factor_calc_date_str = factor_calc_date.strftime('%Y-%m-%d')
            
            # Debug: Print first 5 rebalance attempts
            if debug_count < 5:
                print(f"\n--- DEBUG Rebalance #{debug_count + 1} ---")
                print(f"  Rebalance date: {date_str}")
                print(f"  Factor calc date: {factor_calc_date_str}")
            
            try:
                # Compute factor scores as of this date (using data prior to date)
                factor_scores_df = compute_factor_scores(factor_calc_date_str, price_cache=price_cache)
                
                # Debug: Print factor scores info
                if debug_count < 5:
                    print(f"  Factor scores shape: {factor_scores_df.shape}")
                    print(f"  Factor scores empty: {factor_scores_df.empty}")
                    if not factor_scores_df.empty:
                        print(f"  Top 5 rows:")
                        print(factor_scores_df.head(5).to_string())
                
                if factor_scores_df.empty:
                    if debug_count < 5:
                        print(f"  WARNING: Factor scores empty, skipping rebalance")
                    debug_count += 1
                    continue
                else:
                    # Check market regime: SPY vs 200-day SMA
                    market_exposure = 1.0  # Full exposure by default
                    if spy_prices is not None and spy_sma_200 is not None:
                        # Ensure date is timezone-naive for comparison
                        date_naive = pd.to_datetime(date)
                        if hasattr(date_naive, 'tz') and date_naive.tz is not None:
                            date_naive = date_naive.tz_localize(None)
                        elif isinstance(date_naive, pd.Timestamp) and date_naive.tz is not None:
                            date_naive = date_naive.tz_localize(None)
                        
                        # Ensure spy_prices index is timezone-naive
                        spy_index_naive = spy_prices.index
                        if hasattr(spy_index_naive, 'tz') and spy_index_naive.tz is not None:
                            spy_index_naive = pd.to_datetime(spy_prices.index, utc=True, errors='coerce').tz_convert(None)
                        else:
                            spy_index_naive = pd.to_datetime(spy_prices.index, errors='coerce')
                        
                        # Find closest available SPY date (on or before current date)
                        spy_available_dates = spy_index_naive[spy_index_naive <= date_naive]
                        if len(spy_available_dates) > 0:
                            spy_date_naive = spy_available_dates[-1]
                            # Find matching date in original spy_prices index
                            # Convert spy_prices index to naive for comparison
                            spy_orig_index_naive = pd.to_datetime(spy_prices.index, errors='coerce')
                            if hasattr(spy_prices.index, 'tz') and spy_prices.index.tz is not None:
                                spy_orig_index_naive = pd.to_datetime(spy_prices.index, utc=True, errors='coerce').tz_convert(None)
                            
                            # Find matching index position
                            matching_idx = spy_orig_index_naive == spy_date_naive
                            if matching_idx.any():
                                spy_date_orig = spy_prices.index[matching_idx][0]
                                spy_price_val = spy_prices.loc[spy_date_orig]
                                spy_sma_val = spy_sma_200.loc[spy_date_orig]
                            else:
                                spy_price_val = None
                                spy_sma_val = None
                            
                            if spy_price_val is not None and spy_sma_val is not None:
                                
                                if pd.notna(spy_price_val) and pd.notna(spy_sma_val) and spy_price_val < spy_sma_val:
                                    # Bear market: reduce exposure to 50%
                                    market_exposure = 0.5
                                
                                # Debug: Print SPY info for first 5 rebalances
                                if debug_count < 5:
                                    print(f"  SPY price: ${spy_price_val:.2f}, SMA200: ${spy_sma_val:.2f}")
                                    print(f"  Market exposure: {market_exposure*100:.0f}%")
                                    if len(spy_prices) >= 5:
                                        spy_last_5 = spy_prices.tail(5).index
                                        # Convert to timezone-naive for printing
                                        if hasattr(spy_last_5, 'tz') and spy_last_5.tz is not None:
                                            spy_last_5 = pd.to_datetime(spy_last_5, utc=True, errors='coerce').tz_convert(None)
                                        print(f"  SPY last 5 dates: {spy_last_5.tolist()}")
                    
                    # Rank by combined_score and select top N
                    factor_scores_df = factor_scores_df.sort_values('combined_score', ascending=False, na_position='last')
                    top_tickers = factor_scores_df.head(top_n)['ticker'].tolist()
                    
                    # Debug: Print top tickers
                    if debug_count < 5:
                        print(f"  Top {min(5, len(top_tickers))} tickers: {top_tickers[:5]}")
                    
                    # Filter to tickers that have price data on this date
                    available_tickers = [t for t in top_tickers if t in price_matrix.columns and pd.notna(price_matrix.loc[date, t])]
                    
                    if debug_count < 5:
                        print(f"  Available tickers (with price data): {len(available_tickers)}")
                    
                    if not available_tickers:
                        if debug_count < 5:
                            print(f"  WARNING: No available tickers, skipping rebalance")
                        debug_count += 1
                        continue
                    
                    # Calculate current position weights
                    current_weights = {}
                    for ticker, shares in current_holdings.items():
                        if ticker in price_matrix.columns:
                            price = price_matrix.loc[date, ticker]
                            if pd.notna(price) and price > 0:
                                value = shares * price
                                current_weights[ticker] = value / current_portfolio_value if current_portfolio_value > 0 else 0
                    
                    # Calculate target weights (equal weight, adjusted for market exposure)
                    target_weight_per_stock = market_exposure / len(available_tickers)
                    target_weights = {ticker: target_weight_per_stock for ticker in available_tickers}
                    
                    # Debug: Print target weights for first rebalance
                    if debug_count < 5:
                        print(f"  Target weights (first 5): {dict(list(target_weights.items())[:5])}")
                        print(f"  Total target stock weight: {sum(target_weights.values()):.4f}")
                    
                    # Check if rebalancing is needed (weight change > 1%)
                    needs_rebalance = False
                    if i == 0:
                        needs_rebalance = True
                        if debug_count < 5:
                            print(f"  First rebalance - forcing rebalance")
                    else:
                        # Check if any position needs significant rebalancing
                        all_tickers = set(list(current_weights.keys()) + list(target_weights.keys()))
                        for ticker in all_tickers:
                            current_w = current_weights.get(ticker, 0)
                            target_w = target_weights.get(ticker, 0)
                            weight_diff = abs(target_w - current_w)
                            if weight_diff >= MIN_WEIGHT_CHANGE:
                                needs_rebalance = True
                                break
                        
                        # Also check if market exposure changed
                        current_stock_weight = sum(current_weights.values())
                        target_stock_weight = market_exposure
                        if abs(target_stock_weight - current_stock_weight) >= MIN_WEIGHT_CHANGE:
                            needs_rebalance = True
                    
                    if needs_rebalance:
                        # Calculate target dollar amounts
                        target_values = {}
                        for ticker in available_tickers:
                            price = price_matrix.loc[date, ticker]
                            if pd.notna(price) and price > 0:
                                target_values[ticker] = current_portfolio_value * target_weights[ticker]
                        
                        # Calculate trades needed
                        trades = {}  # ticker -> net trade value (positive = buy, negative = sell)
                        all_tickers_involved = set(list(current_weights.keys()) + list(target_weights.keys()))
                        
                        for ticker in all_tickers_involved:
                            current_val = current_weights.get(ticker, 0) * current_portfolio_value
                            target_val = target_values.get(ticker, 0)
                            trades[ticker] = target_val - current_val
                        
                        # Calculate transaction costs (0.05% on buys and sells)
                        total_traded_value = 0
                        for ticker, trade_value in trades.items():
                            if abs(trade_value) > 0:
                                total_traded_value += abs(trade_value)
                        
                        transaction_cost_amount = total_traded_value * transaction_cost
                        
                        # Adjust portfolio for transaction costs
                        portfolio_after_costs = current_portfolio_value - transaction_cost_amount
                        
                        # Recalculate target values with adjusted portfolio
                        target_values = {}
                        for ticker in available_tickers:
                            price = price_matrix.loc[date, ticker]
                            if pd.notna(price) and price > 0:
                                target_values[ticker] = portfolio_after_costs * target_weights[ticker]
                        
                        # Update holdings
                        new_holdings = {}
                        for ticker in available_tickers:
                            price = price_matrix.loc[date, ticker]
                            if pd.notna(price) and price > 0:
                                target_value = target_values.get(ticker, 0)
                                shares = target_value / price
                                new_holdings[ticker] = shares
                        
                        # Calculate new stock value and cash
                        new_stock_value = sum(
                            shares * price_matrix.loc[date, ticker]
                            for ticker, shares in new_holdings.items()
                            if ticker in price_matrix.columns and pd.notna(price_matrix.loc[date, ticker])
                        )
                        cash = portfolio_after_costs - new_stock_value
                        current_holdings = new_holdings
                        
                        rebalance_count += 1
                        if date in rebalance_dates or i == 0:
                            exposure_str = f"{market_exposure*100:.0f}% exposure" if market_exposure < 1.0 else "full exposure"
                            print(f"  {date_str}: Rebalanced to {len(available_tickers)} positions ({exposure_str})")
                    
                    debug_count += 1
                
            except Exception as e:
                # Keep current holdings on error
                if debug_count < 5:
                    print(f"  ERROR during rebalance: {e}")
                    import traceback
                    traceback.print_exc()
                debug_count += 1
        
        # Calculate portfolio value for this date (stocks + cash)
        stock_value = sum(
            shares * price_matrix.loc[date, ticker]
            for ticker, shares in current_holdings.items()
            if ticker in price_matrix.columns and pd.notna(price_matrix.loc[date, ticker])
        )
        portfolio_value.loc[date] = stock_value + cash
    
    # Load benchmark data (reuse SPY data if available, otherwise load fresh)
    print("\nLoading benchmark data...")
    if spy_prices is not None:
        benchmark_prices = spy_prices.copy()
        # Ensure timezone-naive (should already be, but double-check for safety)
        try:
            if hasattr(benchmark_prices.index, 'tz') and benchmark_prices.index.tz is not None:
                # Timezone-aware: convert to UTC then remove timezone
                benchmark_prices.index = pd.to_datetime(benchmark_prices.index, utc=True, errors='coerce')
                benchmark_prices.index = benchmark_prices.index.tz_convert(None)
            else:
                # Already timezone-naive, just ensure it's datetime
                benchmark_prices.index = pd.to_datetime(benchmark_prices.index, errors='coerce')
        except (AttributeError, TypeError):
            # Fallback: try to convert, then check and remove timezone if needed
            benchmark_prices.index = pd.to_datetime(benchmark_prices.index, errors='coerce')
            if hasattr(benchmark_prices.index, 'tz') and benchmark_prices.index.tz is not None:
                benchmark_prices.index = benchmark_prices.index.tz_convert(None)
        benchmark_prices = benchmark_prices.dropna()
        benchmark_prices = benchmark_prices.sort_index()
    else:
        benchmark_data = load_price_data(benchmark_ticker)
        if benchmark_data.empty or 'Adj Close' not in benchmark_data.columns:
            print(f"Warning: Could not load benchmark {benchmark_ticker}")
            benchmark_equity = pd.Series(dtype=float)
            benchmark_prices = None
        else:
            benchmark_prices = benchmark_data['Adj Close'].copy()
            # Safe timezone conversion
            try:
                if hasattr(benchmark_prices.index, 'tz') and benchmark_prices.index.tz is not None:
                    # Timezone-aware: convert to UTC then remove timezone
                    benchmark_prices.index = pd.to_datetime(benchmark_prices.index, utc=True, errors='coerce')
                    benchmark_prices.index = benchmark_prices.index.tz_convert(None)
                else:
                    # Already timezone-naive, just ensure it's datetime
                    benchmark_prices.index = pd.to_datetime(benchmark_prices.index, errors='coerce')
            except (AttributeError, TypeError):
                # Fallback: try to convert, then check and remove timezone if needed
                benchmark_prices.index = pd.to_datetime(benchmark_prices.index, errors='coerce')
                if hasattr(benchmark_prices.index, 'tz') and benchmark_prices.index.tz is not None:
                    benchmark_prices.index = benchmark_prices.index.tz_convert(None)
            benchmark_prices = benchmark_prices.dropna()
            benchmark_prices = benchmark_prices.sort_index()
    
    # Ensure portfolio_value index is unique (drop duplicates if any) before benchmark alignment
    portfolio_index_final = portfolio_value.index
    if portfolio_index_final.duplicated().any():
        print(f"Warning: Found {portfolio_index_final.duplicated().sum()} duplicate dates in portfolio_value, removing duplicates")
        # Keep first occurrence of each duplicate
        portfolio_index_final = portfolio_index_final.drop_duplicates(keep='first')
        portfolio_value = portfolio_value.loc[portfolio_index_final]
    
    if benchmark_prices is not None:
        # Ensure benchmark_prices index is also unique
        if benchmark_prices.index.duplicated().any():
            print(f"Warning: Found {benchmark_prices.index.duplicated().sum()} duplicate dates in benchmark_prices, removing duplicates")
            benchmark_prices = benchmark_prices[~benchmark_prices.index.duplicated(keep='first')]
        
        # Double-check portfolio_index_final is unique
        if portfolio_index_final.duplicated().any():
            print(f"Warning: portfolio_index_final still has duplicates, fixing...")
            portfolio_index_final = portfolio_index_final.drop_duplicates(keep='first')
            portfolio_value = portfolio_value.loc[portfolio_index_final]
        
        # Align benchmark dates with portfolio dates
        # Use reindex with forward fill for missing dates
        try:
            benchmark_aligned = benchmark_prices.reindex(portfolio_index_final, method='ffill')
        except (TypeError, ValueError) as e:
            # For newer pandas, use ffill
            try:
                benchmark_aligned = benchmark_prices.reindex(portfolio_index_final).ffill()
            except ValueError as e2:
                # Manual fallback alignment if reindex still fails
                print(f"Warning: Standard reindex failed, using manual alignment: {e2}")
                benchmark_aligned = pd.Series(index=portfolio_index_final, dtype=float)
                for date in portfolio_index_final:
                    # Find closest benchmark date <= current date
                    available_dates = benchmark_prices.index[benchmark_prices.index <= date]
                    if len(available_dates) > 0:
                        benchmark_aligned.loc[date] = benchmark_prices.loc[available_dates[-1]]
                    else:
                        benchmark_aligned.loc[date] = np.nan
                benchmark_aligned = benchmark_aligned.ffill()
        
        if benchmark_aligned.isna().all():
            print("Warning: No overlapping dates with benchmark")
            benchmark_equity = pd.Series(dtype=float)
        else:
            # Calculate benchmark equity curve
            benchmark_returns = benchmark_aligned.pct_change().fillna(0)
            benchmark_equity = (1 + benchmark_returns).cumprod() * initial_capital
            benchmark_equity.index = portfolio_index_final
    else:
        benchmark_equity = pd.Series(dtype=float)
    
    # Align equity curves
    if not benchmark_equity.empty:
        common_dates = portfolio_index_final.intersection(benchmark_equity.index)
        portfolio_value_aligned = portfolio_value.loc[common_dates]
        benchmark_equity_aligned = benchmark_equity.loc[common_dates]
    else:
        common_dates = portfolio_index_final
        portfolio_value_aligned = portfolio_value
        benchmark_equity_aligned = pd.Series(dtype=float)
    
    # Create equity curve DataFrame
    equity_curve_df = pd.DataFrame({
        'strategy': portfolio_value_aligned,
        'benchmark': benchmark_equity_aligned
    }, index=common_dates)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    
    strategy_cagr = calculate_cagr(portfolio_value_aligned)
    strategy_volatility = calculate_annualized_volatility(portfolio_value_aligned)
    strategy_sharpe = calculate_sharpe_ratio(portfolio_value_aligned, risk_free_rate=0.0)
    strategy_max_dd = calculate_max_drawdown(portfolio_value_aligned)
    
    if not benchmark_equity_aligned.empty:
        benchmark_cagr = calculate_cagr(benchmark_equity_aligned)
        benchmark_volatility = calculate_annualized_volatility(benchmark_equity_aligned)
        benchmark_sharpe = calculate_sharpe_ratio(benchmark_equity_aligned, risk_free_rate=0.0)
        benchmark_max_dd = calculate_max_drawdown(benchmark_equity_aligned)
    else:
        benchmark_cagr = 0.0
        benchmark_volatility = 0.0
        benchmark_sharpe = 0.0
        benchmark_max_dd = 0.0
    
    metrics = {
        'cagr': strategy_cagr,
        'benchmark_cagr': benchmark_cagr,
        'volatility': strategy_volatility,
        'benchmark_volatility': benchmark_volatility,
        'sharpe_ratio': strategy_sharpe,
        'benchmark_sharpe_ratio': benchmark_sharpe,
        'max_drawdown': strategy_max_dd,
        'benchmark_max_drawdown': benchmark_max_dd,
        'initial_capital': initial_capital,
        'final_value': portfolio_value_aligned.iloc[-1],
        'benchmark_final_value': benchmark_equity_aligned.iloc[-1] if not benchmark_equity_aligned.empty else 0,
        'rebalance_count': rebalance_count,
    }
    
    # Generate plot
    print("\nGenerating plot...")
    plot_path = os.path.join(PLOTS_DIR, 'equity_curve.png')
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    plt.figure(figsize=(14, 8))
    plt.plot(equity_curve_df.index, equity_curve_df['strategy'], 
             label='Strategy', linewidth=2, color='blue')
    
    if not benchmark_equity_aligned.empty:
        plt.plot(equity_curve_df.index, equity_curve_df['benchmark'], 
                 label=f'Benchmark ({benchmark_ticker})', linewidth=2, 
                 linestyle='--', color='orange')
    
    plt.title('Equity Curve - Factor Strategy vs Benchmark', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.close()
    
    print("\n" + "=" * 60)
    print("BACKTEST COMPLETE")
    print("=" * 60)
    
    return {
        'equity_curve': equity_curve_df,
        'metrics': metrics,
        'cagr': strategy_cagr,
        'benchmark_cagr': benchmark_cagr,
        'volatility': strategy_volatility,
        'benchmark_volatility': benchmark_volatility,
        'sharpe_ratio': strategy_sharpe,
        'benchmark_sharpe_ratio': benchmark_sharpe,
        'max_drawdown': strategy_max_dd,
        'benchmark_max_drawdown': benchmark_max_dd,
    }


if __name__ == "__main__":
    results = run_backtest()
    
    if results:
        metrics = results['metrics']
        
        print("\n" + "=" * 70)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("=" * 70)
        
        print(f"\n{'Metric':<25} {'Strategy':>15} {'Benchmark':>15} {'Difference':>15}")
        print("-" * 70)
        
        print(f"{'CAGR':<25} {metrics['cagr']*100:>14.2f}% ", end="")
        if metrics['benchmark_final_value'] > 0:
            print(f"{metrics['benchmark_cagr']*100:>14.2f}% {(metrics['cagr'] - metrics['benchmark_cagr'])*100:>14.2f}%")
        else:
            print(f"{'N/A':>15} {'N/A':>15}")
        
        print(f"{'Annualized Volatility':<25} {metrics['volatility']*100:>14.2f}% ", end="")
        if metrics['benchmark_final_value'] > 0:
            print(f"{metrics['benchmark_volatility']*100:>14.2f}% {(metrics['volatility'] - metrics['benchmark_volatility'])*100:>14.2f}%")
        else:
            print(f"{'N/A':>15} {'N/A':>15}")
        
        print(f"{'Sharpe Ratio (0% Rf)':<25} {metrics['sharpe_ratio']:>15.2f} ", end="")
        if metrics['benchmark_final_value'] > 0:
            print(f"{metrics['benchmark_sharpe_ratio']:>15.2f} {(metrics['sharpe_ratio'] - metrics['benchmark_sharpe_ratio']):>15.2f}")
        else:
            print(f"{'N/A':>15} {'N/A':>15}")
        
        print(f"{'Max Drawdown':<25} {metrics['max_drawdown']*100:>14.2f}% ", end="")
        if metrics['benchmark_final_value'] > 0:
            print(f"{metrics['benchmark_max_drawdown']*100:>14.2f}% {(metrics['max_drawdown'] - metrics['benchmark_max_drawdown'])*100:>14.2f}%")
        else:
            print(f"{'N/A':>15} {'N/A':>15}")
        
        print(f"{'Final Value':<25} ${metrics['final_value']:>13,.2f} ", end="")
        if metrics['benchmark_final_value'] > 0:
            print(f"${metrics['benchmark_final_value']:>13,.2f} ${(metrics['final_value'] - metrics['benchmark_final_value']):>13,.2f}")
        else:
            print(f"{'N/A':>15} {'N/A':>15}")
        
        print(f"{'Total Return':<25} {(metrics['final_value']/metrics['initial_capital'] - 1)*100:>14.2f}% ", end="")
        if metrics['benchmark_final_value'] > 0:
            print(f"{(metrics['benchmark_final_value']/metrics['initial_capital'] - 1)*100:>14.2f}% ", end="")
            print(f"{(metrics['final_value'] - metrics['benchmark_final_value'])/metrics['initial_capital']*100:>14.2f}%")
        else:
            print(f"{'N/A':>15} {'N/A':>15}")
        
        print(f"\nRebalances: {metrics.get('rebalance_count', 0)}")
        print("=" * 70)
    else:
        print("Backtest failed. Check error messages above.")
