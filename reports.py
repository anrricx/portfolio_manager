"""
Report generation module for creating human-readable terminal reports.
"""

import pandas as pd
import numpy as np
import sys
import os
from io import StringIO
from datetime import datetime
from config import REPORT_DECIMAL_PLACES, TOP_N, DATA_DIR
from data_pipeline import load_price_data
from config import get_benchmark_ticker


def load_portfolio_from_csv(csv_path=None):
    """
    Load portfolio holdings from CSV file.
    
    Expected CSV format:
        ticker,quantity,price_paid (or similar)
    
    Args:
        csv_path: Path to CSV file (default: data/my_portfolio.csv)
        
    Returns:
        Dictionary mapping ticker to current dollar value
    """
    if csv_path is None:
        csv_path = os.path.join(DATA_DIR, "my_portfolio.csv")
    
    if not os.path.exists(csv_path):
        return {}
    
    try:
        df = pd.read_csv(csv_path)
        
        # Handle different column name variations
        ticker_col = None
        quantity_col = None
        price_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'ticker' in col_lower or 'symbol' in col_lower:
                ticker_col = col
            elif 'quantity' in col_lower or 'shares' in col_lower or 'qty' in col_lower:
                quantity_col = col
            elif 'price' in col_lower or 'cost' in col_lower:
                price_col = col
        
        if ticker_col is None or quantity_col is None:
            print(f"Warning: Could not find required columns in {csv_path}")
            return {}
        
        current_holdings = {}
        
        for _, row in df.iterrows():
            ticker = str(row[ticker_col]).strip().upper()
            quantity = pd.to_numeric(row[quantity_col], errors='coerce')
            
            if pd.isna(quantity) or quantity <= 0:
                continue
            
            # Try to get current price, fall back to price_paid if available
            current_price = None
            try:
                price_data = load_price_data(ticker)
                if not price_data.empty and 'Adj Close' in price_data.columns:
                    current_price = price_data['Adj Close'].iloc[-1]
            except Exception:
                pass
            
            # If no current price, try to use price_paid column
            if current_price is None or pd.isna(current_price):
                if price_col and price_col in row:
                    current_price = pd.to_numeric(row[price_col], errors='coerce')
            
            # If still no price, skip this ticker
            if current_price is None or pd.isna(current_price):
                print(f"Warning: Could not get price for {ticker}, skipping")
                continue
            
            # Calculate current value
            current_value = quantity * current_price
            current_holdings[ticker] = current_value
        
        return current_holdings
    
    except Exception as e:
        print(f"Error loading portfolio from {csv_path}: {e}")
        return {}


def format_number(value, decimals=REPORT_DECIMAL_PLACES, is_percent=False):
    """Format a number for display."""
    if pd.isna(value) or value is None:
        return "N/A"
    
    if is_percent:
        return f"{value * 100:.{decimals}f}%"
    else:
        return f"{value:.{decimals}f}"


def print_section_header(title, width=80):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def get_market_regime(benchmark_ticker='SPY'):
    """
    Determine market regime based on SPY vs 200-day SMA.
    
    Returns:
        Tuple of (regime_string, spy_price, spy_sma, is_risk_on)
    """
    try:
        spy_data = load_price_data(benchmark_ticker)
        if spy_data.empty or 'Adj Close' not in spy_data.columns:
            return ("UNKNOWN", None, None, None)
        
        spy_prices = spy_data['Adj Close']
        if len(spy_prices) < 200:
            return ("INSUFFICIENT_DATA", None, None, None)
        
        spy_price = spy_prices.iloc[-1]
        spy_sma_200 = spy_prices.tail(200).mean()
        
        is_risk_on = spy_price >= spy_sma_200
        regime = "RISK-ON" if is_risk_on else "RISK-OFF"
        
        return (regime, spy_price, spy_sma_200, is_risk_on)
    except Exception:
        return ("UNKNOWN", None, None, None)


def get_volatility_environment(backtest_metrics):
    """
    Determine volatility environment based on historical volatility.
    
    Returns:
        Tuple of (environment_string, volatility_value)
    """
    if not backtest_metrics:
        return ("UNKNOWN", None)
    
    volatility = backtest_metrics.get('volatility', 0)
    
    # High volatility threshold: > 20% annualized
    if volatility > 0.20:
        return ("HIGH", volatility)
    else:
        return ("NORMAL", volatility)


def calculate_outperformance_probability(sharpe_ratio):
    """
    Estimate probability of outperformance based on Sharpe ratio.
    
    Uses a simple heuristic: higher Sharpe = higher probability of outperformance.
    
    Returns:
        Probability as decimal (0-1)
    """
    if sharpe_ratio is None or pd.isna(sharpe_ratio):
        return 0.50  # Neutral if unknown
    
    # Simple heuristic: Sharpe > 1.0 suggests > 60% chance of outperformance
    # Sharpe < 0 suggests < 40% chance
    if sharpe_ratio > 1.5:
        prob = 0.70 + min(0.20, (sharpe_ratio - 1.5) * 0.10)
    elif sharpe_ratio > 1.0:
        prob = 0.60 + (sharpe_ratio - 1.0) * 0.20
    elif sharpe_ratio > 0.5:
        prob = 0.50 + (sharpe_ratio - 0.5) * 0.20
    elif sharpe_ratio > 0:
        prob = 0.45 + sharpe_ratio * 0.10
    else:
        prob = 0.40 + max(0, sharpe_ratio + 0.5) * 0.20
    
    return min(0.90, max(0.10, prob))


def print_buy_list_ranked(factor_scores_df, top_n=10):
    """
    Print ranked BUY LIST (top N stocks by combined score).
    
    Args:
        factor_scores_df: DataFrame with factor scores (from compute_factor_scores)
        top_n: Number of top stocks to show (default 10)
    """
    if factor_scores_df.empty:
        print("\nNo buy recommendations available.")
        return
    
    print_section_header(f"RANKED BUY LIST (Top {top_n})")
    
    # Get top N stocks (already sorted by combined_score descending)
    top_stocks = factor_scores_df.head(top_n).copy()
    
    print(f"{'Rank':<6} {'Ticker':<10} {'Combined':<12} {'Momentum':<12} {'Value':<12} {'Quality':<12} {'Action':<20}")
    print("-" * 90)
    
    for rank, (idx, row) in enumerate(top_stocks.iterrows(), 1):
        ticker = row['ticker']
        combined = row.get('combined_score', 0)
        momentum = row.get('momentum_score', 0)
        value = row.get('value_score', 0)
        quality = row.get('quality_score', 0)
        
        # Determine action
        if rank <= 3:
            action = "STRONG BUY"
        elif rank <= 7:
            action = "BUY"
        else:
            action = "CONSIDER BUY"
        
        print(f"{rank:<6} {ticker:<10} {format_number(combined, 4):<12} "
              f"{format_number(momentum, 4):<12} {format_number(value, 4):<12} "
              f"{format_number(quality, 4):<12} {action:<20}")


def print_sell_list_ranked(factor_scores_df, top_n_model=TOP_N, bottom_n=10):
    """
    Print ranked SELL LIST (bottom stocks or stocks not in top N).
    
    Args:
        factor_scores_df: DataFrame with factor scores
        top_n_model: Number of stocks in model portfolio (default TOP_N)
        bottom_n: Number of bottom stocks to show (default 10)
    """
    if factor_scores_df.empty:
        print("\nNo sell recommendations available.")
        return
    
    # Get stocks not in top N (these should be sold if held)
    all_stocks = factor_scores_df.copy()
    top_stocks_tickers = set(all_stocks.head(top_n_model)['ticker'].tolist())
    
    # Get bottom N stocks and stocks not in top N
    bottom_stocks = all_stocks.tail(bottom_n)
    not_in_top = all_stocks[~all_stocks['ticker'].isin(top_stocks_tickers)]
    
    # Combine and remove duplicates
    sell_candidates = pd.concat([not_in_top, bottom_stocks]).drop_duplicates(subset=['ticker'])
    
    # Sort by combined score (ascending - worst first)
    sell_candidates = sell_candidates.sort_values('combined_score', ascending=True)
    
    if sell_candidates.empty:
        print("\nNo sell recommendations (all holdings are in top model portfolio).")
        return
    
    print_section_header(f"RANKED SELL LIST (Bottom {min(bottom_n, len(sell_candidates))} or Not in Top {top_n_model})")
    
    print(f"{'Rank':<6} {'Ticker':<10} {'Combined':<12} {'Momentum':<12} {'Value':<12} {'Quality':<12} {'Action':<20}")
    print("-" * 90)
    
    total_stocks = len(all_stocks)
    for idx, (_, row) in enumerate(sell_candidates.head(bottom_n).iterrows()):
        # Calculate rank (from bottom)
        rank = total_stocks - idx
        ticker = row['ticker']
        combined = row.get('combined_score', 0)
        momentum = row.get('momentum_score', 0)
        value = row.get('value_score', 0)
        quality = row.get('quality_score', 0)
        
        # Determine action
        if idx < 3:
            action = "STRONG SELL"
        elif idx < 7:
            action = "SELL"
        else:
            action = "CONSIDER SELL"
        
        print(f"{rank:<6} {ticker:<10} {format_number(combined, 4):<12} "
              f"{format_number(momentum, 4):<12} {format_number(value, 4):<12} "
              f"{format_number(quality, 4):<12} {action:<20}")


def print_hold_list(current_weights, model_weights, threshold=0.01):
    """
    Print HOLD LIST for stocks within +/- threshold of target weight.
    
    Args:
        current_weights: Dictionary of current portfolio weights
        model_weights: Dictionary of model portfolio weights
        threshold: Weight difference threshold (default 0.01 = 1%)
    """
    hold_stocks = []
    
    all_tickers = set(current_weights.keys()) | set(model_weights.keys())
    
    for ticker in all_tickers:
        current = current_weights.get(ticker, 0)
        model = model_weights.get(ticker, 0)
        
        # Only include stocks that are currently held
        if current > 0:
            weight_diff = abs(current - model)
            if weight_diff <= threshold:
                hold_stocks.append({
                    'ticker': ticker,
                    'current_weight': current,
                    'target_weight': model,
                    'weight_difference': weight_diff
                })
    
    if not hold_stocks:
        print("\nNo stocks in HOLD status (all positions need rebalancing).")
        return
    
    print_section_header(f"HOLD LIST (Within ±{threshold*100:.0f}% of Target Weight)")
    
    # Sort by current weight (descending)
    hold_stocks.sort(key=lambda x: x['current_weight'], reverse=True)
    
    print(f"{'Ticker':<10} {'Current Weight':<18} {'Target Weight':<18} {'Difference':<15}")
    print("-" * 65)
    
    for stock in hold_stocks:
        print(f"{stock['ticker']:<10} {format_number(stock['current_weight'], 4, True):<18} "
              f"{format_number(stock['target_weight'], 4, True):<18} "
              f"{format_number(stock['weight_difference'], 4, True):<15}")


def print_portfolio_drift_table(current_weights, model_weights):
    """
    Print portfolio drift table showing all positions.
    
    Args:
        current_weights: Dictionary of current portfolio weights
        model_weights: Dictionary of model portfolio weights
    """
    all_tickers = set(current_weights.keys()) | set(model_weights.keys())
    
    drift_data = []
    for ticker in all_tickers:
        current = current_weights.get(ticker, 0)
        model = model_weights.get(ticker, 0)
        diff = current - model
        
        drift_data.append({
            'ticker': ticker,
            'current_weight': current,
            'target_weight': model,
            'weight_difference': diff
        })
    
    if not drift_data:
        print("\nNo portfolio data available.")
        return
    
    # Convert to DataFrame and sort by absolute difference
    drift_df = pd.DataFrame(drift_data)
    drift_df['abs_diff'] = drift_df['weight_difference'].abs()
    drift_df = drift_df.sort_values('abs_diff', ascending=False)
    
    print_section_header("PORTFOLIO DRIFT TABLE")
    
    print(f"{'Ticker':<10} {'Current Weight':<18} {'Target Weight':<18} {'Weight Difference':<20}")
    print("-" * 70)
    
    for _, row in drift_df.iterrows():
        ticker = row['ticker']
        current = row['current_weight']
        target = row['target_weight']
        diff = row['weight_difference']
        
        # Only show positions that exist (current > 0 or target > 0)
        if current > 0 or target > 0:
            diff_str = format_number(diff, 4, True)
            if diff > 0:
                diff_str = f"+{diff_str}"  # Show positive with +
            
            print(f"{ticker:<10} {format_number(current, 4, True):<18} "
                  f"{format_number(target, 4, True):<18} {diff_str:<20}")


def print_market_regime_analysis(benchmark_ticker='SPY'):
    """Print market regime analysis."""
    regime, spy_price, spy_sma, is_risk_on = get_market_regime(benchmark_ticker)
    
    print_section_header("MARKET REGIME ANALYSIS")
    
    print(f"Market Regime:         {regime}")
    
    if spy_price is not None and spy_sma is not None:
        print(f"SPY Price:             ${spy_price:.2f}")
        print(f"SPY 200-day SMA:       ${spy_sma:.2f}")
        deviation = ((spy_price - spy_sma) / spy_sma) * 100
        print(f"Deviation from SMA:    {deviation:+.2f}%")
        
        if is_risk_on:
            print("\nInterpretation: Market is in RISK-ON mode. Consider full exposure.")
        else:
            print("\nInterpretation: Market is in RISK-OFF mode. Consider reduced exposure (50%).")
    else:
        print("Unable to determine market regime (insufficient data).")


def print_volatility_environment(backtest_metrics):
    """Print volatility environment analysis."""
    env, volatility = get_volatility_environment(backtest_metrics)
    
    print_section_header("VOLATILITY ENVIRONMENT")
    
    print(f"Environment:           {env}")
    
    if volatility is not None:
        print(f"Annualized Volatility: {format_number(volatility, 4, True)}")
        
        if env == "HIGH":
            print("\nInterpretation: High volatility environment detected. ")
            print("                Consider reducing position sizes or increasing cash allocation.")
        else:
            print("\nInterpretation: Normal volatility environment. Standard allocation appropriate.")
    else:
        print("Unable to determine volatility environment.")


def print_expected_returns(backtest_metrics):
    """Print expected returns and performance outlook."""
    if not backtest_metrics:
        print("\nNo backtest metrics available.")
        return
    
    print_section_header("EXPECTED RETURNS & PERFORMANCE OUTLOOK")
    
    cagr = backtest_metrics.get('cagr', 0)
    sharpe = backtest_metrics.get('sharpe_ratio', 0)
    volatility = backtest_metrics.get('volatility', 0)
    
    print(f"Expected Long-term Return (CAGR):  {format_number(cagr, 4, True)}")
    print(f"Historical Sharpe Ratio:          {format_number(sharpe, 4)}")
    print(f"Annualized Volatility:            {format_number(volatility, 4, True)}")
    
    # Calculate probability of outperformance
    prob_outperform = calculate_outperformance_probability(sharpe)
    print(f"Estimated Outperformance Probability: {format_number(prob_outperform, 2, True)}")
    
    if 'benchmark_cagr' in backtest_metrics:
        benchmark_cagr = backtest_metrics.get('benchmark_cagr', 0)
        excess_return = cagr - benchmark_cagr
        print(f"\nBenchmark Expected Return:        {format_number(benchmark_cagr, 4, True)}")
        print(f"Expected Excess Return:           {format_number(excess_return, 4, True)}")


def print_actionable_notes(portfolio_analysis, backtest_metrics, market_regime):
    """Print actionable notes and recommendations."""
    print_section_header("ACTIONABLE NOTES & RECOMMENDATIONS")
    
    notes = []
    
    # Market regime notes
    regime, _, _, is_risk_on = market_regime if isinstance(market_regime, tuple) else get_market_regime()
    if not is_risk_on:
        notes.append("⚠️  Current environment suggests CAUTION: Market is in RISK-OFF mode.")
        notes.append("   Consider reducing equity exposure to 50% and holding 50% cash.")
    elif is_risk_on:
        notes.append("✓  Market is in RISK-ON mode. Full exposure recommended.")
    
    # Volatility notes
    env, volatility = get_volatility_environment(backtest_metrics)
    if env == "HIGH":
        notes.append("⚠️  High volatility environment detected.")
        notes.append("   Consider reducing position sizes or increasing cash allocation.")
    
    # Portfolio drift notes
    drift_metrics = portfolio_analysis.get('drift_metrics', {})
    drift_magnitude = drift_metrics.get('drift_magnitude', 0)
    if drift_magnitude > 0.20:
        notes.append("⚠️  Significant portfolio drift detected (>20%).")
        notes.append("   Consider rebalancing to align with model portfolio.")
    elif drift_magnitude > 0.10:
        notes.append("ℹ️  Moderate portfolio drift detected (10-20%).")
        notes.append("   Monitor positions and consider gradual rebalancing.")
    
    # Buy list notes
    trade_suggestions = portfolio_analysis.get('trade_suggestions', {})
    buy_list = trade_suggestions.get('buy', [])
    if len(buy_list) > 0:
        top_buy = buy_list[0] if buy_list else None
        if top_buy:
            notes.append(f"💡 Consider increasing exposure to {top_buy['ticker']} "
                        f"(target weight: {format_number(top_buy['target_weight'], 2, True)}).")
    
    # Sell list notes
    sell_list = trade_suggestions.get('sell', [])
    if len(sell_list) > 0:
        top_sell = sell_list[0] if sell_list else None
        if top_sell:
            notes.append(f"💡 Consider trimming or selling {top_sell['ticker']} "
                        f"(current weight: {format_number(top_sell['current_weight'], 2, True)}).")
    
    # Performance notes
    if backtest_metrics:
        sharpe = backtest_metrics.get('sharpe_ratio', 0)
        if sharpe > 1.5:
            notes.append("✓  Strong historical risk-adjusted returns (Sharpe > 1.5).")
            notes.append("   Strategy has shown consistent outperformance.")
        elif sharpe < 0.5:
            notes.append("⚠️  Low historical risk-adjusted returns (Sharpe < 0.5).")
            notes.append("   Review strategy parameters or consider alternative approaches.")
    
    if not notes:
        notes.append("No specific recommendations at this time.")
        notes.append("Portfolio appears well-aligned with model.")
    
    for note in notes:
        print(f"  {note}")
    
    print("\n⚠️  IMPORTANT: This is INFORMATION-ONLY. Do NOT execute trades automatically.")
    print("   Review all recommendations carefully before making any investment decisions.")


def generate_full_report(portfolio_analysis=None, backtest_results=None, factor_scores_df=None):
    """
    Generate a complete human-readable report with clear recommendations.
    
    Args:
        portfolio_analysis: Dictionary from portfolio_analysis.analyze_portfolio() (optional)
        backtest_results: Dictionary from backtest.run_backtest() (optional)
        factor_scores_df: DataFrame with factor scores (optional)
    
    Returns:
        String containing the full report
    """
    # Capture stdout to string
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    try:
        # If no portfolio_analysis provided, generate it
        if portfolio_analysis is None:
            from portfolio_analysis import analyze_portfolio
            from factors import compute_factor_scores
            from datetime import datetime
            
            # Get current date
            today = datetime.now().strftime('%Y-%m-%d')
            
            # Compute factor scores for today
            factor_scores = compute_factor_scores(today)
            
            # Load portfolio from CSV file
            current_holdings = load_portfolio_from_csv()
            
            if current_holdings:
                total_value = sum(current_holdings.values())
                print(f"\nLoaded portfolio from CSV: {len(current_holdings)} positions, total value: ${total_value:,.2f}")
            else:
                print("\nNo portfolio found in CSV. Using empty portfolio for analysis.")
            
            # Analyze portfolio
            portfolio_analysis = analyze_portfolio(current_holdings, factor_scores, date=today)
            
            # Use factor_scores as factor_scores_df if not provided
            if factor_scores_df is None:
                factor_scores_df = factor_scores
        
        # If no backtest_results provided, run backtest
        if backtest_results is None:
            from backtest import run_backtest
            backtest_results = run_backtest()
        
        print("\n" + "=" * 90)
        print("  STOCK INTELLIGENCE SYSTEM - PORTFOLIO REPORT")
        print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 90)
        
        # Extract components
        current_weights = portfolio_analysis.get('current_weights', {})
        model_weights = portfolio_analysis.get('model_weights', {})
        trade_suggestions = portfolio_analysis.get('trade_suggestions', {})
        backtest_metrics = backtest_results.get('metrics', {}) if backtest_results else {}
        
        # Get factor scores DataFrame (use from portfolio_analysis if not provided)
        if factor_scores_df is None:
            factor_scores_df = portfolio_analysis.get('factor_scores_df')
        
        # Get market regime
        benchmark_ticker = get_benchmark_ticker()
        market_regime = get_market_regime(benchmark_ticker)
        
        # Print sections in order
        print_market_regime_analysis(benchmark_ticker)
        print_volatility_environment(backtest_metrics)
        print_expected_returns(backtest_metrics)
        
        # Rankings and recommendations
        if factor_scores_df is not None and not factor_scores_df.empty:
            print_buy_list_ranked(factor_scores_df, top_n=10)
            print_sell_list_ranked(factor_scores_df, top_n_model=TOP_N, bottom_n=10)
        else:
            print("\n⚠️  Factor scores not available. Cannot generate ranked buy/sell lists.")
        
        print_hold_list(current_weights, model_weights, threshold=0.01)
        print_portfolio_drift_table(current_weights, model_weights)
        
        # Actionable notes
        print_actionable_notes(portfolio_analysis, backtest_metrics, market_regime)
        
        print("\n" + "=" * 90)
        print("  END OF REPORT")
        print("=" * 90 + "\n")
        
        # Get the captured output
        report_text = captured_output.getvalue()
        
    finally:
        # Restore stdout
        sys.stdout = old_stdout
    
    return report_text


if __name__ == "__main__":
    report = generate_full_report()
    print(report)
    
    # Save a copy to a file
    with open("latest_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
