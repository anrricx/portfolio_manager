"""
Portfolio analysis module for comparing current holdings to model portfolio.
"""

import pandas as pd
import numpy as np
from config import MAX_POSITIONS, MIN_POSITION_SIZE, MAX_POSITION_SIZE
from factors import get_top_stocks


def calculate_current_weights(current_holdings):
    """
    Calculate portfolio weights from current holdings.
    
    Args:
        current_holdings: Dictionary mapping ticker to number of shares or value
        
    Returns:
        Dictionary mapping ticker to weight (0-1)
    """
    total_value = sum(current_holdings.values())
    
    if total_value == 0:
        return {}
    
    weights = {ticker: value / total_value for ticker, value in current_holdings.items()}
    return weights


def get_model_portfolio(factor_scores, date=None, top_n=MAX_POSITIONS):
    """
    Get model portfolio weights based on factor scores.
    
    Args:
        factor_scores: DataFrame with factor scores
        date: Date to use for portfolio (if None, uses most recent)
        top_n: Number of positions to hold
        
    Returns:
        Dictionary mapping ticker to target weight
    """
    top_stocks_df = get_top_stocks(factor_scores, date=date, top_n=top_n, factor='combined_score')
    
    if top_stocks_df.empty:
        return {}
    
    # Equal weight portfolio
    weight_per_stock = 1.0 / len(top_stocks_df)
    
    model_portfolio = {}
    for ticker in top_stocks_df['ticker']:
        model_portfolio[ticker] = weight_per_stock
    
    return model_portfolio


def calculate_portfolio_drift(current_weights, model_weights):
    """
    Calculate drift between current and model portfolio.
    
    Args:
        current_weights: Dictionary of current portfolio weights
        model_weights: Dictionary of model portfolio weights
        
    Returns:
        Dictionary with drift metrics
    """
    all_tickers = set(current_weights.keys()) | set(model_weights.keys())
    
    drift = {}
    total_absolute_drift = 0
    
    for ticker in all_tickers:
        current = current_weights.get(ticker, 0)
        model = model_weights.get(ticker, 0)
        diff = current - model
        drift[ticker] = diff
        total_absolute_drift += abs(diff)
    
    return {
        'drift_by_ticker': drift,
        'total_absolute_drift': total_absolute_drift,
        'drift_magnitude': total_absolute_drift / 2,  # Normalized to 0-1
    }


def generate_trade_suggestions(current_weights, model_weights, 
                               hold_threshold=0.01, rebalance_threshold=0.01):
    """
    Generate buy/sell suggestions to rebalance portfolio.
    
    Args:
        current_weights: Dictionary of current portfolio weights
        model_weights: Dictionary of model portfolio weights
        hold_threshold: Weight difference threshold for HOLD (default 0.01 = 1%)
        rebalance_threshold: Threshold for considering a rebalance (default 0.01 = 1%)
        
    Returns:
        Dictionary with 'buy', 'sell', 'trim', 'hold' lists
    """
    all_tickers = set(current_weights.keys()) | set(model_weights.keys())
    
    buy_list = []
    sell_list = []
    trim_list = []
    hold_list = []
    
    for ticker in all_tickers:
        current = current_weights.get(ticker, 0)
        model = model_weights.get(ticker, 0)
        diff = model - current
        abs_diff = abs(diff)
        
        # HOLD: within +/- 1% of target weight
        if abs_diff <= hold_threshold and current > 0:
            hold_list.append({
                'ticker': ticker,
                'current_weight': current,
                'target_weight': model,
                'weight_difference': diff
            })
            continue
        
        if model > 0 and current == 0:
            # New position to buy
            buy_list.append({
                'ticker': ticker,
                'target_weight': model,
                'current_weight': 0,
                'action': 'buy'
            })
        elif model == 0 and current > 0:
            # Position to sell completely
            sell_list.append({
                'ticker': ticker,
                'target_weight': 0,
                'current_weight': current,
                'action': 'sell'
            })
        elif diff > rebalance_threshold:
            # Need to increase position
            buy_list.append({
                'ticker': ticker,
                'target_weight': model,
                'current_weight': current,
                'action': 'add'
            })
        elif diff < -rebalance_threshold:
            # Need to decrease position
            if current > model * 0.5 and model > 0:
                # Trim (reduce but keep)
                trim_list.append({
                    'ticker': ticker,
                    'target_weight': model,
                    'current_weight': current,
                    'action': 'trim'
                })
            else:
                # Sell (reduce significantly or eliminate)
                sell_list.append({
                    'ticker': ticker,
                    'target_weight': model,
                    'current_weight': current,
                    'action': 'reduce'
                })
    
    # Sort by magnitude of change
    buy_list.sort(key=lambda x: abs(x['target_weight'] - x['current_weight']), reverse=True)
    sell_list.sort(key=lambda x: abs(x['target_weight'] - x['current_weight']), reverse=True)
    trim_list.sort(key=lambda x: abs(x['target_weight'] - x['current_weight']), reverse=True)
    
    return {
        'buy': buy_list,
        'sell': sell_list,
        'trim': trim_list,
        'hold': hold_list,
    }


def analyze_portfolio(current_holdings, factor_scores, date=None):
    """
    Complete portfolio analysis comparing current holdings to model portfolio.
    
    Args:
        current_holdings: Dictionary mapping ticker to value (or shares * price)
        factor_scores: DataFrame with factor scores
        date: Date to use for analysis (if None, uses most recent)
        
    Returns:
        Dictionary with analysis results
    """
    # Calculate current weights
    current_weights = calculate_current_weights(current_holdings)
    
    # Get model portfolio
    model_weights = get_model_portfolio(factor_scores, date=date)
    
    # Calculate drift
    drift_metrics = calculate_portfolio_drift(current_weights, model_weights)
    
    # Generate trade suggestions
    trade_suggestions = generate_trade_suggestions(current_weights, model_weights)
    
    # Get rankings
    from factors import get_top_stocks
    # Determine number of tickers to rank
    if not factor_scores.empty:
        if 'ticker' in factor_scores.columns:
            n_tickers = len(factor_scores['ticker'].unique())
        else:
            n_tickers = 100
    else:
        n_tickers = 100
    rankings_df = get_top_stocks(factor_scores, date=date, top_n=n_tickers, factor='combined_score')
    
    rankings = {}
    if not rankings_df.empty:
        for idx, row in rankings_df.iterrows():
            ticker = row['ticker']
            rankings[ticker] = {
                'rank': idx + 1,
                'combined_score': row.get('combined_score', row.get('combined', 0)),
                'momentum_score': row.get('momentum_score', row.get('momentum', 0)),
                'value_score': row.get('value_score', row.get('value', 0)),
            }
    
    # Get full factor scores DataFrame for reporting (if available)
    factor_scores_df = None
    if isinstance(factor_scores, pd.DataFrame):
        factor_scores_df = factor_scores.copy()
        
        # Filter to the specified date if provided
        if date is not None:
            if 'date' in factor_scores_df.columns:
                factor_scores_df = factor_scores_df[factor_scores_df['date'] == date]
            elif 'as_of_date' in factor_scores_df.columns:
                factor_scores_df = factor_scores_df[factor_scores_df['as_of_date'] == date]
            elif isinstance(factor_scores_df.index, pd.DatetimeIndex):
                # If index is datetime, filter by date
                factor_scores_df = factor_scores_df[factor_scores_df.index == date]
        
        # Ensure we have the right columns
        if 'ticker' in factor_scores_df.columns:
            # Sort by combined_score descending
            if 'combined_score' in factor_scores_df.columns:
                factor_scores_df = factor_scores_df.sort_values('combined_score', ascending=False)
            elif 'combined' in factor_scores_df.columns:
                # Handle old column name
                factor_scores_df = factor_scores_df.rename(columns={'combined': 'combined_score'})
                factor_scores_df = factor_scores_df.sort_values('combined_score', ascending=False)
    
    return {
        'current_weights': current_weights,
        'model_weights': model_weights,
        'drift_metrics': drift_metrics,
        'trade_suggestions': trade_suggestions,
        'rankings': rankings,
        'factor_scores_df': factor_scores_df,  # Add for reporting
    }


if __name__ == "__main__":
    print("Portfolio analysis module loaded. Use analyze_portfolio() to analyze your holdings.")

