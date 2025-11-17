"""
Configuration constants for the stock intelligence system.
Pure configuration file - no imports from other project modules.
"""

# Historical data period
START_DATE = "2010-01-01"
END_DATE = "2100-01-01"

# Data settings
DATA_DIR = "data"
PRICES_DIR = "data/prices"
FUNDAMENTALS_DIR = "data/fundamentals"
PLOTS_DIR = "plots"

# Rebalance frequency
REBALANCE_FREQUENCY = "M"  # Monthly rebalancing (pandas frequency code)

# Portfolio settings
TOP_N = 20  # Number of stocks to include in the model portfolio
MAX_POSITIONS = TOP_N  # Maximum number of positions in portfolio (alias for compatibility)
MIN_POSITION_SIZE = 0.01  # Minimum position weight (1%)
MAX_POSITION_SIZE = 0.10  # Maximum position weight (10%)

# Backtest settings
INITIAL_CAPITAL = 100000  # Starting capital for backtest
TRANSACTION_COST_BPS = 5  # Transaction cost in basis points (0.05% per trade)
TRANSACTION_COST = TRANSACTION_COST_BPS / 10000  # Transaction cost as fraction (for backward compatibility)

# Factor calculation parameters
MOMENTUM_LOOKBACK = 252  # Trading days (approximately 1 year)
VALUE_LOOKBACK = 252  # Trading days for value metrics

# Risk metrics
RISK_FREE_RATE = 0.02  # Annual risk-free rate (2%)
VOLATILITY_WINDOW = 252  # Trading days for volatility calculation

# Reporting
REPORT_DECIMAL_PLACES = 4  # Decimal places for reporting numbers

# Universe settings
MIN_MARKET_CAP = 5_000_000_000  # $5 billion minimum market cap
MIN_AVG_VOLUME = 1_000_000  # 1 million shares per day minimum


def get_benchmark_ticker():
    """
    Returns the benchmark ticker symbol for comparison.
    
    Returns:
        String ticker symbol (default: "SPY")
    """
    return "SPY"


def get_universe():
    """
    Returns a list of ticker symbols to analyze.
    
    This is a simple hardcoded list. For dynamic filtering based on
    market cap and liquidity, use universe_utils.get_filtered_universe().
    
    Returns:
        List of ticker symbols (strings)
    """
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
        'V', 'JNJ', 'WMT', 'JPM', 'MA', 'PG', 'UNH', 'HD', 'DIS', 'BAC',
        'ADBE', 'NFLX', 'CRM', 'PYPL', 'INTC', 'CMCSA', 'PEP', 'TMO', 'COST',
        'AVGO', 'CSCO', 'ABT', 'NKE', 'MRK', 'TXN', 'QCOM', 'ACN', 'DHR',
        'VZ', 'LIN', 'NEE', 'PM', 'HON', 'UNP', 'RTX', 'UPS', 'BMY', 'LOW',
        'AMGN', 'SPGI', 'INTU', 'T', 'DE', 'BKNG', 'AXP', 'SBUX', 'ADP',
        'GILD', 'MDT', 'ZTS', 'C', 'TJX', 'ISRG', 'SYK', 'CL', 'GE', 'MMC',
        'CAT', 'GS', 'MS', 'BLK', 'SCHW', 'CVX', 'XOM', 'COP', 'SLB', 'EOG'
    ]


# Backward compatibility - BENCHMARK constant
BENCHMARK = get_benchmark_ticker()
