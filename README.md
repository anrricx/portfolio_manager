# Portfolio Manager - Quantitative Stock Intelligence System

A Python-based quantitative algorithm for stock analysis, factor investing, and portfolio optimization.

## Features

- **Multi-Factor Stock Selection**: Combines momentum, value, and quality factors
- **Backtesting Engine**: Historical performance analysis with transaction costs and market regime filters
- **Portfolio Analysis**: Compare your holdings to model portfolio with drift analysis
- **Automated Reports**: Generate actionable buy/sell recommendations
- **Smart Data Caching**: Efficient data pipeline with automatic updates

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Download Historical Data
```bash
python data_pipeline.py
```
Downloads price and fundamental data for all stocks in the universe.

### 2. Run Backtest
```bash
python backtest.py
```
Runs historical backtest and generates performance metrics and equity curve plot.

### 3. Generate Portfolio Report
```bash
python reports.py
```
Generates a comprehensive report with buy/sell recommendations. Saves to `latest_report.txt`.

## Project Structure

```
QuantAlgo/
├── config.py              # Configuration constants
├── data_pipeline.py       # Data downloading and caching
├── factors.py             # Factor score calculations (momentum, value, quality)
├── backtest.py            # Backtesting engine
├── portfolio_analysis.py  # Portfolio comparison and drift analysis
├── reports.py             # Report generation
├── universe_utils.py      # Universe filtering utilities
├── data/
│   ├── prices/           # Cached price data (CSV files)
│   ├── fundamentals/     # Fundamental data
│   └── my_portfolio.csv  # Your portfolio (create this)
├── plots/                # Generated charts
└── requirements.txt      # Python dependencies
```

## Configuration

Edit `config.py` to customize:

- **Universe**: List of stocks to analyze (`get_universe()`)
- **TOP_N**: Number of positions in model portfolio (default: 20)
- **REBALANCE_FREQUENCY**: How often to rebalance (default: Monthly)
- **TRANSACTION_COST_BPS**: Transaction costs (default: 5 bps = 0.05%)
- **START_DATE**: Historical data start date (default: 2010-01-01)

## Factor Model

The system uses a three-factor model:

1. **Momentum** (40% weight)
   - 12-month momentum (excluding last 21 days)
   - 6-month momentum (excluding last 21 days)
   - Combined: 70% 12m + 30% 6m

2. **Value** (30% weight)
   - Inverse PE ratio
   - Inverse PB ratio
   - Winsorized and normalized

3. **Quality** (30% weight)
   - Return on Equity (ROE)
   - Gross Margin
   - Debt-to-Equity (inverse)
   - Earnings Stability

**Combined Score**: `0.4 * momentum + 0.3 * value + 0.3 * quality`

## Portfolio Management

### Adding Your Portfolio

Create `data/my_portfolio.csv` with your holdings:

```csv
ticker,quantity,price_paid
AAPL,10,150.00
MSFT,5,300.00
```

The system will:
- Load your portfolio automatically
- Calculate current values using latest prices
- Compare to model portfolio
- Generate buy/sell recommendations

### Report Output

The report includes:
- **Ranked BUY LIST**: Top 10 stocks to buy
- **Ranked SELL LIST**: Stocks to sell or trim
- **HOLD LIST**: Positions within ±1% of target
- **Portfolio Drift Table**: Current vs target weights
- **Market Regime**: RISK-ON / RISK-OFF analysis
- **Expected Returns**: Based on backtest CAGR
- **Actionable Notes**: Specific recommendations

## Backtesting Features

- Monthly rebalancing (last trading day of month)
- Transaction costs (0.05% per trade)
- Market regime filter (50% cash when SPY < 200-day SMA)
- Minimum rebalance threshold (1% weight change)
- Performance metrics: CAGR, Sharpe Ratio, Max Drawdown, Volatility

## Requirements

- Python 3.8+
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- yfinance >= 0.2.28

## Usage Examples

### Generate Report for Today
```bash
python reports.py
```

### Run Full Backtest
```bash
python backtest.py
```

### Update All Data
```bash
python data_pipeline.py
```



## Contributing

Contributions welcome! Please open an issue or pull request.

