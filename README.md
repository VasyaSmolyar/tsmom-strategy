# TSMOM Backtest Project

## Overview
This project implements and backtests a Time Series Momentum (TSMOM) strategy based on the approach described in Moskowitz, Ooi, and Pedersen (2012). The strategy opens long positions for assets with positive momentum and short positions for assets with negative momentum based on 12-month lookback returns.

## 🚀 Quick Start

### Installation
```bash
# Install dependencies
pip install -r requirements.txt
```

### Run Backtest
```bash
# Quick start (uses IMOEX benchmark for MOEX data)
python run_backtest.py

# MOEX-specific backtest with IMOEX benchmark
python run_moex_backtest.py

# Detailed IMOEX comparison
python compare_with_imoex.py

# Or use main module
python src/main.py
```

## 📊 Strategy Details

### Core Strategy
- **Type**: Time Series Momentum (TSMOM)
- **Lookback Period**: 12 months
- **Holding Period**: 1 month
- **Target Volatility**: 40% annual
- **Rebalancing**: Monthly
- **Position Sizing**: Inverse volatility weighting

### Asset Universe
The strategy uses 20+ assets across multiple asset classes:
- **Equity Indices**: S&P 500, NASDAQ, Dow Jones, FTSE 100, DAX, CAC 40
- **Bonds**: Treasury yields, bond ETFs
- **Currencies**: Major FX pairs (EUR/USD, GBP/USD, etc.)
- **Commodities**: Gold, oil, agricultural futures
- **Russian Futures**: MOEX futures (Si, GOLD, RTS, GAZR, etc.)

## 📁 Project Structure

```
tsmom_backtest/
├── config/                 # Configuration files
│   └── config.yaml        # Strategy parameters
├── data/                  # Data storage
│   ├── raw/              # Raw downloaded data
│   │   └── moex/         # MOEX futures data files
│   └── processed/        # Cleaned data
├── src/                   # Source code
│   ├── data/             # Data handling modules
│   ├── strategy/         # TSMOM strategy implementation
│   ├── analysis/         # Performance analysis
│   ├── utils/            # Utility functions
│   └── main.py           # Main execution script
├── notebooks/            # Jupyter notebooks for analysis
├── reports/              # Generated reports and plots
├── tests/                # Unit tests
├── docs/                 # Documentation
├── requirements.txt       # Python dependencies
├── run_backtest.py       # Quick start script
└── README.md            # This file
```

## 🔧 Key Features

### Data Management
- ✅ Automatic data download from Yahoo Finance
- ✅ **NEW**: MOEX futures data loading from local CSV files
- ✅ **NEW**: Comprehensive data integrity checks
- ✅ Data cleaning and validation
- ✅ Support for multiple asset classes
- ✅ Historical data from 2000 to present
- ✅ **NEW**: Support for 16+ Russian futures contracts

### Strategy Implementation
- ✅ Time Series Momentum signals
- ✅ Inverse volatility position sizing
- ✅ Transaction cost modeling
- ✅ Risk management constraints

### Performance Analysis
- ✅ Comprehensive performance metrics
- ✅ Benchmark comparison (S&P 500 / IMOEX for MOEX data)
- ✅ Risk analysis (VaR, CVaR, drawdown)
- ✅ Rolling performance analysis
- ✅ Sensitivity analysis

### Visualization & Reporting
- ✅ Cumulative returns plots
- ✅ Drawdown analysis
- ✅ Returns distribution analysis
- ✅ Rolling metrics visualization
- ✅ Markdown reports generation

## 📈 Performance Metrics

The analysis provides comprehensive metrics:

### Return Metrics
- Total Return, Annual Return, Excess Return

### Risk Metrics
- Volatility, Maximum Drawdown, VaR, CVaR

### Risk-Adjusted Metrics
- Sharpe Ratio, Sortino Ratio, Calmar Ratio, Information Ratio

### Distribution Metrics
- Skewness, Kurtosis, Win Rate, Profit Factor

## 🎯 Usage Examples

### Basic Backtest
```bash
python run_backtest.py
```

### Advanced Usage
```bash
# Run with custom configuration
python src/main.py --config my_config.yaml

# Run sensitivity analysis
python src/main.py --sensitivity

# Run without downloading fresh data
python src/main.py --no-download

# Debug mode
python src/main.py --log-level DEBUG
```

### Jupyter Analysis
```bash
jupyter notebook notebooks/tsmom_analysis.ipynb
```

## 📋 Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- yfinance (for data download)
- scipy, statsmodels (for analysis)
- pytest (for testing)

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## 📚 Documentation

- [Usage Guide](docs/USAGE.md) - Detailed usage instructions
- [Configuration](config/config.yaml) - Strategy parameters
- [Jupyter Notebook](notebooks/tsmom_analysis.ipynb) - Interactive analysis

## 🔬 Research Background

This implementation is based on:
- **Paper**: "Time Series Momentum" by Moskowitz, Ooi, and Pedersen (2012)
- **Strategy**: Momentum-based asset allocation across multiple asset classes
- **Key Insight**: Past performance predicts future performance across time series

## 📊 Output Files

### Reports
- `reports/performance_report.md` - Comprehensive performance report
- `reports/tables/performance_metrics.csv` - Performance metrics table

### Plots
- `reports/plots/cumulative_returns.png` - Strategy vs benchmark
- `reports/plots/drawdown.png` - Drawdown analysis
- `reports/plots/returns_distribution.png` - Returns distribution
- `reports/plots/rolling_metrics.png` - Rolling performance

### Data
- `data/raw/raw_prices.csv` - Raw price data
- `data/processed/cleaned_prices.csv` - Cleaned data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Moskowitz, T., Ooi, Y. H., & Pedersen, L. H. (2012). Time series momentum. Journal of Financial Economics, 104(2), 228-250.
- Yahoo Finance for data access
- Python community for excellent libraries

---

**Note**: This is a research and educational tool. Past performance does not guarantee future results. Use at your own risk for actual trading decisions. 