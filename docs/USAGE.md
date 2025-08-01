# TSMOM Backtest Project - Usage Guide

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd tsmom_backtest

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Backtest

#### Option A: Quick Start Script
```bash
python run_backtest.py
```

#### Option B: Main Module
```bash
python src/main.py
```

#### Option C: Command Line Interface
```bash
# Full backtest with data download and report generation
python src/main.py --config config/config.yaml

# Run without downloading fresh data
python src/main.py --no-download

# Run without generating reports
python src/main.py --no-report

# Run sensitivity analysis
python src/main.py --sensitivity

# Set log level
python src/main.py --log-level DEBUG
```

### 3. Jupyter Notebook Analysis

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/tsmom_analysis.ipynb
```

## Project Structure

```
tsmom_backtest/
├── config/                 # Configuration files
│   └── config.yaml        # Main configuration
├── data/                  # Data storage
│   ├── raw/              # Raw downloaded data
│   └── processed/        # Cleaned data
├── src/                   # Source code
│   ├── data/             # Data handling
│   ├── strategy/         # TSMOM strategy
│   ├── analysis/         # Performance analysis
│   ├── utils/            # Utilities
│   └── main.py           # Main execution
├── notebooks/            # Jupyter notebooks
├── reports/              # Generated reports
├── tests/                # Unit tests
├── requirements.txt       # Dependencies
├── run_backtest.py       # Quick start script
└── README.md            # Project documentation
```

## Configuration

The main configuration file is `config/config.yaml`. Key parameters:

### Strategy Parameters
- `lookback_period`: 12 months (momentum calculation period)
- `holding_period`: 1 month (position holding period)
- `target_volatility`: 40% (annual target volatility)
- `transaction_cost`: 0.1% (per trade cost)

### Asset Universe
The strategy uses 20+ assets across:
- Equity indices (S&P 500, NASDAQ, etc.)
- Bonds (Treasury yields, bond ETFs)
- Currencies (major FX pairs)
- Commodities (gold, oil, agricultural)

### Data Parameters
- `start_date`: 2000-01-01
- `end_date`: 2023-12-31
- `frequency`: Daily data

## Output Files

### Reports
- `reports/performance_report.md`: Comprehensive performance report
- `reports/tables/performance_metrics.csv`: Performance metrics table

### Plots
- `reports/plots/cumulative_returns.png`: Strategy vs benchmark returns
- `reports/plots/drawdown.png`: Drawdown analysis
- `reports/plots/returns_distribution.png`: Returns distribution
- `reports/plots/rolling_metrics.png`: Rolling performance metrics

### Data
- `data/raw/raw_prices.csv`: Raw downloaded price data
- `data/processed/cleaned_prices.csv`: Cleaned price data

### Logs
- `tsmom_backtest.log`: Execution log file

## Performance Metrics

The analysis provides comprehensive metrics:

### Return Metrics
- Total Return
- Annual Return
- Excess Return (vs benchmark)

### Risk Metrics
- Volatility
- Maximum Drawdown
- Value at Risk (VaR)
- Conditional VaR (CVaR)

### Risk-Adjusted Metrics
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Information Ratio

### Distribution Metrics
- Skewness
- Kurtosis
- Win Rate
- Profit Factor

## Customization

### Adding New Assets
Edit `config/config.yaml` and add symbols to the asset universe:

```yaml
assets:
  equity_indices:
    - "^GSPC"  # S&P 500
    - "^DJI"   # Dow Jones
    # Add your symbols here
```

### Modifying Strategy Parameters
Change strategy parameters in `config/config.yaml`:

```yaml
strategy:
  lookback_period: 12    # Change lookback period
  target_volatility: 0.40  # Change target volatility
  transaction_cost: 0.001  # Change transaction costs
```

### Custom Analysis
Create custom analysis by importing modules:

```python
from src.data.data_loader import DataLoader
from src.strategy.tsmom_strategy import TSMOMStrategy
from src.analysis.performance_analyzer import PerformanceAnalyzer

# Load data
loader = DataLoader()
prices = loader.load_processed_data()
returns = loader.calculate_returns(prices, 'D')

# Run strategy
strategy = TSMOMStrategy()
results = strategy.run_strategy(returns)

# Analyze performance
analyzer = PerformanceAnalyzer()
analysis = analyzer.generate_comprehensive_report(results['returns'])
```

## Troubleshooting

### Common Issues

1. **Data Download Errors**
   - Check internet connection
   - Verify symbol names in config
   - Try different data sources

2. **Memory Issues**
   - Reduce number of assets
   - Use shorter date range
   - Process data in chunks

3. **Performance Issues**
   - Use `--no-report` flag for faster execution
   - Reduce lookback period for testing
   - Use smaller asset universe

### Debug Mode
```bash
python src/main.py --log-level DEBUG
```

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_strategy.py

# Run with coverage
pytest --cov=src tests/
```

## Advanced Usage

### Sensitivity Analysis
```bash
python src/main.py --sensitivity
```

### Custom Configuration
```bash
python src/main.py --config my_config.yaml
```

### Batch Processing
```python
# Run multiple configurations
configs = ['config1.yaml', 'config2.yaml', 'config3.yaml']
for config in configs:
    results = run_full_backtest(config_path=config)
    # Process results
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details. 