# TSMOM Backtest Project

## 📈 Overview
This project implements and backtests a **Time Series Momentum (TSMOM)** strategy based on the seminal research by Moskowitz, Ooi, and Pedersen (2012). The strategy systematically opens long positions for assets with positive momentum and short positions for assets with negative momentum, using a 12-month lookback period for momentum calculation.

**Key Achievement**: The strategy demonstrates exceptional performance with **+4100% returns** on Russian markets (MOEX) and **+16400% returns** on US markets (Yahoo Finance) over the backtest period, significantly outperforming traditional benchmarks.

### 🎯 Strategy Performance Highlights
- **Russian Market (MOEX)**: 33.75% annual return, 1.55 Sharpe ratio, -27.50% max drawdown
- **US Market (Yahoo)**: 34.51% annual return, 1.72 Sharpe ratio, -41.34% max drawdown
- **Risk Management**: Target 40% volatility with inverse volatility position sizing
- **Diversification**: 20+ assets across multiple asset classes and geographies

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment recommended

### Installation
```bash
# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Backtest
```bash
# Quick start - Run both markets (US + Russian)
python run_backtest.py

# Run specific market only
python run_backtest.py --source yahoo    # US Market (S&P500 benchmark)
python run_backtest.py --source moex     # Russian Market (IMOEX benchmark)

# Advanced usage with main module
python src/main.py --data-source Yahoo   # or MOEX
python src/main.py --sensitivity         # Sensitivity analysis
python src/main.py --no-download         # Use existing data
```

## 📊 Strategy Details

### Core Strategy
- **Type**: Time Series Momentum (TSMOM)
- **Lookback Period**: 12 months
- **Holding Period**: 1 month
- **Target Volatility**: 40% annual
- **Rebalancing**: Monthly
- **Position Sizing**: Inverse volatility weighting

### Dual Market Coverage
The project supports **two complete market ecosystems**:

#### 🇺🇸 US/Global Markets (Yahoo Finance)
- **Equity Indices**: S&P 500, NASDAQ, Dow Jones, FTSE 100, DAX, CAC 40, Nikkei 225
- **Bonds**: 10Y/30Y Treasury, Treasury ETFs (TLT, IEF)
- **Currencies**: Major FX pairs (EUR/USD, GBP/USD, JPY/USD, etc.)
- **Commodities**: Gold, Crude Oil, Agricultural futures
- **Benchmark**: S&P 500 Index

#### 🇷🇺 Russian Markets (MOEX Data)
- **Futures Contracts**: 16+ Russian futures including:
  - **Currency**: Si (USD/RUB), Eu (EUR/RUB), CNY (CNY/RUB)
  - **Equity**: RTS, GAZR, LKOH, SBRF, ROSN, VTBR, GMKN
  - **Commodities**: GOLD, SILV, PLT, BZ (Brent), NG (Natural Gas)
  - **Mixed**: MIX futures
- **Index Data**: IMOEX benchmark data
- **Benchmark**: IMOEX (Moscow Exchange Index)

## 📁 Project Structure

```
Mop/
├── config/                 # Configuration files
│   └── config.yaml        # Strategy parameters & asset universe
├── data/                  # Data storage
│   ├── raw/              # Raw data sources
│   │   └── moex/         # MOEX futures CSV files (16+ contracts)
│   │       └── imoex/    # IMOEX benchmark data
│   └── processed/        # Cleaned & processed data
├── src/                   # Source code
│   ├── data/             # Data loading & processing
│   │   ├── base_loader.py        # Base data loader class
│   │   ├── data_loader.py        # Main data loader factory
│   │   ├── moex_loader.py        # MOEX-specific data loader
│   │   └── yahoo_loader.py       # Yahoo Finance loader
│   ├── strategy/         # Strategy implementation
│   │   └── tsmom_strategy.py     # Core TSMOM strategy
│   ├── analysis/         # Performance analysis
│   │   └── performance_analyzer.py  # Comprehensive analytics
│   ├── utils/            # Utility functions & comparisons
│   │   ├── compare_moex_vs_yahoo.py  # Market comparison
│   │   ├── compare_with_imoex.py     # IMOEX benchmark analysis
│   │   ├── compare_with_sp500.py     # S&P500 benchmark analysis
│   │   └── helpers.py              # General utilities
│   └── main.py           # Main execution orchestrator
├── reports/              # Generated performance reports
│   ├── moex/            # Russian market reports & plots
│   └── yahoo/           # US market reports & plots
├── tests/                # Comprehensive test suite
├── docs/                 # Documentation
├── notebooks/            # Jupyter analysis notebooks
├── requirements.txt      # Python dependencies
├── run_backtest.py       # Unified backtest runner
├── TSMOM_Backtest_Specification.markdown  # Technical specification
└── README.md            # This file
```

## 🔧 Key Features

### 🌍 Dual Market Support
- ✅ **Yahoo Finance Integration**: 20+ global assets (US, European, Asian markets)
- ✅ **MOEX Data Integration**: 16+ Russian futures contracts from local CSV files
- ✅ **Dual Benchmark System**: S&P 500 for global markets, IMOEX for Russian markets
- ✅ **Unified Interface**: Single command runs both market backtests
- ✅ **Market Comparison**: Side-by-side performance analysis

### 📊 Advanced Data Management
- ✅ **Multi-Source Loading**: Factory pattern for different data sources
- ✅ **Data Integrity Checks**: Comprehensive validation and cleaning
- ✅ **Historical Coverage**: 2000-2025 with different periods per contract
- ✅ **Asset Class Diversity**: Equities, bonds, currencies, commodities
- ✅ **Real-time Processing**: Daily and monthly return calculations

### 🎯 Sophisticated Strategy Engine
- ✅ **TSMOM Implementation**: 12-month momentum with 1-month holding
- ✅ **Risk-Adjusted Sizing**: Inverse volatility weighting with 40% target volatility
- ✅ **Transaction Cost Modeling**: Realistic trading cost assumptions
- ✅ **Position Management**: Long/short positioning based on momentum signals
- ✅ **Monthly Rebalancing**: Systematic portfolio rebalancing

### 📈 Comprehensive Analytics
- ✅ **Performance Metrics**: 20+ risk and return metrics
- ✅ **Risk Analysis**: VaR, CVaR, drawdown, and tail risk measures
- ✅ **Benchmark Comparison**: Alpha, beta, information ratio, tracking error
- ✅ **Rolling Analysis**: Time-varying performance characteristics
- ✅ **Trade-Level Analysis**: Detailed transaction logs and P&L tracking

### 🎨 Professional Reporting
- ✅ **Markdown Reports**: Comprehensive performance summaries
- ✅ **Visual Analytics**: Cumulative returns, drawdowns, distributions
- ✅ **Comparative Plots**: Strategy vs benchmark performance
- ✅ **Market-Specific Reports**: Separate reports for each market
- ✅ **Trade Logs**: CSV exports for detailed analysis

## 📈 Proven Performance Results

### 🏆 Backtest Performance Summary

| Market | Total Return | Annual Return | Sharpe Ratio | Max Drawdown | Assets Tested |
|--------|--------------|---------------|--------------|--------------|---------------|
| **🇷🇺 Russian (MOEX)** | **+4,109%** | **33.75%** | **1.55** | **-27.50%** | 16+ futures |
| **🇺🇸 US/Global (Yahoo)** | **+16,393%** | **34.51%** | **1.72** | **-41.34%** | 20+ assets |

### 📊 Comprehensive Metrics Suite

#### Return Analysis
- **Total & Annual Returns**: Absolute and annualized performance
- **Excess Returns**: Strategy outperformance vs benchmarks
- **Rolling Returns**: Time-varying performance characteristics

#### Risk Assessment
- **Volatility**: 20-22% realized volatility (target: 40%)
- **Maximum Drawdown**: Peak-to-trough decline analysis
- **VaR/CVaR**: 95% Value at Risk and Conditional VaR
- **Underwater Duration**: Time spent in drawdown periods

#### Risk-Adjusted Performance
- **Sharpe Ratio**: 1.55 (MOEX) | 1.72 (Yahoo) - Excellent risk-adjusted returns
- **Sortino Ratio**: 2.19 (MOEX) | 2.45 (Yahoo) - Superior downside risk management
- **Calmar Ratio**: 1.23 (MOEX) | 0.83 (Yahoo) - Strong drawdown-adjusted returns
- **Information Ratio**: 0.64 (MOEX) | 0.68 (Yahoo) - Consistent alpha generation

#### Statistical Properties
- **Win Rate**: ~55% - Consistent positive momentum capture
- **Profit Factor**: 1.34 - Profitable trades exceed losses
- **Skewness & Kurtosis**: Distribution analysis for tail risk
- **Beta**: Negative (-0.30) - Natural hedge against market downturns

## 🎯 Usage Examples

### 🚀 Basic Usage
```bash
# Run both markets with single command
python run_backtest.py

# Results:
# ✅ US Market: reports/yahoo/
# ✅ Russian Market: reports/moex/ 
# ✅ Comparative analysis printed to console
```

### 🎛️ Market-Specific Backtests
```bash
# US/Global markets only (Yahoo Finance + S&P500 benchmark)
python run_backtest.py --source yahoo

# Russian markets only (MOEX data + IMOEX benchmark)  
python run_backtest.py --source moex
```

### ⚙️ Advanced Configuration
```bash
# Use main module with data source selection
python src/main.py --data-source MOEX        # Russian markets
python src/main.py --data-source Yahoo       # US/Global markets

# Sensitivity analysis across lookback periods (3, 6, 12, 18 months)
python src/main.py --sensitivity --data-source Yahoo

# Skip data re-download (use existing processed data)
python src/main.py --no-download --data-source MOEX

# Skip report generation (strategy execution only)
python src/main.py --no-report --data-source Yahoo

# Debug mode with detailed logging
python src/main.py --log-level DEBUG
```

### 📊 Custom Analysis & Research
```bash
# Compare market performances
python src/utils/compare_moex_vs_yahoo.py

# IMOEX benchmark analysis
python src/utils/compare_with_imoex.py

# S&P500 benchmark analysis  
python src/utils/compare_with_sp500.py

# Check installation and data integrity
python src/utils/check_installation.py
```

### 🔬 Jupyter Notebook Analysis
```bash
# Start Jupyter environment
jupyter notebook

# Navigate to notebooks/ directory for interactive analysis
# Available notebooks for deep-dive research and visualization
```

## 📋 Technical Requirements

### System Requirements
- **Python**: 3.8+ (3.9+ recommended)
- **Operating System**: Cross-platform (Windows, macOS, Linux)
- **Memory**: 4GB+ RAM (8GB+ recommended for large datasets)
- **Storage**: 500MB+ for data and reports

### Core Dependencies
```txt
# Data Science Stack
pandas>=1.5.0          # Data manipulation and analysis
numpy>=1.21.0           # Numerical computing
matplotlib>=3.5.0       # Plotting and visualization  
seaborn>=0.11.0         # Statistical data visualization

# Data Sources
yfinance>=0.2.0         # Yahoo Finance API
quandl>=3.7.0          # Alternative data source

# Statistical Analysis
scipy>=1.9.0            # Scientific computing
statsmodels>=0.13.0     # Statistical models

# Financial Analysis
pandas-ta>=0.3.14b0     # Technical analysis indicators
pyfolio-reloaded>=0.9.2 # Portfolio performance analysis

# Configuration & Utilities
pyyaml>=6.0             # YAML configuration files
python-dotenv>=0.19.0   # Environment variable management

# Development & Testing
pytest>=7.0.0           # Testing framework
pytest-cov>=4.0.0       # Coverage reporting
black>=22.0.0           # Code formatting
flake8>=5.0.0          # Code linting

# Interactive Analysis
jupyter>=1.0.0          # Jupyter notebooks
ipykernel>=6.0.0        # Jupyter kernel
```

## 🧪 Testing & Validation

### Test Suite
```bash
# Run complete test suite
pytest tests/

# Run with coverage reporting
pytest --cov=src tests/

# Run specific test modules
pytest tests/test_strategy.py          # Strategy implementation tests
pytest tests/test_moex_loader.py       # MOEX data loader tests  
pytest tests/test_data_integrity.py    # Data quality tests
```

### Data Integrity Checks
```bash
# Validate data quality and completeness
python src/utils/check_installation.py

# Verify MOEX data files
python tests/test_data_integrity.py
```

### Performance Validation
```bash
# Compare backtests across markets
python src/utils/compare_moex_vs_yahoo.py

# Validate against benchmarks
python src/utils/compare_with_imoex.py     # Russian market
python src/utils/compare_with_sp500.py     # US market
```

## 📚 Documentation & Resources

### 📖 Core Documentation
- **[Usage Guide](docs/USAGE.md)** - Comprehensive setup and usage instructions
- **[Technical Specification](TSMOM_Backtest_Specification.markdown)** - Detailed strategy requirements (Russian)
- **[Configuration Reference](config/config.yaml)** - Complete parameter documentation
- **Log Files**: `tsmom_backtest.log` - Detailed execution logs

### 📊 Generated Reports
- **Performance Reports**: `reports/{yahoo|moex}/performance_report.md`
- **Metrics Tables**: `reports/{yahoo|moex}/tables/performance_metrics.csv`
- **Visualizations**: `reports/{yahoo|moex}/plots/` (PNG format)
- **Trade Logs**: `reports/{yahoo|moex}/trade_history.csv`

### 🔬 Research Notebooks
- **Interactive Analysis**: `notebooks/` directory
- **Market Comparisons**: Side-by-side strategy analysis
- **Custom Visualization**: Create your own charts and analysis

## 🔬 Research Background & Strategy Foundation

### 📄 Academic Foundation
- **Primary Research**: ["Time Series Momentum"](https://doi.org/10.1016/j.jfineco.2011.11.003) by Moskowitz, Ooi, and Pedersen (2012)
- **Journal**: Journal of Financial Economics, 104(2), 228-250
- **Key Finding**: Momentum patterns exist across asset classes and time series
- **Innovation**: Extension beyond traditional cross-sectional momentum

### 🧠 Strategy Philosophy
- **Core Insight**: Assets with positive (negative) performance over past 12 months tend to continue performing well (poorly)
- **Time Series Focus**: Unlike cross-sectional momentum, TSMOM examines each asset's own historical performance
- **Diversification Benefit**: Works across asset classes - equities, bonds, currencies, commodities
- **Risk Management**: Systematic volatility targeting prevents concentration in high-volatility periods

### 🌍 Implementation Innovation
This project extends the original research by:
- **Dual Market Testing**: Simultaneous US/global and Russian market implementation
- **Local Data Integration**: Direct MOEX futures data instead of proxy instruments  
- **Modern Infrastructure**: Python-based implementation with comprehensive testing
- **Extended Universe**: 16+ Russian futures contracts + 20+ global assets
- **Benchmark Alignment**: Market-appropriate benchmarks (IMOEX vs S&P500)

## 📂 Output Structure & Generated Files

### 📈 Market-Specific Reports
```
reports/
├── yahoo/                          # US/Global market results
│   ├── performance_report.md       # Comprehensive analysis report
│   ├── trade_history.csv          # Detailed trade log with P&L
│   ├── tables/
│   │   └── performance_metrics.csv # Metrics summary table
│   └── plots/
│       ├── cumulative_returns.png  # Strategy vs S&P500 performance
│       ├── drawdown.png           # Drawdown analysis over time
│       ├── returns_distribution.png # Return distribution histogram
│       └── rolling_metrics.png     # Rolling Sharpe ratio and metrics
└── moex/                           # Russian market results
    ├── performance_report.md       # Comprehensive analysis report  
    ├── trade_history.csv          # Detailed trade log with P&L
    ├── tables/
    │   └── performance_metrics.csv # Metrics summary table
    └── plots/
        ├── cumulative_returns.png  # Strategy vs IMOEX performance
        ├── drawdown.png           # Drawdown analysis over time
        ├── returns_distribution.png # Return distribution histogram
        └── rolling_metrics.png     # Rolling Sharpe ratio and metrics
```

### 💾 Data Files
```
data/
├── raw/
│   └── moex/                       # Original MOEX CSV files
│       ├── Si_*.csv               # USD/RUB futures contracts
│       ├── GOLD_*.csv             # Gold futures contracts
│       ├── RTS_*.csv              # RTS index futures
│       └── [16+ other contracts]   # Complete futures universe
└── processed/                      # Cleaned and processed data
    ├── yahoo_cleaned_prices.csv   # Yahoo Finance processed data
    └── moex_cleaned_prices.csv     # MOEX processed data
```

### 📋 Execution Logs
- `tsmom_backtest.log` - Detailed execution log with timestamps
- Console output with real-time progress and summary statistics

## 🚀 Getting Started Checklist

### ✅ Quick Setup (5 minutes)
1. **Clone & Setup**:
   ```bash
   git clone <repository-url> && cd Mop
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run First Backtest**:
   ```bash
   python run_backtest.py --source yahoo  # US market (~2 min)
   ```

3. **View Results**:
   - Check `reports/yahoo/performance_report.md`
   - Review plots in `reports/yahoo/plots/`

### 🎯 Next Steps
- **Run Russian market**: `python run_backtest.py --source moex`
- **Both markets**: `python run_backtest.py` 
- **Explore**: Check `docs/USAGE.md` for advanced features

## 🤝 Contributing & Development

### Contributing Guidelines
1. **Fork** the repository
2. **Create** feature branch: `git checkout -b feature/amazing-feature`
3. **Add tests** for new functionality
4. **Ensure** all tests pass: `pytest tests/`
5. **Follow** code style: `black src/` and `flake8 src/`
6. **Submit** pull request with clear description

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests with coverage
pytest --cov=src tests/

# Format code
black src/ tests/

# Lint code  
flake8 src/ tests/
```

## ⚠️ Important Disclaimers

### 🔬 Research & Educational Purpose
- **Academic Research**: Implementation for educational and research purposes
- **Historical Analysis**: Based on historical data - past performance ≠ future results
- **No Investment Advice**: This tool does not constitute financial or investment advice
- **Risk Warning**: Use at your own risk for any actual trading decisions

### 📊 Data & Methodology Notes
- **Data Sources**: Yahoo Finance (global) + MOEX CSV files (Russian markets)
- **Backtesting Limitations**: Does not account for all real-world trading constraints
- **Market Impact**: Assumes infinite liquidity and no market impact
- **Transaction Costs**: Simplified transaction cost model (0.1% per trade)

## 📄 License & Acknowledgments

### License
**MIT License** - See LICENSE file for complete terms

### Academic Citation
```bibtex
@article{moskowitz2012time,
  title={Time series momentum},
  author={Moskowitz, Tobias J and Ooi, Yao Hua and Pedersen, Lasse Heje},
  journal={Journal of Financial Economics},
  volume={104},
  number={2},
  pages={228--250},
  year={2012},
  publisher={Elsevier}
}
```

### Acknowledgments
- **Original Research**: Moskowitz, Ooi, and Pedersen (2012) for foundational TSMOM research
- **Data Providers**: Yahoo Finance and MOEX for market data access
- **Open Source**: Python scientific computing ecosystem
- **Community**: Contributors and users providing feedback and improvements

---
*Last Updated: August 2025 | Version: 2.0 | Status: Production Ready* 
