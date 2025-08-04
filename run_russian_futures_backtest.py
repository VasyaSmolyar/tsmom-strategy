#!/usr/bin/env python3
"""
Russian Futures TSMOM Backtest Runner
Runs TSMOM strategy on Russian futures using Tinkoff API data.
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.tinvest_loader import TInvestLoader
from strategy.tsmom_strategy import TSMOMStrategy
from analysis.performance_analyzer import PerformanceAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Run TSMOM backtest on Russian futures."""
    
    # Check if TINKOFF_TOKEN is set
    if not os.getenv('TINKOFF_TOKEN'):
        print("ERROR: TINKOFF_TOKEN environment variable is not set")
        print("Please set your Tinkoff API token:")
        print("export TINKOFF_TOKEN='your_token_here'")
        return
    
    try:
        # Load configuration
        config_path = "config/config_tinkoff.yaml"
        
        # Initialize data loader
        print("Initializing Tinkoff data loader...")
        loader = TInvestLoader(config_path)
        
        # Download data
        print("Downloading Russian futures data...")
        prices = loader.download_data()
        
        if prices.empty:
            print("No data downloaded. Exiting.")
            return
        
        print(f"Downloaded data shape: {prices.shape}")
        print(f"Available assets: {list(prices.columns)}")
        
        # Calculate returns
        print("Calculating returns...")
        returns = loader.calculate_returns(prices, 'D')
        
        # Initialize strategy
        print("Initializing TSMOM strategy...")
        strategy = TSMOMStrategy(config_path)
        
        # Run backtest
        print("Running backtest...")
        portfolio_returns = strategy.run_backtest(returns)
        
        # Analyze performance
        print("Analyzing performance...")
        analyzer = PerformanceAnalyzer()
        results = analyzer.analyze_performance(portfolio_returns, returns)
        
        # Save results
        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)
        
        # Save portfolio returns
        portfolio_file = output_dir / "russian_futures_portfolio_returns.csv"
        portfolio_returns.to_csv(portfolio_file)
        print(f"Portfolio returns saved to {portfolio_file}")
        
        # Generate report
        report_file = output_dir / "russian_futures_performance_report.md"
        with open(report_file, 'w') as f:
            f.write("# Russian Futures TSMOM Strategy Performance Report\n\n")
            f.write(f"## Data Summary\n")
            f.write(f"- Period: {prices.index.min().date()} to {prices.index.max().date()}\n")
            f.write(f"- Assets: {len(prices.columns)}\n")
            f.write(f"- Total observations: {len(prices)}\n\n")
            
            f.write(f"## Performance Metrics\n")
            f.write(f"- Total Return: {results['total_return']:.2%}\n")
            f.write(f"- Annualized Return: {results['annualized_return']:.2%}\n")
            f.write(f"- Volatility: {results['volatility']:.2%}\n")
            f.write(f"- Sharpe Ratio: {results['sharpe_ratio']:.2f}\n")
            f.write(f"- Max Drawdown: {results['max_drawdown']:.2%}\n\n")
            
            f.write(f"## Asset Universe\n")
            for asset in prices.columns:
                asset_return = returns[asset].mean() * 252
                asset_vol = returns[asset].std() * (252 ** 0.5)
                f.write(f"- {asset}: {asset_return:.2%} return, {asset_vol:.2%} volatility\n")
        
        print(f"Performance report saved to {report_file}")
        
        # Print summary
        print("\n" + "="*50)
        print("RUSSIAN FUTURES TSMOM BACKTEST RESULTS")
        print("="*50)
        print(f"Period: {prices.index.min().date()} to {prices.index.max().date()}")
        print(f"Assets: {len(prices.columns)}")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Annualized Return: {results['annualized_return']:.2%}")
        print(f"Volatility: {results['volatility']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print("="*50)
        
    except Exception as e:
        print(f"Error during backtest: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 