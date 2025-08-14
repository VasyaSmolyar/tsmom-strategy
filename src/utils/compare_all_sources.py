#!/usr/bin/env python3
"""
Comparison script for TSMOM backtest results using MOEX vs YahooFutures vs YahooCrypto data sources.
"""

import sys
from pathlib import Path
import yaml
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent))

from main import run_full_backtest, setup_logging

def run_backtest_with_source(data_source: str, benchmark: str) -> dict:
    """
    Run backtest with specified data source and benchmark.
    
    Args:
        data_source: "MOEX", "YahooFutures" or "YahooCrypto"
        benchmark: Benchmark symbol
    
    Returns:
        Dictionary with backtest results
    """
    # Load current config
    config_path = "../../config/config.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Update config for this run
    config['data']['source'] = data_source
    config['backtest']['benchmark'] = benchmark
    
    # Save temporary config
    temp_config_path = f"../../config/config_{data_source.lower()}.yaml"
    with open(temp_config_path, 'w') as file:
        yaml.dump(config, file)
    
    try:
        # Run backtest
        results = run_full_backtest(
            config_path=temp_config_path,
            download_data=True,
            generate_report=True  # We need the analysis results
        )
        return results
    finally:
        # Clean up temp config
        Path(temp_config_path).unlink(missing_ok=True)

def main():
    """Compare MOEX vs YahooFutures vs YahooCrypto backtest results."""
    print("TSMOM Strategy Comparison: MOEX vs YahooFutures vs YahooCrypto Data")
    print("=" * 70)
    
    # Setup logging
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    try:
        # Run MOEX backtest
        print("\n1. Running MOEX backtest...")
        moex_results = run_backtest_with_source("MOEX", "IMOEX")
        
        # Run Yahoo Futures backtest
        print("\n2. Running YahooFutures backtest...")
        yahoo_futures_results = run_backtest_with_source("YahooFutures", "^GSPC")
        
        # Run Yahoo Crypto backtest
        print("\n3. Running YahooCrypto backtest...")
        yahoo_crypto_results = run_backtest_with_source("YahooCrypto", "BTC-USD")
        
        # Extract key metrics
        moex_metrics = moex_results.get('analysis_results', {}).get('metrics', {}) if moex_results.get('analysis_results') else {}
        yahoo_futures_metrics = yahoo_futures_results.get('analysis_results', {}).get('metrics', {}) if yahoo_futures_results.get('analysis_results') else {}
        yahoo_crypto_metrics = yahoo_crypto_results.get('analysis_results', {}).get('metrics', {}) if yahoo_crypto_results.get('analysis_results') else {}
        
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        
        # Print comparison table
        print(f"\n{'Metric':<25} {'MOEX (IMOEX)':<15} {'Futures (S&P500)':<15} {'Crypto (BTC)':<15}")
        print("-" * 75)
        
        metrics_to_compare = [
            ('Annual Return', 'annual_return', '{:.2%}'),
            ('Sharpe Ratio', 'sharpe_ratio', '{:.2f}'),
            ('Max Drawdown', 'max_drawdown', '{:.2%}'),
            ('Volatility', 'volatility', '{:.2%}'),
            ('Total Return', 'total_return', '{:.2%}'),
            ('Benchmark Return', 'benchmark_annual_return', '{:.2%}'),
            ('Excess Return', 'excess_return', '{:.2%}'),
            ('Information Ratio', 'information_ratio', '{:.2f}'),
            ('Beta', 'beta', '{:.4f}'),
            ('Alpha', 'alpha', '{:.2%}')
        ]
        
        for metric_name, metric_key, format_str in metrics_to_compare:
            moex_val = moex_metrics.get(metric_key, 0)
            futures_val = yahoo_futures_metrics.get(metric_key, 0)
            crypto_val = yahoo_crypto_metrics.get(metric_key, 0)
            
            print(f"{metric_name:<25} {format_str.format(moex_val):<15} {format_str.format(futures_val):<15} {format_str.format(crypto_val):<15}")
        
        # Print data summary
        print(f"\n{'Data Summary':<25} {'MOEX':<15} {'Futures':<15} {'Crypto':<15}")
        print("-" * 75)
        
        moex_prices = moex_results.get('prices')
        futures_prices = yahoo_futures_results.get('prices')
        crypto_prices = yahoo_crypto_results.get('prices')
        
        if moex_prices is not None:
            moex_assets = len(moex_prices.columns)
            moex_periods = len(moex_prices)
            moex_start = moex_prices.index.min().strftime('%Y-%m-%d')
            moex_end = moex_prices.index.max().strftime('%Y-%m-%d')
        else:
            moex_assets = moex_periods = moex_start = moex_end = "N/A"
            
        if futures_prices is not None:
            futures_assets = len(futures_prices.columns)
            futures_periods = len(futures_prices)
            futures_start = futures_prices.index.min().strftime('%Y-%m-%d')
            futures_end = futures_prices.index.max().strftime('%Y-%m-%d')
        else:
            futures_assets = futures_periods = futures_start = futures_end = "N/A"
            
        if crypto_prices is not None:
            crypto_assets = len(crypto_prices.columns)
            crypto_periods = len(crypto_prices)
            crypto_start = crypto_prices.index.min().strftime('%Y-%m-%d')
            crypto_end = crypto_prices.index.max().strftime('%Y-%m-%d')
        else:
            crypto_assets = crypto_periods = crypto_start = crypto_end = "N/A"
        
        print(f"{'Assets':<25} {moex_assets:<15} {futures_assets:<15} {crypto_assets:<15}")
        print(f"{'Periods':<25} {moex_periods:<15} {futures_periods:<15} {crypto_periods:<15}")
        print(f"{'Date Range Start':<25} {moex_start:<15} {futures_start:<15} {crypto_start:<15}")
        print(f"{'Date Range End':<25} {moex_end:<15} {futures_end:<15} {crypto_end:<15}")
        
        # Print conclusions
        print(f"\n{'Conclusions':<75}")
        print("=" * 75)
        
        moex_sharpe = moex_metrics.get('sharpe_ratio', 0)
        futures_sharpe = yahoo_futures_metrics.get('sharpe_ratio', 0)
        crypto_sharpe = yahoo_crypto_metrics.get('sharpe_ratio', 0)
        
        # Find best performing strategy by Sharpe ratio
        sharpe_values = {'MOEX': moex_sharpe, 'Futures': futures_sharpe, 'Crypto': crypto_sharpe}
        best_sharpe = max(sharpe_values, key=sharpe_values.get)
        print(f"✓ Best risk-adjusted returns: {best_sharpe} (Sharpe: {sharpe_values[best_sharpe]:.2f})")
        
        # Find lowest drawdown
        moex_dd = abs(moex_metrics.get('max_drawdown', 0))
        futures_dd = abs(yahoo_futures_metrics.get('max_drawdown', 0))
        crypto_dd = abs(yahoo_crypto_metrics.get('max_drawdown', 0))
        
        dd_values = {'MOEX': moex_dd, 'Futures': futures_dd, 'Crypto': crypto_dd}
        best_dd = min(dd_values, key=dd_values.get)
        print(f"✓ Lowest maximum drawdown: {best_dd} ({dd_values[best_dd]:.2%})")
        
        # Find best excess returns
        moex_excess = moex_metrics.get('excess_return', 0)
        futures_excess = yahoo_futures_metrics.get('excess_return', 0)
        crypto_excess = yahoo_crypto_metrics.get('excess_return', 0)
        
        excess_values = {'MOEX': moex_excess, 'Futures': futures_excess, 'Crypto': crypto_excess}
        best_excess = max(excess_values, key=excess_values.get)
        print(f"✓ Best excess returns: {best_excess} ({excess_values[best_excess]:.2%})")
        
        print(f"\nReports saved to: reports/")
        print(f"Log file: tsmom_backtest.log")
        
    except Exception as e:
        logger.error(f"Error during comparison: {e}")
        print(f"\nERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
