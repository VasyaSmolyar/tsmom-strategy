#!/usr/bin/env python3
"""
Comparison script for TSMOM backtest results using MOEX vs Yahoo data sources.
"""

import sys
from pathlib import Path
import yaml
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from main import run_full_backtest, setup_logging

def run_backtest_with_source(data_source: str, benchmark: str) -> dict:
    """
    Run backtest with specified data source and benchmark.
    
    Args:
        data_source: "MOEX" or "Yahoo"
        benchmark: Benchmark symbol
    
    Returns:
        Dictionary with backtest results
    """
    # Load current config
    config_path = "config/config.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Update config for this run
    config['data']['source'] = data_source
    config['backtest']['benchmark'] = benchmark
    
    # Save temporary config
    temp_config_path = f"config/config_{data_source.lower()}.yaml"
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
    """Compare MOEX vs Yahoo backtest results."""
    print("TSMOM Strategy Comparison: MOEX vs Yahoo Data")
    print("=" * 60)
    
    # Setup logging
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    try:
        # Run MOEX backtest
        print("\n1. Running MOEX backtest...")
        moex_results = run_backtest_with_source("MOEX", "IMOEX")
        
        # Run Yahoo backtest
        print("\n2. Running Yahoo backtest...")
        yahoo_results = run_backtest_with_source("Yahoo", "^GSPC")
        
        # Extract key metrics
        moex_metrics = moex_results.get('analysis_results', {}).get('metrics', {}) if moex_results.get('analysis_results') else {}
        yahoo_metrics = yahoo_results.get('analysis_results', {}).get('metrics', {}) if yahoo_results.get('analysis_results') else {}
        
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        
        # Print comparison table
        print(f"\n{'Metric':<25} {'MOEX (IMOEX)':<15} {'Yahoo (S&P500)':<15} {'Difference':<15}")
        print("-" * 70)
        
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
            yahoo_val = yahoo_metrics.get(metric_key, 0)
            diff = moex_val - yahoo_val
            
            print(f"{metric_name:<25} {format_str.format(moex_val):<15} {format_str.format(yahoo_val):<15} {format_str.format(diff):<15}")
        
        # Print data summary
        print(f"\n{'Data Summary':<25} {'MOEX':<15} {'Yahoo':<15}")
        print("-" * 55)
        
        moex_prices = moex_results.get('prices')
        yahoo_prices = yahoo_results.get('prices')
        
        if moex_prices is not None and yahoo_prices is not None:
            print(f"{'Assets':<25} {len(moex_prices.columns):<15} {len(yahoo_prices.columns):<15}")
            print(f"{'Periods':<25} {len(moex_prices):<15} {len(yahoo_prices):<15}")
            print(f"{'Date Range Start':<25} {moex_prices.index.min().strftime('%Y-%m-%d'):<15} {yahoo_prices.index.min().strftime('%Y-%m-%d'):<15}")
            print(f"{'Date Range End':<25} {moex_prices.index.max().strftime('%Y-%m-%d'):<15} {yahoo_prices.index.max().strftime('%Y-%m-%d'):<15}")
        
        # Print conclusions
        print(f"\n{'Conclusions':<60}")
        print("=" * 60)
        
        moex_sharpe = moex_metrics.get('sharpe_ratio', 0)
        yahoo_sharpe = yahoo_metrics.get('sharpe_ratio', 0)
        
        if moex_sharpe > yahoo_sharpe:
            print(f"✓ MOEX strategy shows better risk-adjusted returns (Sharpe: {moex_sharpe:.2f} vs {yahoo_sharpe:.2f})")
        else:
            print(f"✗ Yahoo strategy shows better risk-adjusted returns (Sharpe: {yahoo_sharpe:.2f} vs {moex_sharpe:.2f})")
        
        moex_dd = moex_metrics.get('max_drawdown', 0)
        yahoo_dd = yahoo_metrics.get('max_drawdown', 0)
        
        if abs(moex_dd) < abs(yahoo_dd):
            print(f"✓ MOEX strategy shows lower maximum drawdown ({moex_dd:.2%} vs {yahoo_dd:.2%})")
        else:
            print(f"✗ Yahoo strategy shows lower maximum drawdown ({yahoo_dd:.2%} vs {moex_dd:.2%})")
        
        moex_excess = moex_metrics.get('excess_return', 0)
        yahoo_excess = yahoo_metrics.get('excess_return', 0)
        
        if moex_excess > yahoo_excess:
            print(f"✓ MOEX strategy shows better excess returns ({moex_excess:.2%} vs {yahoo_excess:.2%})")
        else:
            print(f"✗ Yahoo strategy shows better excess returns ({yahoo_excess:.2%} vs {moex_excess:.2%})")
        
        print(f"\nReports saved to: reports/")
        print(f"Log file: tsmom_backtest.log")
        
    except Exception as e:
        logger.error(f"Error during comparison: {e}")
        print(f"\nERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
