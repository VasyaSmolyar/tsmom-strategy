#!/usr/bin/env python3
"""
MOEX Backtest Script for TSMOM Strategy.
Runs backtest using MOEX data and compares performance with IMOEX index.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from main import run_full_backtest, setup_logging
import logging
import yaml

def main():
    """Run the TSMOM backtest with MOEX data and IMOEX benchmark."""
    print("TSMOM Backtest with MOEX Data")
    print("=" * 50)
    
    # Setup logging
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config_path = "config/config.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Verify we're using MOEX data source
    data_source = config['data'].get('source', 'Yahoo')
    benchmark = config['backtest'].get('benchmark', '^GSPC')
    
    if data_source != 'MOEX':
        print(f"WARNING: Data source is set to '{data_source}', not 'MOEX'")
        print("Please update config/config.yaml to use MOEX data source")
        return
    
    if benchmark != 'IMOEX':
        print(f"WARNING: Benchmark is set to '{benchmark}', not 'IMOEX'")
        print("Please update config/config.yaml to use IMOEX benchmark")
        return
    
    print(f"Data Source: {data_source}")
    print(f"Benchmark: {benchmark}")
    print("=" * 50)
    
    try:
        # Run the complete backtest
        results = run_full_backtest(
            config_path=config_path,
            download_data=True,
            generate_report=True
        )
        
        print("\n" + "=" * 50)
        print("MOEX BACKTEST COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        # Print summary
        if results.get('analysis_results') and results['analysis_results'].get('metrics'):
            metrics = results['analysis_results']['metrics']
            print(f"\nKey Results:")
            print(f"Total Return: {metrics.get('total_return', 0):.2%}")
            print(f"Annual Return: {metrics.get('annual_return', 0):.2%}")
            print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            
            if 'benchmark_annual_return' in metrics:
                print(f"\nBenchmark Comparison (IMOEX):")
                print(f"Benchmark Return: {metrics.get('benchmark_annual_return', 0):.2%}")
                print(f"Excess Return: {metrics.get('excess_return', 0):.2%}")
                print(f"Information Ratio: {metrics.get('information_ratio', 0):.2f}")
                print(f"Beta: {metrics.get('beta', 0):.4f}")
                print(f"Alpha: {metrics.get('alpha', 0):.2%}")
                
                # Print strategy start date info
                print(f"\nNote: Comparison period starts from strategy application date")
        
        print(f"\nReports saved to: reports/")
        print(f"Log file: tsmom_backtest.log")
        
        # Print data information
        if results.get('prices') is not None:
            prices = results['prices']
            print(f"\nData Summary:")
            print(f"Assets: {len(prices.columns)}")
            print(f"Periods: {len(prices)}")
            print(f"Date Range: {prices.index.min()} to {prices.index.max()}")
            print(f"Assets: {list(prices.columns)}")
        
    except Exception as e:
        logger.error(f"Error during MOEX backtest: {e}")
        print(f"\nERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
