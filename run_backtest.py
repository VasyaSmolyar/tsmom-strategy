#!/usr/bin/env python3
"""
Quick start script for TSMOM backtest.
Run this script to execute the complete backtest pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from main import run_full_backtest, setup_logging
import logging

def main():
    """Run the complete TSMOM backtest."""
    print("TSMOM Backtest - Quick Start")
    print("=" * 50)
    
    # Setup logging
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    try:
        # Run the complete backtest
        results = run_full_backtest(
            config_path="config/config.yaml",
            download_data=True,
            generate_report=True
        )
        
        print("\n" + "=" * 50)
        print("BACKTEST COMPLETED SUCCESSFULLY!")
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
                print(f"Benchmark Return: {metrics.get('benchmark_annual_return', 0):.2%}")
                print(f"Excess Return: {metrics.get('excess_return', 0):.2%}")
        
        print(f"\nReports saved to: reports/")
        print(f"Log file: tsmom_backtest.log")
        
    except Exception as e:
        logger.error(f"Error during backtest: {e}")
        print(f"\nERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 