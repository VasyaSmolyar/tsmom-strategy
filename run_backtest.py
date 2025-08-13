#!/usr/bin/env python3
"""
Unified Backtest Script for TSMOM Strategy.

Runs backtests for Yahoo (US) and/or MOEX (Russian) markets based on command line arguments.

Usage:
    python run_backtest.py                    # Run both Yahoo and MOEX backtests
    python run_backtest.py --source yahoo     # Run only Yahoo backtest
    python run_backtest.py --source moex      # Run only MOEX backtest
"""

import sys
import argparse
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from main import run_full_backtest, setup_logging
import logging


def run_market_backtest(data_source: str, config_path: str = "config/config.yaml") -> dict:
    """
    Run backtest for a specific market.
    
    Args:
        data_source: "Yahoo" or "MOEX"
        config_path: Path to configuration file
    
    Returns:
        Dictionary with backtest results
    """
    logger = logging.getLogger(__name__)
    
    market_name = "US" if data_source == "Yahoo" else "Russian"
    benchmark_name = "S&P500" if data_source == "Yahoo" else "IMOEX"
    
    print(f"\n{'=' * 60}")
    print(f"Running {market_name} Market Backtest")
    print(f"Data Source: {data_source}")
    print(f"Benchmark: {benchmark_name}")
    print(f"{'=' * 60}")
    
    try:
        start_time = time.time()
        
        # Run the backtest with the specified data source
        results = run_full_backtest(
            config_path=config_path,
            download_data=True,
            generate_report=True,
            data_source=data_source
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"\n{market_name} Market Backtest Completed Successfully!")
        print(f"Execution Time: {execution_time:.1f} seconds")
        
        # Print summary
        if results.get('analysis_results') and results['analysis_results'].get('metrics'):
            metrics = results['analysis_results']['metrics']
            print(f"\nKey Results ({market_name} Market):")
            print(f"Total Return: {metrics.get('total_return', 0):.2%}")
            print(f"Annual Return: {metrics.get('annual_return', 0):.2%}")
            print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            
            if 'benchmark_annual_return' in metrics:
                print(f"\nBenchmark Comparison ({benchmark_name}):")
                print(f"Benchmark Return: {metrics.get('benchmark_annual_return', 0):.2%}")
                print(f"Excess Return: {metrics.get('excess_return', 0):.2%}")
                print(f"Information Ratio: {metrics.get('information_ratio', 0):.2f}")
                print(f"Beta: {metrics.get('beta', 0):.4f}")
                print(f"Alpha: {metrics.get('alpha', 0):.2%}")
        
        # Print data information
        if results.get('prices') is not None:
            prices = results['prices']
            print(f"\nData Summary ({market_name} Market):")
            print(f"Assets: {len(prices.columns)}")
            print(f"Periods: {len(prices)}")
            print(f"Date Range: {prices.index.min()} to {prices.index.max()}")
            print(f"Assets: {list(prices.columns)}")
        
        # Print trade summary
        if results.get('trade_log') is not None:
            trade_log = results['trade_log']
            print(f"\nTrade Summary ({market_name} Market):")
            print(f"Total Trades: {len(trade_log)}")
            if len(trade_log) > 0:
                profitable_trades = len(trade_log[trade_log['pnl'] > 0])
                print(f"Profitable Trades: {profitable_trades} ({profitable_trades/len(trade_log):.1%})")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during {market_name} market backtest: {e}")
        print(f"\nERROR in {market_name} market backtest: {e}")
        return None


def run_both_backtests(config_path: str = "config/config.yaml") -> dict:
    """Run backtests for both markets."""
    logger = logging.getLogger(__name__)
    overall_start_time = time.time()
    all_results = {}
    
    print("Running backtests for both markets:")
    print("1. US Market (Yahoo Finance data, S&P500 benchmark)")
    print("2. Russian Market (MOEX data, IMOEX benchmark)")
    print("Results will be saved to reports/yahoo/ and reports/moex/")
    
    # Run Yahoo (US) backtest
    logger.info("Starting US Market (Yahoo) backtest...")
    yahoo_results = run_market_backtest("Yahoo", config_path)
    if yahoo_results:
        all_results['yahoo'] = yahoo_results
        logger.info("US Market backtest completed successfully")
    else:
        logger.error("US Market backtest failed")
    
    print(f"\n{'=' * 60}")
    print("Waiting 2 seconds before starting Russian market backtest...")
    print(f"{'=' * 60}")
    time.sleep(2)
    
    # Run MOEX (Russian) backtest  
    logger.info("Starting Russian Market (MOEX) backtest...")
    moex_results = run_market_backtest("MOEX", config_path)
    if moex_results:
        all_results['moex'] = moex_results
        logger.info("Russian Market backtest completed successfully")
    else:
        logger.error("Russian Market backtest failed")
    
    overall_end_time = time.time()
    total_execution_time = overall_end_time - overall_start_time
    
    # Final summary
    print(f"\n{'=' * 60}")
    print("UNIFIED BACKTEST COMPLETED!")
    print(f"{'=' * 60}")
    print(f"Total Execution Time: {total_execution_time:.1f} seconds")
    print(f"Successful Backtests: {len(all_results)}/2")
    
    if len(all_results) > 0:
        print(f"\nResults saved to:")
        for market in all_results.keys():
            print(f"  - reports/{market}/")
        
        # Comparative summary
        if len(all_results) == 2:
            print(f"\nComparative Summary:")
            print(f"{'Metric':<20} {'US Market':<15} {'Russian Market':<15}")
            print(f"{'-' * 50}")
            
            us_annual = us_sharpe = us_drawdown = "N/A"
            ru_annual = ru_sharpe = ru_drawdown = "N/A"
            
            for market, results in all_results.items():
                if results.get('analysis_results') and results['analysis_results'].get('metrics'):
                    metrics = results['analysis_results']['metrics']
                    
                    if market == 'yahoo':
                        us_annual = f"{metrics.get('annual_return', 0):.2%}"
                        us_sharpe = f"{metrics.get('sharpe_ratio', 0):.2f}"
                        us_drawdown = f"{metrics.get('max_drawdown', 0):.2%}"
                    else:
                        ru_annual = f"{metrics.get('annual_return', 0):.2%}"
                        ru_sharpe = f"{metrics.get('sharpe_ratio', 0):.2f}"
                        ru_drawdown = f"{metrics.get('max_drawdown', 0):.2%}"
            
            print(f"{'Annual Return':<20} {us_annual:<15} {ru_annual:<15}")
            print(f"{'Sharpe Ratio':<20} {us_sharpe:<15} {ru_sharpe:<15}")
            print(f"{'Max Drawdown':<20} {us_drawdown:<15} {ru_drawdown:<15}")
    
    return all_results


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="TSMOM Strategy Backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_backtest.py                    # Run both Yahoo and MOEX backtests
  python run_backtest.py --source yahoo     # Run only Yahoo backtest
  python run_backtest.py --source moex      # Run only MOEX backtest
        """
    )
    
    parser.add_argument(
        '--source', 
        choices=['yahoo', 'moex'], 
        help='Data source for backtest (yahoo or moex). If not specified, runs both.'
    )
    
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    args = parser.parse_args()
    
    print("TSMOM Strategy Backtest")
    print("=" * 60)
    
    # Setup logging
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    try:
        if args.source:
            # Run single backtest
            source_map = {'yahoo': 'Yahoo', 'moex': 'MOEX'}
            data_source = source_map[args.source.lower()]
            
            print(f"Running {args.source.upper()} backtest only")
            print("=" * 60)
            
            results = run_market_backtest(data_source, args.config)
            
            if results:
                print(f"\n{args.source.upper()} backtest completed successfully!")
                print(f"Results saved to: reports/{args.source}/")
            else:
                print(f"\n{args.source.upper()} backtest failed!")
                sys.exit(1)
        else:
            # Run both backtests
            print("No source specified - running both Yahoo and MOEX backtests")
            print("=" * 60)
            
            all_results = run_both_backtests(args.config)
            
            if len(all_results) == 0:
                logger.error("All backtests failed!")
                sys.exit(1)
            elif len(all_results) == 1:
                logger.warning("Only one backtest completed successfully")
            else:
                logger.info("All backtests completed successfully!")
        
        print(f"\nLog file: tsmom_backtest.log")
        
    except KeyboardInterrupt:
        print("\nBacktest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during backtest: {e}")
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()