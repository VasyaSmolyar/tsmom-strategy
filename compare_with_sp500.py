#!/usr/bin/env python3
"""
Compare TSMOM Strategy with S&P500 starting from the first day when strategy is applied.
This script ensures that the benchmark comparison starts from the actual strategy start date,
not from the beginning of the data period.
"""

import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.data_loader import DataLoader
from strategy.tsmom_strategy import TSMOMStrategy
from analysis.performance_analyzer import PerformanceAnalyzer


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('strategy_comparison.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def print_comparison_summary(strategy_returns, benchmark_returns, analyzer):
    """Print detailed comparison summary."""
    print("\n" + "="*80)
    print("STRATEGY vs S&P500 COMPARISON SUMMARY")
    print("="*80)
    
    # Get strategy start date
    strategy_start = analyzer.get_strategy_start_date(strategy_returns)
    print(f"Strategy Start Date: {strategy_start}")
    
    # Align data
    strategy_aligned, benchmark_aligned = analyzer.align_strategy_and_benchmark(
        strategy_returns, benchmark_returns
    )
    
    if len(strategy_aligned) == 0 or len(benchmark_aligned) == 0:
        print("ERROR: Could not align strategy and benchmark data")
        return
    
    # Calculate metrics
    metrics = analyzer.calculate_comprehensive_metrics(strategy_aligned, benchmark_aligned)
    
    print(f"\nComparison Period: {len(strategy_aligned)} trading days")
    print(f"From: {strategy_aligned.index[0].strftime('%Y-%m-%d')}")
    print(f"To: {strategy_aligned.index[-1].strftime('%Y-%m-%d')}")
    
    print("\n" + "-"*50)
    print("PERFORMANCE METRICS")
    print("-"*50)
    
    # Strategy metrics
    print(f"Strategy Total Return: {metrics.get('total_return', 0):.2%}")
    print(f"Strategy Annual Return: {metrics.get('annual_return', 0):.2%}")
    print(f"Strategy Volatility: {metrics.get('volatility', 0):.2%}")
    print(f"Strategy Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Strategy Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    
    # Benchmark metrics
    if 'benchmark_total_return' in metrics:
        print(f"\nS&P500 Total Return: {metrics.get('benchmark_total_return', 0):.2%}")
        print(f"S&P500 Annual Return: {metrics.get('benchmark_annual_return', 0):.2%}")
        print(f"S&P500 Volatility: {metrics.get('benchmark_volatility', 0):.2%}")
        print(f"S&P500 Sharpe Ratio: {metrics.get('benchmark_sharpe_ratio', 0):.2f}")
    
    # Comparison metrics
    if 'excess_return' in metrics:
        print(f"\nExcess Return: {metrics.get('excess_return', 0):.2%}")
        print(f"Information Ratio: {metrics.get('information_ratio', 0):.2f}")
        print(f"Beta: {metrics.get('beta', 0):.4f}")
        print(f"Alpha: {metrics.get('alpha', 0):.2%}")
        print(f"Tracking Error: {metrics.get('tracking_error', 0):.2%}")
    
    # Risk metrics
    print(f"\nRisk Metrics:")
    print(f"Strategy VaR (95%): {metrics.get('var_95', 0):.4f}")
    print(f"Strategy CVaR (95%): {metrics.get('cvar_95', 0):.4f}")
    print(f"Strategy Skewness: {metrics.get('skewness', 0):.4f}")
    print(f"Strategy Kurtosis: {metrics.get('kurtosis', 0):.4f}")
    
    # Win rate analysis
    print(f"\nWin Rate Analysis:")
    print(f"Strategy Win Rate: {metrics.get('win_rate', 0):.2%}")
    print(f"Strategy Avg Win: {metrics.get('avg_win', 0):.4f}")
    print(f"Strategy Avg Loss: {metrics.get('avg_loss', 0):.4f}")
    print(f"Strategy Profit Factor: {metrics.get('profit_factor', 0):.2f}")


def plot_detailed_comparison(strategy_returns, benchmark_returns, analyzer):
    """Create detailed comparison plots."""
    # Align data
    strategy_aligned, benchmark_aligned = analyzer.align_strategy_and_benchmark(
        strategy_returns, benchmark_returns
    )
    
    if len(strategy_aligned) == 0 or len(benchmark_aligned) == 0:
        print("ERROR: Could not align data for plotting")
        return
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Cumulative returns comparison
    strategy_cumulative = (1 + strategy_aligned).cumprod()
    benchmark_cumulative = (1 + benchmark_aligned).cumprod()
    
    ax1.plot(strategy_cumulative.index, strategy_cumulative.values, 
             label='TSMOM Strategy', linewidth=2, color='blue')
    ax1.plot(benchmark_cumulative.index, benchmark_cumulative.values, 
             label='S&P500', linewidth=2, color='red', alpha=0.7)
    ax1.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Rolling Sharpe ratio comparison
    window = 252  # 1 year
    strategy_rolling_mean = strategy_aligned.rolling(window=window).mean() * 252
    strategy_rolling_std = strategy_aligned.rolling(window=window).std() * np.sqrt(252)
    strategy_rolling_sharpe = strategy_rolling_mean / strategy_rolling_std
    
    benchmark_rolling_mean = benchmark_aligned.rolling(window=window).mean() * 252
    benchmark_rolling_std = benchmark_aligned.rolling(window=window).std() * np.sqrt(252)
    benchmark_rolling_sharpe = benchmark_rolling_mean / benchmark_rolling_std
    
    ax2.plot(strategy_rolling_sharpe.index, strategy_rolling_sharpe.values, 
             label='TSMOM Strategy', linewidth=2, color='blue')
    ax2.plot(benchmark_rolling_sharpe.index, benchmark_rolling_sharpe.values, 
             label='S&P500', linewidth=2, color='red', alpha=0.7)
    ax2.set_title('Rolling Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Rolling volatility comparison
    ax3.plot(strategy_rolling_std.index, strategy_rolling_std.values, 
             label='TSMOM Strategy', linewidth=2, color='blue')
    ax3.plot(benchmark_rolling_std.index, benchmark_rolling_std.values, 
             label='S&P500', linewidth=2, color='red', alpha=0.7)
    ax3.set_title('Rolling Volatility Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Volatility')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Excess returns over time
    excess_returns = strategy_aligned - benchmark_aligned
    cumulative_excess = (1 + excess_returns).cumprod()
    
    ax4.plot(cumulative_excess.index, cumulative_excess.values, 
             label='Cumulative Excess Return', linewidth=2, color='green')
    ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax4.set_title('Cumulative Excess Return vs S&P500', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Cumulative Excess Return')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path("reports/plots/strategy_vs_sp500_comparison.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nDetailed comparison plot saved to: {plot_path}")


def main():
    """Main function to run strategy comparison with S&P500."""
    print("TSMOM Strategy vs S&P500 Comparison")
    print("="*50)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Step 1: Load data
        logger.info("Loading data...")
        loader = DataLoader()
        prices = loader.load_processed_data()
        returns = loader.calculate_returns(prices, 'D')
        
        logger.info(f"Data loaded: {prices.shape[0]} observations, {prices.shape[1]} assets")
        logger.info(f"Date range: {prices.index.min()} to {prices.index.max()}")
        
        # Step 2: Run strategy
        logger.info("Running TSMOM strategy...")
        strategy = TSMOMStrategy()
        strategy_results = strategy.run_strategy(returns, prices)
        
        logger.info(f"Strategy executed: {len(strategy_results['returns'])} periods")
        
        # Step 3: Get benchmark data aligned with strategy start
        logger.info("Getting S&P500 benchmark data...")
        analyzer = PerformanceAnalyzer()
        
        benchmark_returns = analyzer.generate_benchmark_data_from_strategy_start(
            strategy_results['returns']
        )
        
        if len(benchmark_returns) == 0:
            logger.error("Failed to download benchmark data")
            return
        
        # Step 4: Print comparison summary
        print_comparison_summary(
            strategy_results['returns'], 
            benchmark_returns, 
            analyzer
        )
        
        # Step 5: Create detailed comparison plots
        logger.info("Creating detailed comparison plots...")
        plot_detailed_comparison(
            strategy_results['returns'], 
            benchmark_returns, 
            analyzer
        )
        
        # Step 6: Generate comprehensive report
        logger.info("Generating comprehensive report...")
        analysis_results = analyzer.generate_comprehensive_report(
            strategy_results['returns'],
            benchmark_returns,
            align_with_strategy_start=True,
            data_source="Yahoo"
        )
        
        print("\n" + "="*50)
        print("COMPARISON COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Reports saved to: reports/")
        print(f"Log file: strategy_comparison.log")
        
    except Exception as e:
        logger.error(f"Error during comparison: {e}")
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 