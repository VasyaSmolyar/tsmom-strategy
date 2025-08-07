#!/usr/bin/env python3
"""
IMOEX Comparison Script for TSMOM Strategy.
Compares TSMOM strategy performance with IMOEX index benchmark.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from main import run_full_backtest, setup_logging
from analysis.performance_analyzer import PerformanceAnalyzer
import yaml

def print_comparison_summary(strategy_returns, benchmark_returns, analyzer):
    """
    Print detailed comparison summary between strategy and IMOEX benchmark.
    
    Args:
        strategy_returns: Strategy returns series
        benchmark_returns: IMOEX benchmark returns series
        analyzer: PerformanceAnalyzer instance
    """
    print("\n" + "=" * 80)
    print("DETAILED IMOEX COMPARISON")
    print("=" * 80)
    
    # Align strategy and benchmark data
    strategy_aligned, benchmark_aligned = analyzer.align_strategy_and_benchmark(
        strategy_returns, benchmark_returns
    )
    
    if len(strategy_aligned) == 0 or len(benchmark_aligned) == 0:
        print("ERROR: Could not align strategy and benchmark data")
        return
    
    print(f"Comparison Period: {strategy_aligned.index.min()} to {strategy_aligned.index.max()}")
    print(f"Total Periods: {len(strategy_aligned)}")
    print()
    
    # Calculate comprehensive metrics
    metrics = analyzer.calculate_comprehensive_metrics(strategy_aligned, benchmark_aligned)
    
    # Strategy metrics
    print("STRATEGY PERFORMANCE:")
    print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
    print(f"  Annual Return: {metrics.get('annual_return', 0):.2%}")
    print(f"  Volatility: {metrics.get('volatility', 0):.2%}")
    print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    print(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
    print(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    
    # IMOEX benchmark metrics
    print("\nIMOEX BENCHMARK PERFORMANCE:")
    if 'benchmark_total_return' in metrics:
        print(f"  Total Return: {metrics.get('benchmark_total_return', 0):.2%}")
        print(f"  Annual Return: {metrics.get('benchmark_annual_return', 0):.2%}")
        print(f"  Volatility: {metrics.get('benchmark_volatility', 0):.2%}")
        print(f"  Sharpe Ratio: {metrics.get('benchmark_sharpe_ratio', 0):.2f}")
    
    # Comparison metrics
    print("\nCOMPARISON METRICS:")
    print(f"  Excess Return: {metrics.get('excess_return', 0):.2%}")
    print(f"  Tracking Error: {metrics.get('tracking_error', 0):.2%}")
    print(f"  Information Ratio: {metrics.get('information_ratio', 0):.2f}")
    print(f"  Beta: {metrics.get('beta', 0):.4f}")
    print(f"  Alpha: {metrics.get('alpha', 0):.2%}")
    
    # Risk metrics
    print("\nRISK METRICS:")
    print(f"  Strategy VaR (95%): {metrics.get('var_95', 0):.2%}")
    print(f"  Strategy CVaR (95%): {metrics.get('cvar_95', 0):.2%}")
    print(f"  Strategy Skewness: {metrics.get('skewness', 0):.2f}")
    print(f"  Strategy Kurtosis: {metrics.get('kurtosis', 0):.2f}")
    
    return metrics

def plot_detailed_comparison(strategy_returns, benchmark_returns, analyzer):
    """
    Create detailed comparison plots for strategy vs IMOEX.
    
    Args:
        strategy_returns: Strategy returns series
        benchmark_returns: IMOEX benchmark returns series
        analyzer: PerformanceAnalyzer instance
    """
    # Align data
    strategy_aligned, benchmark_aligned = analyzer.align_strategy_and_benchmark(
        strategy_returns, benchmark_returns
    )
    
    if len(strategy_aligned) == 0 or len(benchmark_aligned) == 0:
        print("ERROR: Could not align strategy and benchmark data for plotting")
        return
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Cumulative Returns Comparison
    strategy_cumulative = (1 + strategy_aligned).cumprod()
    benchmark_cumulative = (1 + benchmark_aligned).cumprod()
    
    ax1.plot(strategy_cumulative.index, strategy_cumulative.values,
             label='TSMOM Strategy', linewidth=2, color='blue')
    ax1.plot(benchmark_cumulative.index, benchmark_cumulative.values,
             label='IMOEX Index', linewidth=2, color='red', alpha=0.7)
    ax1.set_title('Cumulative Returns: TSMOM vs IMOEX', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Rolling Sharpe Ratios
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
             label='IMOEX Index', linewidth=2, color='red', alpha=0.7)
    ax2.set_title('Rolling Sharpe Ratios (1-year window)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 3. Rolling Volatilities
    ax3.plot(strategy_rolling_std.index, strategy_rolling_std.values,
             label='TSMOM Strategy', linewidth=2, color='blue')
    ax3.plot(benchmark_rolling_std.index, benchmark_rolling_std.values,
             label='IMOEX Index', linewidth=2, color='red', alpha=0.7)
    ax3.set_title('Rolling Volatilities (1-year window)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Annualized Volatility')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Excess Returns
    excess_returns = strategy_aligned - benchmark_aligned
    cumulative_excess = (1 + excess_returns).cumprod()
    
    ax4.plot(cumulative_excess.index, cumulative_excess.values,
             label='Cumulative Excess Return', linewidth=2, color='green')
    ax4.set_title('Cumulative Excess Return (Strategy - IMOEX)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Cumulative Excess Return')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path("reports/plots") / "strategy_vs_imoex_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Detailed comparison plot saved to: {plot_path}")

def main():
    """Run detailed IMOEX comparison analysis."""
    print("IMOEX Comparison Analysis")
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
        
        if not results.get('analysis_results'):
            print("ERROR: No analysis results available")
            return
        
        # Get strategy and benchmark returns
        strategy_returns = results['strategy_results']['returns']
        benchmark_returns = results['analysis_results'].get('benchmark_returns')
        
        if benchmark_returns is None or len(benchmark_returns) == 0:
            print("ERROR: No benchmark returns available")
            return
        
        # Create analyzer
        analyzer = PerformanceAnalyzer()
        
        # Print detailed comparison
        metrics = print_comparison_summary(strategy_returns, benchmark_returns, analyzer)
        
        # Create detailed comparison plots
        plot_detailed_comparison(strategy_returns, benchmark_returns, analyzer)
        
        print("\n" + "=" * 80)
        print("IMOEX COMPARISON COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Reports saved to: reports/")
        print(f"Detailed comparison plot: reports/plots/strategy_vs_imoex_comparison.png")
        
    except Exception as e:
        logger.error(f"Error during IMOEX comparison: {e}")
        print(f"\nERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
