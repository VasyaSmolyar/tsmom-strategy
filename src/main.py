"""
Main module for TSMOM Backtest Project.
Orchestrates the entire backtesting process.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import argparse
import yaml

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from data.data_loader import create_data_loader
from strategy.tsmom_strategy import TSMOMStrategy
from analysis.performance_analyzer import PerformanceAnalyzer


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('tsmom_backtest.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def run_full_backtest(config_path: str = "config/config.yaml",
                     download_data: bool = True,
                     generate_report: bool = True) -> dict:
    """
    Run the complete TSMOM backtest.
    
    Args:
        config_path: Path to configuration file
        download_data: Whether to download fresh data
        generate_report: Whether to generate performance report
    
    Returns:
        Dictionary with all results
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting TSMOM Backtest")
    logger.info("=" * 50)
    
    # Step 1: Data Loading and Processing
    logger.info("Step 1: Loading and processing data...")
    loader = create_data_loader(config_path)
    
    if download_data:
        logger.info("Downloading fresh data...")
        prices = loader.load_processed_data()
    else:
        logger.info("Loading existing processed data...")
        prices = loader.load_processed_data()
    
    # Calculate returns
    daily_returns = loader.calculate_returns(prices, 'D')
    monthly_returns = loader.calculate_returns(prices, 'M')
    
    logger.info(f"Data loaded: {prices.shape[0]} observations, {prices.shape[1]} assets")
    logger.info(f"Date range: {prices.index.min()} to {prices.index.max()}")
    
    # Step 2: Strategy Execution
    logger.info("Step 2: Executing TSMOM strategy...")
    strategy = TSMOMStrategy(config_path)
    strategy_results = strategy.run_strategy(daily_returns)
    
    # Generate and save trade history CSV
    try:
        initial_capital = 1_000_000.0
        try:
            initial_capital = float(
                loader.config.get('backtest', {}).get('initial_capital', initial_capital)
            )
        except Exception:
            pass
        trade_log = strategy.generate_trade_log(
            strategy_results['weights'],
            daily_returns,
            strategy_results['returns'],
            prices,
            initial_capital=initial_capital,
        )
        logger.info(f"Generated trade history with {len(trade_log)} trades")
    except Exception as e:
        logger.warning(f"Failed to generate trade history: {e}")
        trade_log = None
    
    logger.info("Strategy execution completed")
    logger.info(f"Strategy returns: {len(strategy_results['returns'])} periods")
    
    # Step 3: Performance Analysis
    if generate_report:
        logger.info("Step 3: Generating performance analysis...")
        analyzer = PerformanceAnalyzer()
        
        # Get benchmark data aligned with strategy start
        # Check if we're using MOEX data source
        with open(config_path, 'r') as file:
            import yaml
            config = yaml.safe_load(file)
        
        data_source = config['data'].get('source', 'Yahoo')
        benchmark_symbol = config['backtest'].get('benchmark', '^GSPC')
        
        if data_source == 'MOEX' and benchmark_symbol == 'IMOEX':
            # For MOEX data, load IMOEX benchmark from the same data loader
            logger.info("Loading IMOEX benchmark data from MOEX loader...")
            imoex_data = loader.load_imoex_data()
            if not imoex_data.empty:
                # Calculate IMOEX returns
                imoex_returns = imoex_data['IMOEX'].pct_change().dropna()
                # Align with strategy start
                strategy_start_date = analyzer.get_strategy_start_date_dt(strategy_results['returns'])
                benchmark_returns = imoex_returns[imoex_returns.index >= strategy_start_date]
                logger.info(f"Loaded IMOEX benchmark data: {len(benchmark_returns)} periods")
            else:
                logger.warning("No IMOEX data available, using default benchmark")
                benchmark_returns = analyzer.generate_benchmark_data_from_strategy_start(
                    strategy_results['returns']
                )
        else:
            # Use default benchmark (S&P 500)
            benchmark_returns = analyzer.generate_benchmark_data_from_strategy_start(
                strategy_results['returns']
            )
        
        # Generate comprehensive report with alignment
        analysis_results = analyzer.generate_comprehensive_report(
            strategy_results['returns'],
            benchmark_returns,
            align_with_strategy_start=True,
            data_source=data_source
        )
        
        logger.info("Performance analysis completed")
        
        # Print key metrics
        metrics = analysis_results['metrics']
        logger.info("Key Performance Metrics:")
        logger.info(f"  Total Return: {metrics.get('total_return', 0):.2%}")
        logger.info(f"  Annual Return: {metrics.get('annual_return', 0):.2%}")
        logger.info(f"  Volatility: {metrics.get('volatility', 0):.2%}")
        logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        
        if 'benchmark_annual_return' in metrics:
            logger.info(f"  Benchmark Return: {metrics.get('benchmark_annual_return', 0):.2%}")
            logger.info(f"  Excess Return: {metrics.get('excess_return', 0):.2%}")
            logger.info(f"  Information Ratio: {metrics.get('information_ratio', 0):.2f}")
    else:
        analysis_results = None
    
    # Step 4: Summary
    logger.info("Step 4: Backtest completed successfully!")
    logger.info("=" * 50)
    
    # Compile results
    results = {
        'prices': prices,
        'daily_returns': daily_returns,
        'monthly_returns': monthly_returns,
        'strategy_results': strategy_results,
        'trade_log': trade_log,
        'analysis_results': analysis_results,
        'timestamp': datetime.now().isoformat()
    }
    
    return results


def run_sensitivity_analysis(config_path: str = "config/config.yaml") -> dict:
    """
    Run sensitivity analysis with different parameters.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Dictionary with sensitivity analysis results
    """
    logger = logging.getLogger(__name__)
    logger.info("Running sensitivity analysis...")
    
    # Load data
    loader = create_data_loader(config_path)
    prices = loader.load_processed_data()
    returns = loader.calculate_returns(prices, 'D')
    
    # Test different lookback periods
    lookback_periods = [3, 6, 12, 18]
    sensitivity_results = {}
    
    for lookback in lookback_periods:
        logger.info(f"Testing lookback period: {lookback} months")
        
        # Create strategy with modified parameters
        strategy = TSMOMStrategy(config_path)
        strategy.lookback_period = lookback
        
        # Run strategy
        results = strategy.run_strategy(returns)
        
        # Calculate basic metrics
        metrics = strategy.calculate_performance_metrics(results['returns'])
        sensitivity_results[f'lookback_{lookback}'] = {
            'metrics': metrics,
            'returns': results['returns']
        }
        
        logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"  Annual Return: {metrics.get('annual_return', 0):.2%}")
    
    return sensitivity_results


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='TSMOM Backtest Project')
    parser.add_argument('--config', default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--no-download', action='store_true',
                       help='Skip data download (use existing data)')
    parser.add_argument('--no-report', action='store_true',
                       help='Skip report generation')
    parser.add_argument('--sensitivity', action='store_true',
                       help='Run sensitivity analysis')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        if args.sensitivity:
            logger.info("Running sensitivity analysis...")
            results = run_sensitivity_analysis(args.config)
            logger.info("Sensitivity analysis completed")
        else:
            logger.info("Running full backtest...")
            results = run_full_backtest(
                config_path=args.config,
                download_data=not args.no_download,
                generate_report=not args.no_report
            )
            logger.info("Full backtest completed")
        
        logger.info("All operations completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise


if __name__ == "__main__":
    main() 