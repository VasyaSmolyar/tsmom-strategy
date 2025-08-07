"""
Performance Analysis Module for TSMOM Strategy.
Provides comprehensive analysis and visualization of strategy performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """Analyzes and visualizes TSMOM strategy performance."""
    
    def __init__(self, reports_dir: str = "reports"):
        """Initialize PerformanceAnalyzer."""
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.reports_dir / "plots").mkdir(exist_ok=True)
        (self.reports_dir / "tables").mkdir(exist_ok=True)
    
    def calculate_comprehensive_metrics(self, strategy_returns: pd.Series,
                                     benchmark_returns: Optional[pd.Series] = None) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            strategy_returns: Strategy returns series
            benchmark_returns: Benchmark returns series (optional)
        
        Returns:
            Dictionary with comprehensive metrics
        """
        # Remove NaN values
        strategy_clean = strategy_returns.dropna()
        
        if len(strategy_clean) == 0:
            return {}
        
        # Basic return metrics
        total_return = (1 + strategy_clean).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(strategy_clean)) - 1
        volatility = strategy_clean.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        cumulative_returns = (1 + strategy_clean).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate drawdown duration
        underwater = drawdown < 0
        underwater_periods = underwater.sum()
        max_underwater_duration = underwater.astype(int).groupby(
            underwater.ne(underwater.shift()).cumsum()
        ).sum().max()
        
        # Risk metrics
        var_95 = np.percentile(strategy_clean, 5)
        cvar_95 = strategy_clean[strategy_clean <= var_95].mean()
        
        # Return distribution metrics
        positive_returns = strategy_clean[strategy_clean > 0]
        negative_returns = strategy_clean[strategy_clean < 0]
        
        win_rate = len(positive_returns) / len(strategy_clean)
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
        profit_factor = abs(positive_returns.sum() / negative_returns.sum()) if negative_returns.sum() != 0 else float('inf')
        
        # Skewness and kurtosis
        skewness = strategy_clean.skew()
        kurtosis = strategy_clean.kurtosis()
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio (using downside deviation)
        downside_returns = strategy_clean[strategy_clean < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annual_return / downside_deviation if downside_deviation > 0 else 0
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'max_underwater_duration': max_underwater_duration,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'num_periods': len(strategy_clean)
        }
        
        # Benchmark comparison if provided
        if benchmark_returns is not None:
            benchmark_clean = benchmark_returns.dropna()
            if len(benchmark_clean) > 0:
                # Align dates
                common_dates = strategy_clean.index.intersection(benchmark_clean.index)
                if len(common_dates) > 0:
                    strategy_aligned = strategy_clean.loc[common_dates]
                    benchmark_aligned = benchmark_clean.loc[common_dates]
                    
                    # Benchmark metrics
                    benchmark_total_return = (1 + benchmark_aligned).prod() - 1
                    benchmark_annual_return = (1 + benchmark_total_return) ** (252 / len(benchmark_aligned)) - 1
                    benchmark_volatility = benchmark_aligned.std() * np.sqrt(252)
                    benchmark_sharpe = benchmark_annual_return / benchmark_volatility if benchmark_volatility > 0 else 0
                    
                    # Excess returns
                    excess_returns = strategy_aligned - benchmark_aligned
                    excess_return = excess_returns.mean() * 252
                    tracking_error = excess_returns.std() * np.sqrt(252)
                    information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
                    
                    # Beta and Alpha
                    covariance = np.cov(strategy_aligned, benchmark_aligned)[0, 1]
                    benchmark_variance = benchmark_aligned.var()
                    beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                    
                    alpha = excess_return - (beta * benchmark_annual_return)
                    
                    metrics.update({
                        'benchmark_total_return': benchmark_total_return,
                        'benchmark_annual_return': benchmark_annual_return,
                        'benchmark_volatility': benchmark_volatility,
                        'benchmark_sharpe_ratio': benchmark_sharpe,
                        'excess_return': excess_return,
                        'tracking_error': tracking_error,
                        'information_ratio': information_ratio,
                        'beta': beta,
                        'alpha': alpha
                    })
        
        return metrics
    
    def plot_cumulative_returns(self, strategy_returns: pd.Series,
                               benchmark_returns: Optional[pd.Series] = None,
                               title: str = "Cumulative Returns") -> None:
        """Plot cumulative returns comparison."""
        plt.figure(figsize=(12, 8))
        
        # Strategy cumulative returns
        strategy_cumulative = (1 + strategy_returns.dropna()).cumprod()
        plt.plot(strategy_cumulative.index, strategy_cumulative.values, 
                label='TSMOM Strategy', linewidth=2)
        
        # Benchmark cumulative returns
        if benchmark_returns is not None:
            benchmark_cumulative = (1 + benchmark_returns.dropna()).cumprod()
            plt.plot(benchmark_cumulative.index, benchmark_cumulative.values, 
                    label='Benchmark (S&P 500)', linewidth=2, alpha=0.7)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Return', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Set y-axis to logarithmic scale
        plt.tight_layout()
        
        # Save plot
        plot_path = self.reports_dir / "plots" / "cumulative_returns.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved cumulative returns plot to {plot_path}")
    
    def plot_drawdown(self, strategy_returns: pd.Series,
                     title: str = "Drawdown Analysis") -> None:
        """Plot drawdown analysis."""
        plt.figure(figsize=(12, 8))
        
        # Calculate drawdown
        cumulative_returns = (1 + strategy_returns.dropna()).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Plot drawdown
        plt.fill_between(drawdown.index, drawdown.values, 0, 
                        alpha=0.3, color='red', label='Drawdown')
        plt.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Drawdown', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = self.reports_dir / "plots" / "drawdown.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved drawdown plot to {plot_path}")
    
    def plot_returns_distribution(self, strategy_returns: pd.Series,
                                 title: str = "Returns Distribution") -> None:
        """Plot returns distribution analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        returns_clean = strategy_returns.dropna()
        
        # Histogram
        ax1.hist(returns_clean, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(returns_clean.mean(), color='red', linestyle='--', 
                   label=f'Mean: {returns_clean.mean():.4f}')
        ax1.axvline(returns_clean.median(), color='green', linestyle='--', 
                   label=f'Median: {returns_clean.median():.4f}')
        ax1.set_title('Returns Histogram', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Return', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(returns_clean, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normal Distribution)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.reports_dir / "plots" / "returns_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved returns distribution plot to {plot_path}")
    
    def plot_rolling_metrics(self, strategy_returns: pd.Series,
                            window: int = 252,
                            title: str = "Rolling Performance Metrics") -> None:
        """Plot rolling performance metrics."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        returns_clean = strategy_returns.dropna()
        
        # Rolling Sharpe ratio
        rolling_mean = returns_clean.rolling(window=window).mean() * 252
        rolling_std = returns_clean.rolling(window=window).std() * np.sqrt(252)
        rolling_sharpe = rolling_mean / rolling_std
        ax1.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2)
        ax1.set_title('Rolling Sharpe Ratio', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Sharpe Ratio', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Rolling volatility
        ax2.plot(rolling_std.index, rolling_std.values, linewidth=2, color='orange')
        ax2.set_title('Rolling Volatility', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Volatility', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Rolling annual return
        ax3.plot(rolling_mean.index, rolling_mean.values, linewidth=2, color='green')
        ax3.set_title('Rolling Annual Return', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Annual Return', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Rolling drawdown
        cumulative_returns = (1 + returns_clean).cumprod()
        rolling_max = cumulative_returns.rolling(window=window).max()
        rolling_drawdown = (cumulative_returns - rolling_max) / rolling_max
        ax4.fill_between(rolling_drawdown.index, rolling_drawdown.values, 0, 
                        alpha=0.3, color='red')
        ax4.plot(rolling_drawdown.index, rolling_drawdown.values, color='red', linewidth=1)
        ax4.set_title('Rolling Drawdown', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Drawdown', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.reports_dir / "plots" / "rolling_metrics.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved rolling metrics plot to {plot_path}")
    
    def create_performance_table(self, metrics: Dict) -> pd.DataFrame:
        """Create a formatted performance metrics table."""
        # Convert metrics to DataFrame
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        
        # Format numeric values
        def format_value(value):
            if isinstance(value, float):
                if abs(value) < 0.01:
                    return f"{value:.6f}"
                elif abs(value) < 1:
                    return f"{value:.4f}"
                else:
                    return f"{value:.2f}"
            return str(value)
        
        metrics_df['Formatted_Value'] = metrics_df['Value'].apply(format_value)
        
        # Save table
        table_path = self.reports_dir / "tables" / "performance_metrics.csv"
        metrics_df.to_csv(table_path, index=False)
        
        logger.info(f"Saved performance metrics table to {table_path}")
        
        return metrics_df
    
    def generate_benchmark_data(self, symbol: str = "^GSPC",
                               start_date: str = "2000-01-01",
                               end_date: str = "2023-12-31") -> pd.Series:
        """Download benchmark data (S&P 500 by default)."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            benchmark_returns = data['Close'].pct_change().dropna()
            logger.info(f"Downloaded benchmark data for {symbol}")
            return benchmark_returns
        except Exception as e:
            logger.error(f"Error downloading benchmark data: {e}")
            return pd.Series()
    
    def get_strategy_start_date(self, strategy_returns: pd.Series) -> str:
        """
        Get the first date when strategy is actually applied (first non-zero return).
        
        Args:
            strategy_returns: Strategy returns series
        
        Returns:
            String with the first strategy date
        """
        # Find first non-zero return (strategy actually starts)
        non_zero_returns = strategy_returns[strategy_returns != 0]
        if len(non_zero_returns) > 0:
            first_strategy_date = non_zero_returns.index[0]
            return first_strategy_date.strftime('%Y-%m-%d')
        else:
            # Fallback to first non-null date
            first_date = strategy_returns.dropna().index[0]
            return first_date.strftime('%Y-%m-%d')
    
    def get_strategy_start_date_dt(self, strategy_returns: pd.Series) -> pd.Timestamp:
        """
        Get the first date when strategy is actually applied (first non-zero return).
        
        Args:
            strategy_returns: Strategy returns series
        
        Returns:
            Timestamp with the first strategy date
        """
        # Find first non-zero return (strategy actually starts)
        non_zero_returns = strategy_returns[strategy_returns != 0]
        if len(non_zero_returns) > 0:
            return non_zero_returns.index[0]
        else:
            # Fallback to first non-null date
            return strategy_returns.dropna().index[0]
    
    def generate_benchmark_data_from_strategy_start(self, strategy_returns: pd.Series,
                                                   symbol: str = "^GSPC",
                                                   end_date: str = "2023-12-31") -> pd.Series:
        """
        Download benchmark data starting from the first day when strategy is applied.
        
        Args:
            strategy_returns: Strategy returns series
            symbol: Benchmark symbol (default: S&P 500)
            end_date: End date for benchmark data
        
        Returns:
            Benchmark returns series aligned with strategy start
        """
        # Get the first strategy date
        start_date = self.get_strategy_start_date(strategy_returns)
        
        # Add some buffer to ensure we have data
        if isinstance(start_date, str):
            start_date_dt = pd.to_datetime(start_date)
        else:
            start_date_dt = start_date
        
        # Go back a few days to ensure we have the start date
        buffer_start = start_date_dt - pd.Timedelta(days=5)
        buffer_start_str = buffer_start.strftime('%Y-%m-%d')
        
        try:
            # For IMOEX, we need to use a different approach since it's not available on Yahoo Finance
            if symbol == "IMOEX":
                # Try to load IMOEX data from local files or use alternative source
                logger.info("Loading IMOEX benchmark data from local files...")
                # This will be handled by the data loader
                return pd.Series()
            else:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=buffer_start_str, end=end_date)
                benchmark_returns = data['Close'].pct_change().dropna()
                
                # Normalize timezone to UTC and then remove timezone info
                benchmark_returns.index = benchmark_returns.index.tz_localize(None)
                
                logger.info(f"Downloaded benchmark data for {symbol} from {buffer_start_str}")
                logger.info(f"Strategy start date: {start_date}")
                logger.info(f"Benchmark data points: {len(benchmark_returns)}")
                
                return benchmark_returns
        except Exception as e:
            logger.error(f"Error downloading benchmark data: {e}")
            return pd.Series()
    
    def align_strategy_and_benchmark(self, strategy_returns: pd.Series,
                                   benchmark_returns: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Align strategy returns with benchmark returns starting from strategy start date.
        
        Args:
            strategy_returns: Strategy returns series
            benchmark_returns: Benchmark returns series
        
        Returns:
            Tuple of aligned strategy and benchmark returns
        """
        # Get strategy start date
        strategy_start_date = self.get_strategy_start_date_dt(strategy_returns)
        logger.info(f"Strategy start date: {strategy_start_date}")
        
        # Filter strategy returns from start date
        strategy_aligned = strategy_returns[strategy_returns.index >= strategy_start_date]
        logger.info(f"Strategy aligned data range: {strategy_aligned.index.min()} to {strategy_aligned.index.max()}")
        logger.info(f"Strategy aligned data points: {len(strategy_aligned)}")
        
        # Log benchmark data range
        logger.info(f"Benchmark data range: {benchmark_returns.index.min()} to {benchmark_returns.index.max()}")
        logger.info(f"Benchmark data points: {len(benchmark_returns)}")
        
        # Align benchmark returns with strategy
        common_dates = strategy_aligned.index.intersection(benchmark_returns.index)
        
        if len(common_dates) == 0:
            logger.warning("No common dates between strategy and benchmark")
            logger.warning(f"Strategy dates: {strategy_aligned.index[:5].tolist()} ... {strategy_aligned.index[-5:].tolist()}")
            logger.warning(f"Benchmark dates: {benchmark_returns.index[:5].tolist()} ... {benchmark_returns.index[-5:].tolist()}")
            return strategy_aligned, pd.Series()
        
        strategy_final = strategy_aligned.loc[common_dates]
        benchmark_final = benchmark_returns.loc[common_dates]
        
        logger.info(f"Aligned strategy and benchmark from {strategy_start_date}")
        logger.info(f"Aligned period: {len(strategy_final)} periods")
        logger.info(f"Final aligned range: {strategy_final.index.min()} to {strategy_final.index.max()}")
        
        return strategy_final, benchmark_final
    
    def generate_comprehensive_report(self, strategy_returns: pd.Series,
                                   benchmark_returns: Optional[pd.Series] = None,
                                   align_with_strategy_start: bool = True) -> Dict:
        """
        Generate comprehensive performance report.
        
        Args:
            strategy_returns: Strategy returns series
            benchmark_returns: Benchmark returns series (optional)
            align_with_strategy_start: Whether to align benchmark with strategy start date
        """
        logger.info("Generating comprehensive performance report...")
        
        # If benchmark is provided and we want to align with strategy start
        if benchmark_returns is not None and align_with_strategy_start:
            strategy_aligned, benchmark_aligned = self.align_strategy_and_benchmark(
                strategy_returns, benchmark_returns
            )
            
            if len(strategy_aligned) > 0 and len(benchmark_aligned) > 0:
                strategy_returns = strategy_aligned
                benchmark_returns = benchmark_aligned
                logger.info("Aligned strategy and benchmark with strategy start date")
            else:
                logger.warning("Could not align strategy and benchmark, using original data")
        
        # Calculate metrics
        metrics = self.calculate_comprehensive_metrics(strategy_returns, benchmark_returns)
        
        # Generate plots
        self.plot_cumulative_returns(strategy_returns, benchmark_returns)
        self.plot_drawdown(strategy_returns)
        self.plot_returns_distribution(strategy_returns)
        self.plot_rolling_metrics(strategy_returns)
        
        # Create performance table
        metrics_table = self.create_performance_table(metrics)
        
        # Generate markdown report
        self.generate_markdown_report(metrics, strategy_returns, benchmark_returns)
        
        logger.info("Comprehensive report generated successfully")
        
        return {
            'metrics': metrics,
            'metrics_table': metrics_table,
            'strategy_returns': strategy_returns,
            'benchmark_returns': benchmark_returns
        }
    
    def generate_markdown_report(self, metrics: Dict,
                               strategy_returns: pd.Series,
                               benchmark_returns: Optional[pd.Series] = None) -> None:
        """Generate markdown report."""
        report_path = self.reports_dir / "performance_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# TSMOM Strategy Performance Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Return**: {metrics.get('total_return', 0):.2%}\n")
            f.write(f"- **Annual Return**: {metrics.get('annual_return', 0):.2%}\n")
            f.write(f"- **Volatility**: {metrics.get('volatility', 0):.2%}\n")
            f.write(f"- **Sharpe Ratio**: {metrics.get('sharpe_ratio', 0):.2f}\n")
            f.write(f"- **Maximum Drawdown**: {metrics.get('max_drawdown', 0):.2%}\n\n")
            
            f.write("## Performance Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            
            for metric, value in metrics.items():
                if isinstance(value, float):
                    if abs(value) < 0.01:
                        formatted_value = f"{value:.6f}"
                    elif abs(value) < 1:
                        formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)
                
                f.write(f"| {metric.replace('_', ' ').title()} | {formatted_value} |\n")
            
            f.write("\n## Risk Analysis\n\n")
            f.write(f"- **Value at Risk (95%)**: {metrics.get('var_95', 0):.4f}\n")
            f.write(f"- **Conditional VaR (95%)**: {metrics.get('cvar_95', 0):.4f}\n")
            f.write(f"- **Skewness**: {metrics.get('skewness', 0):.4f}\n")
            f.write(f"- **Kurtosis**: {metrics.get('kurtosis', 0):.4f}\n\n")
            
            if benchmark_returns is not None:
                f.write("## Benchmark Comparison\n\n")
                f.write(f"- **Excess Return**: {metrics.get('excess_return', 0):.2%}\n")
                f.write(f"- **Information Ratio**: {metrics.get('information_ratio', 0):.2f}\n")
                f.write(f"- **Beta**: {metrics.get('beta', 0):.4f}\n")
                f.write(f"- **Alpha**: {metrics.get('alpha', 0):.2%}\n\n")
            
            f.write("## Generated Plots\n\n")
            f.write("The following plots have been generated:\n")
            f.write("- Cumulative Returns Comparison\n")
            f.write("- Drawdown Analysis\n")
            f.write("- Returns Distribution\n")
            f.write("- Rolling Performance Metrics\n\n")
            
            f.write("## Methodology\n\n")
            f.write("This analysis is based on the TSMOM (Time Series Momentum) strategy:\n")
            f.write("- **Lookback Period**: 12 months\n")
            f.write("- **Holding Period**: 1 month\n")
            f.write("- **Target Volatility**: 40% annual\n")
            f.write("- **Rebalancing**: Monthly\n")
            f.write("- **Position Sizing**: Inverse volatility weighting\n\n")
        
        logger.info(f"Generated markdown report: {report_path}")


def main():
    """Test the performance analyzer."""
    import sys
    sys.path.append('src')
    
    from data.data_loader import DataLoader
    from strategy.tsmom_strategy import TSMOMStrategy
    
    # Load data and run strategy
    loader = DataLoader()
    prices = loader.load_processed_data()
    returns = loader.calculate_returns(prices, 'D')
    
    strategy = TSMOMStrategy()
    results = strategy.run_strategy(returns)
    
    # Analyze performance
    analyzer = PerformanceAnalyzer()
    
    # Get benchmark data aligned with strategy start
    benchmark_returns = analyzer.generate_benchmark_data_from_strategy_start(
        results['returns']
    )
    
    # Generate comprehensive report with alignment
    report = analyzer.generate_comprehensive_report(
        results['returns'], 
        benchmark_returns,
        align_with_strategy_start=True
    )
    
    print("Performance analysis completed!")
    print(f"Reports saved to: {analyzer.reports_dir}")


if __name__ == "__main__":
    main() 