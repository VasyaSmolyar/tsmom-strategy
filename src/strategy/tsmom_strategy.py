"""
TSMOM (Time Series Momentum) Strategy Implementation.
Based on Moskowitz, Ooi, and Pedersen (2012).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


class TSMOMStrategy:
    """
    Time Series Momentum Strategy implementation.
    
    The strategy:
    1. Calculates momentum signals based on lookback period returns
    2. Takes long positions for positive momentum, short for negative
    3. Sizes positions inversely to volatility (target volatility approach)
    4. Rebalances monthly
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize TSMOM strategy with configuration."""
        self.config = self._load_config(config_path)
        self.strategy_config = self.config['strategy']
        self.risk_config = self.config['risk']
        
        # Strategy parameters
        self.lookback_period = self.strategy_config['lookback_period']  # months
        self.holding_period = self.strategy_config['holding_period']    # months
        self.target_volatility = self.strategy_config['target_volatility']
        self.transaction_cost = self.strategy_config['transaction_cost']
        
        # Risk management parameters
        self.volatility_lookback = self.risk_config['volatility_lookback']  # months
        self.max_position_size = self.risk_config['max_position_size']
        
        logger.info(f"Initialized TSMOM strategy with lookback={self.lookback_period} months")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def calculate_momentum_signals(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum signals based on lookback period returns.
        
        Args:
            returns: DataFrame with asset returns (columns are assets, index is time)
        
        Returns:
            DataFrame with momentum signals (1 for long, -1 for short, 0 for neutral)
        """
        logger.info("Calculating momentum signals...")
        
        # Calculate lookback period returns
        lookback_days = self.lookback_period * 21  # Approximate trading days per month
        
        # Calculate cumulative returns over lookback period
        momentum_returns = returns.rolling(window=lookback_days).apply(
            lambda x: (1 + x).prod() - 1, raw=True
        )
        
        # Generate signals based on momentum
        signals = pd.DataFrame(index=momentum_returns.index, columns=momentum_returns.columns)
        
        # Long position for positive momentum, short for negative
        signals[momentum_returns > 0] = 1   # Long
        signals[momentum_returns < 0] = -1  # Short
        signals[momentum_returns == 0] = 0  # Neutral
        
        # Forward fill signals for holding period
        signals = signals.fillna(method='ffill', limit=self.holding_period * 21)
        
        logger.info(f"Generated momentum signals for {signals.shape[1]} assets")
        
        return signals
    
    def calculate_volatility(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rolling volatility for each asset.
        
        Args:
            returns: DataFrame with asset returns
        
        Returns:
            DataFrame with annualized volatility
        """
        logger.info("Calculating volatility...")
        
        # Calculate rolling volatility
        vol_lookback_days = self.volatility_lookback * 21
        volatility = returns.rolling(window=vol_lookback_days).std() * np.sqrt(252)
        
        return volatility
    
    def calculate_position_sizes(self, signals: pd.DataFrame, 
                               volatility: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate position sizes using inverse volatility weighting.
        
        Args:
            signals: DataFrame with momentum signals
            volatility: DataFrame with asset volatilities
        
        Returns:
            DataFrame with position sizes (weights)
        """
        logger.info("Calculating position sizes...")
        
        # Initialize position sizes
        position_sizes = pd.DataFrame(0, index=signals.index, columns=signals.columns)
        
        for date in signals.index:
            if date in volatility.index:
                # Get current signals and volatilities
                current_signals = signals.loc[date]
                current_vol = volatility.loc[date]
                
                # Calculate inverse volatility weights
                # Only for assets with valid volatility data
                valid_assets = current_vol.dropna().index
                
                if len(valid_assets) > 0:
                    # Calculate inverse volatility weights
                    inv_vol = 1 / current_vol[valid_assets]
                    weights = inv_vol / inv_vol.sum()
                    
                    # Apply target volatility scaling
                    portfolio_vol = np.sqrt((weights ** 2 * current_vol[valid_assets] ** 2).sum())
                    if portfolio_vol > 0:
                        vol_scale = self.target_volatility / portfolio_vol
                        weights = weights * vol_scale
                    
                    # Apply momentum signals
                    for asset in valid_assets:
                        if asset in current_signals.index:
                            position_sizes.loc[date, asset] = (
                                current_signals[asset] * weights[asset]
                            )
        
        # Apply maximum position size constraint
        position_sizes = position_sizes.clip(-self.max_position_size, self.max_position_size)
        
        logger.info("Position sizes calculated")
        
        return position_sizes
    
    def generate_portfolio_weights(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Generate complete portfolio weights for the TSMOM strategy.
        
        Args:
            returns: DataFrame with asset returns
        
        Returns:
            DataFrame with portfolio weights
        """
        logger.info("Generating portfolio weights...")
        
        # Calculate momentum signals
        signals = self.calculate_momentum_signals(returns)
        
        # Calculate volatility
        volatility = self.calculate_volatility(returns)
        
        # Calculate position sizes
        weights = self.calculate_position_sizes(signals, volatility)
        
        # Ensure weights sum to target volatility (approximately)
        # This is a simplified approach - in practice, you might want more sophisticated
        # portfolio construction that ensures exact target volatility
        
        logger.info(f"Generated portfolio weights with shape: {weights.shape}")
        
        return weights
    
    def calculate_strategy_returns(self, weights: pd.DataFrame, 
                                 returns: pd.DataFrame) -> pd.Series:
        """
        Calculate strategy returns based on portfolio weights and asset returns.
        
        Args:
            weights: DataFrame with portfolio weights
            returns: DataFrame with asset returns
        
        Returns:
            Series with strategy returns
        """
        logger.info("Calculating strategy returns...")
        
        # Align weights and returns
        common_dates = weights.index.intersection(returns.index)
        weights_aligned = weights.loc[common_dates]
        returns_aligned = returns.loc[common_dates]
        
        # Calculate strategy returns
        strategy_returns = (weights_aligned * returns_aligned).sum(axis=1)
        
        # Apply transaction costs (simplified)
        # Calculate weight changes
        weight_changes = weights_aligned.diff().abs().sum(axis=1)
        transaction_costs = weight_changes * self.transaction_cost
        
        # Net returns after transaction costs
        net_returns = strategy_returns - transaction_costs
        
        logger.info(f"Calculated strategy returns for {len(net_returns)} periods")
        
        return net_returns
    
    def run_strategy(self, returns: pd.DataFrame) -> Dict:
        """
        Run the complete TSMOM strategy.
        
        Args:
            returns: DataFrame with asset returns
        
        Returns:
            Dictionary with strategy results
        """
        logger.info("Running TSMOM strategy...")
        
        # Generate portfolio weights
        weights = self.generate_portfolio_weights(returns)
        
        # Calculate strategy returns
        strategy_returns = self.calculate_strategy_returns(weights, returns)
        
        # Calculate performance metrics
        performance_metrics = self.calculate_performance_metrics(strategy_returns)
        
        results = {
            'weights': weights,
            'returns': strategy_returns,
            'performance_metrics': performance_metrics
        }
        
        logger.info("Strategy execution completed")
        
        return results
    

    
    def calculate_performance_metrics(self, returns: pd.Series) -> Dict:
        """
        Calculate key performance metrics for the strategy.
        
        Args:
            returns: Series with strategy returns
        
        Returns:
            Dictionary with performance metrics
        """
        # Remove NaN values
        returns_clean = returns.dropna()
        
        if len(returns_clean) == 0:
            return {}
        
        # Basic metrics
        total_return = (1 + returns_clean).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns_clean)) - 1
        volatility = returns_clean.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + returns_clean).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Additional metrics
        positive_returns = returns_clean[returns_clean > 0]
        negative_returns = returns_clean[returns_clean < 0]
        
        win_rate = len(positive_returns) / len(returns_clean) if len(returns_clean) > 0 else 0
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
        profit_factor = abs(positive_returns.sum() / negative_returns.sum()) if negative_returns.sum() != 0 else float('inf')
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'num_periods': len(returns_clean)
        }
        
        return metrics


def main():
    """Test the TSMOM strategy."""
    import sys
    sys.path.append('src')
    
    from data.data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    prices = loader.load_processed_data()
    returns = loader.calculate_returns(prices, 'D')
    
    # Run strategy
    strategy = TSMOMStrategy()
    results = strategy.run_strategy(returns)
    
    # Print results
    print("\nTSMOM Strategy Results:")
    print("=" * 50)
    for metric, value in results['performance_metrics'].items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")


if __name__ == "__main__":
    main() 