"""
Tests for TSMOM strategy implementation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from strategy.tsmom_strategy import TSMOMStrategy
from data.data_loader import DataLoader
from analysis.performance_analyzer import PerformanceAnalyzer


class TestTSMOMStrategy:
    """Test cases for TSMOM strategy."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data for testing."""
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        np.random.seed(42)
        
        # Create sample returns for 5 assets
        returns_data = {}
        for i in range(5):
            asset_name = f'Asset_{i}'
            # Generate returns with some momentum
            returns = np.random.normal(0.0005, 0.02, len(dates))
            # Add some momentum effect
            for j in range(252, len(returns)):
                if np.random.random() > 0.5:
                    returns[j] += returns[j-252] * 0.1
            returns_data[asset_name] = returns
        
        return pd.DataFrame(returns_data, index=dates)
    
    @pytest.fixture
    def strategy(self):
        """Create TSMOM strategy instance."""
        return TSMOMStrategy()
    
    def test_strategy_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy.lookback_period == 12
        assert strategy.holding_period == 1
        assert strategy.target_volatility == 0.40
        assert strategy.transaction_cost == 0.001
    
    def test_momentum_signals(self, strategy, sample_returns):
        """Test momentum signal calculation."""
        signals = strategy.calculate_momentum_signals(sample_returns)
        
        # Check signal values are valid
        assert signals.isin([-1, 0, 1]).all().all()
        
        # Check signals have expected shape
        assert signals.shape == sample_returns.shape
        
        # Check signals are not all NaN
        assert not signals.isna().all().all()
    
    def test_volatility_calculation(self, strategy, sample_returns):
        """Test volatility calculation."""
        volatility = strategy.calculate_volatility(sample_returns)
        
        # Check volatility is positive
        assert (volatility > 0).all().all()
        
        # Check volatility has expected shape
        assert volatility.shape == sample_returns.shape
        
        # Check volatility is reasonable (annualized)
        assert (volatility < 1.0).all().all()  # Should be less than 100%
    
    def test_position_sizing(self, strategy, sample_returns):
        """Test position sizing calculation."""
        signals = strategy.calculate_momentum_signals(sample_returns)
        volatility = strategy.calculate_volatility(sample_returns)
        
        position_sizes = strategy.calculate_position_sizes(signals, volatility)
        
        # Check position sizes are within bounds
        assert (position_sizes >= -strategy.max_position_size).all().all()
        assert (position_sizes <= strategy.max_position_size).all().all()
        
        # Check position sizes have expected shape
        assert position_sizes.shape == sample_returns.shape
    
    def test_portfolio_weights(self, strategy, sample_returns):
        """Test portfolio weight generation."""
        weights = strategy.generate_portfolio_weights(sample_returns)
        
        # Check weights have expected shape
        assert weights.shape == sample_returns.shape
        
        # Check weights are within bounds
        assert (weights >= -strategy.max_position_size).all().all()
        assert (weights <= strategy.max_position_size).all().all()
    
    def test_strategy_returns(self, strategy, sample_returns):
        """Test strategy returns calculation."""
        weights = strategy.generate_portfolio_weights(sample_returns)
        strategy_returns = strategy.calculate_strategy_returns(weights, sample_returns)
        
        # Check returns have expected length
        assert len(strategy_returns) > 0
        
        # Check returns are not all NaN
        assert not strategy_returns.isna().all()
    
    def test_performance_metrics(self, strategy, sample_returns):
        """Test performance metrics calculation."""
        weights = strategy.generate_portfolio_weights(sample_returns)
        strategy_returns = strategy.calculate_strategy_returns(weights, sample_returns)
        
        metrics = strategy.calculate_performance_metrics(strategy_returns)
        
        # Check required metrics are present
        required_metrics = ['total_return', 'annual_return', 'volatility', 'sharpe_ratio']
        for metric in required_metrics:
            assert metric in metrics
        
        # Check metrics are reasonable
        assert -1 <= metrics['total_return'] <= 10  # Reasonable range
        assert 0 <= metrics['volatility'] <= 1  # Annualized volatility
        assert metrics['num_periods'] > 0
    
    def test_full_strategy_run(self, strategy, sample_returns):
        """Test complete strategy execution."""
        # Create mock prices (starting at 100 for each asset)
        sample_prices = pd.DataFrame(
            100 * (1 + sample_returns).cumprod(),
            index=sample_returns.index,
            columns=sample_returns.columns
        )
        results = strategy.run_strategy(sample_returns, sample_prices)
        
        # Check results structure
        assert 'weights' in results
        assert 'returns' in results
        assert 'performance_metrics' in results
        
        # Check data types
        assert isinstance(results['weights'], pd.DataFrame)
        assert isinstance(results['returns'], pd.Series)
        assert isinstance(results['performance_metrics'], dict)
        
        # Check data shapes
        assert results['weights'].shape == sample_returns.shape
        assert len(results['returns']) > 0


class TestDataLoader:
    """Test cases for data loader."""
    
    @pytest.fixture
    def loader(self):
        """Create data loader instance."""
        return DataLoader()
    
    def test_config_loading(self, loader):
        """Test configuration loading."""
        assert 'strategy' in loader.config
        assert 'data' in loader.config
        assert 'assets' in loader.config
    
    def test_asset_universe(self, loader):
        """Test asset universe extraction."""
        assets = loader.get_asset_universe()
        
        # Check we have assets
        assert len(assets) > 0
        
        # Check assets are strings
        assert all(isinstance(asset, str) for asset in assets)
    
    def test_returns_calculation(self, loader):
        """Test returns calculation."""
        # Create sample price data
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        prices = pd.DataFrame({
            'Asset_1': np.cumprod(1 + np.random.normal(0.0005, 0.02, len(dates))),
            'Asset_2': np.cumprod(1 + np.random.normal(0.0003, 0.015, len(dates)))
        }, index=dates)
        
        # Test daily returns
        daily_returns = loader.calculate_returns(prices, 'D')
        assert daily_returns.shape == prices.shape
        assert not daily_returns.isna().all().all()
        
        # Test monthly returns
        monthly_returns = loader.calculate_returns(prices, 'M')
        assert len(monthly_returns) < len(daily_returns)  # Fewer monthly observations


class TestPerformanceAnalyzer:
    """Test cases for performance analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create performance analyzer instance."""
        return PerformanceAnalyzer()
    
    @pytest.fixture
    def sample_strategy_returns(self):
        """Create sample strategy returns."""
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, len(dates))
        return pd.Series(returns, index=dates)
    
    def test_comprehensive_metrics(self, analyzer, sample_strategy_returns):
        """Test comprehensive metrics calculation."""
        metrics = analyzer.calculate_comprehensive_metrics(sample_strategy_returns)
        
        # Check required metrics are present
        required_metrics = ['total_return', 'annual_return', 'volatility', 'sharpe_ratio']
        for metric in required_metrics:
            assert metric in metrics
        
        # Check metrics are reasonable
        assert isinstance(metrics['total_return'], float)
        assert isinstance(metrics['sharpe_ratio'], float)
        assert metrics['num_periods'] > 0
    
    def test_benchmark_comparison(self, analyzer, sample_strategy_returns):
        """Test benchmark comparison."""
        # Create sample benchmark returns
        dates = sample_strategy_returns.index
        np.random.seed(43)
        benchmark_returns = pd.Series(np.random.normal(0.0003, 0.015, len(dates)), index=dates)
        
        metrics = analyzer.calculate_comprehensive_metrics(sample_strategy_returns, benchmark_returns)
        
        # Check benchmark metrics are present
        benchmark_metrics = ['benchmark_total_return', 'benchmark_annual_return', 'excess_return']
        for metric in benchmark_metrics:
            assert metric in metrics
    
    def test_performance_table_creation(self, analyzer, sample_strategy_returns):
        """Test performance table creation."""
        metrics = analyzer.calculate_comprehensive_metrics(sample_strategy_returns)
        table = analyzer.create_performance_table(metrics)
        
        # Check table structure
        assert 'Metric' in table.columns
        assert 'Value' in table.columns
        assert 'Formatted_Value' in table.columns
        
        # Check table has data
        assert len(table) > 0


def test_integration():
    """Integration test for the complete pipeline."""
    # This test would run the complete pipeline
    # For now, we'll just test that modules can be imported
    try:
        from strategy.tsmom_strategy import TSMOMStrategy
        from data.data_loader import DataLoader
        from analysis.performance_analyzer import PerformanceAnalyzer
        
        # Create instances
        strategy = TSMOMStrategy()
        loader = DataLoader()
        analyzer = PerformanceAnalyzer()
        
        assert strategy is not None
        assert loader is not None
        assert analyzer is not None
        
    except ImportError as e:
        pytest.fail(f"Import error: {e}")


if __name__ == "__main__":
    pytest.main([__file__]) 