"""
Helper utilities for TSMOM backtest project.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import yaml
import json

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        raise


def save_results(results: Dict, output_path: str) -> None:
    """Save results to JSON file."""
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, default=str, indent=2)
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving results to {output_path}: {e}")
        raise


def calculate_rolling_metrics(returns: pd.Series, window: int = 252) -> pd.DataFrame:
    """
    Calculate rolling performance metrics.
    
    Args:
        returns: Returns series
        window: Rolling window size
    
    Returns:
        DataFrame with rolling metrics
    """
    metrics = pd.DataFrame(index=returns.index)
    
    # Rolling return
    metrics['rolling_return'] = returns.rolling(window=window).mean() * 252
    
    # Rolling volatility
    metrics['rolling_volatility'] = returns.rolling(window=window).std() * np.sqrt(252)
    
    # Rolling Sharpe ratio
    metrics['rolling_sharpe'] = (metrics['rolling_return'] / metrics['rolling_volatility'])
    
    # Rolling drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.rolling(window=window).max()
    metrics['rolling_drawdown'] = (cumulative - rolling_max) / rolling_max
    
    return metrics


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format float as percentage string."""
    return f"{value:.{decimals}%}"


def format_number(value: float, decimals: int = 4) -> str:
    """Format float with specified decimals."""
    return f"{value:.{decimals}f}"


def create_summary_table(metrics: Dict) -> pd.DataFrame:
    """Create a formatted summary table from metrics."""
    summary_data = []
    
    for metric, value in metrics.items():
        if isinstance(value, float):
            if 'return' in metric.lower() or 'drawdown' in metric.lower():
                formatted_value = format_percentage(value)
            else:
                formatted_value = format_number(value)
        else:
            formatted_value = str(value)
        
        summary_data.append({
            'Metric': metric.replace('_', ' ').title(),
            'Value': formatted_value,
            'Raw_Value': value
        })
    
    return pd.DataFrame(summary_data)


def validate_data(data: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate data quality.
    
    Args:
        data: DataFrame to validate
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check for missing values
    missing_pct = data.isnull().sum() / len(data)
    high_missing = missing_pct[missing_pct > 0.1]
    if len(high_missing) > 0:
        issues.append(f"High missing values in columns: {list(high_missing.index)}")
    
    # Check for infinite values
    inf_count = np.isinf(data).sum().sum()
    if inf_count > 0:
        issues.append(f"Found {inf_count} infinite values")
    
    # Check for zero variance columns
    zero_var_cols = data.columns[data.var() == 0]
    if len(zero_var_cols) > 0:
        issues.append(f"Zero variance columns: {list(zero_var_cols)}")
    
    # Check data range
    for col in data.columns:
        if data[col].max() > 1000 or data[col].min() < -1000:
            issues.append(f"Unusual values in {col}: min={data[col].min():.2f}, max={data[col].max():.2f}")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def calculate_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation matrix for returns."""
    return returns.corr()


def calculate_rolling_correlation(returns: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Calculate rolling correlation matrix."""
    # This is a simplified version - in practice you might want more sophisticated
    # rolling correlation calculation
    correlations = {}
    
    for i in range(len(returns) - window + 1):
        window_data = returns.iloc[i:i+window]
        corr_matrix = window_data.corr()
        correlations[returns.index[i + window - 1]] = corr_matrix
    
    return pd.concat(correlations, axis=0)


def detect_regime_changes(returns: pd.Series, window: int = 252) -> List[Tuple]:
    """
    Detect potential regime changes in returns.
    
    Args:
        returns: Returns series
        window: Window for regime detection
    
    Returns:
        List of (date, regime_type) tuples
    """
    regime_changes = []
    
    # Calculate rolling volatility
    rolling_vol = returns.rolling(window=window).std()
    
    # Detect high volatility periods
    vol_threshold = rolling_vol.quantile(0.9)
    high_vol_periods = rolling_vol > vol_threshold
    
    # Detect regime changes
    regime_changes_raw = high_vol_periods.diff().fillna(False)
    change_dates = returns.index[regime_changes_raw]
    
    for date in change_dates:
        if high_vol_periods.loc[date]:
            regime_changes.append((date, 'high_volatility'))
        else:
            regime_changes.append((date, 'normal_volatility'))
    
    return regime_changes


def calculate_risk_metrics(returns: pd.Series) -> Dict:
    """
    Calculate comprehensive risk metrics.
    
    Args:
        returns: Returns series
    
    Returns:
        Dictionary with risk metrics
    """
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return {}
    
    # Basic risk metrics
    volatility = returns_clean.std() * np.sqrt(252)
    
    # Value at Risk
    var_95 = np.percentile(returns_clean, 5)
    var_99 = np.percentile(returns_clean, 1)
    
    # Conditional Value at Risk
    cvar_95 = returns_clean[returns_clean <= var_95].mean()
    cvar_99 = returns_clean[returns_clean <= var_99].mean()
    
    # Downside deviation
    downside_returns = returns_clean[returns_clean < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    
    # Maximum drawdown
    cumulative = (1 + returns_clean).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Skewness and kurtosis
    skewness = returns_clean.skew()
    kurtosis = returns_clean.kurtosis()
    
    risk_metrics = {
        'volatility': volatility,
        'var_95': var_95,
        'var_99': var_99,
        'cvar_95': cvar_95,
        'cvar_99': cvar_99,
        'downside_deviation': downside_deviation,
        'max_drawdown': max_drawdown,
        'skewness': skewness,
        'kurtosis': kurtosis
    }
    
    return risk_metrics


def create_asset_allocation_summary(weights: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary of asset allocation over time.
    
    Args:
        weights: Portfolio weights DataFrame
    
    Returns:
        Summary DataFrame
    """
    summary = pd.DataFrame()
    
    # Average weights
    summary['avg_weight'] = weights.mean()
    
    # Maximum weights
    summary['max_weight'] = weights.max()
    
    # Minimum weights
    summary['min_weight'] = weights.min()
    
    # Weight volatility
    summary['weight_volatility'] = weights.std()
    
    # Frequency of positive weights
    summary['positive_freq'] = (weights > 0).mean()
    
    # Frequency of negative weights
    summary['negative_freq'] = (weights < 0).mean()
    
    return summary


def print_project_info():
    """Print project information."""
    print("=" * 60)
    print("TSMOM (Time Series Momentum) Backtest Project")
    print("=" * 60)
    print("Based on Moskowitz, Ooi, and Pedersen (2012)")
    print("Strategy: Long positive momentum, short negative momentum")
    print("Lookback Period: 12 months")
    print("Holding Period: 1 month")
    print("Target Volatility: 40% annual")
    print("Rebalancing: Monthly")
    print("=" * 60)


if __name__ == "__main__":
    print_project_info() 