"""
Data loader module for TSMOM backtest.
Factory functions and main execution logic.
"""

import logging
import yaml
from .base_loader import DataLoader
from .yahoo_loader import YahooLoader
from .tinvest_loader import TInvestLoader


def create_data_loader(config_path: str = "config/config.yaml") -> DataLoader:
    """
    Factory function to create appropriate data loader based on configuration.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Appropriate DataLoader instance
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    data_source = config['data'].get('source', 'Yahoo')
    
    if data_source == 'Yahoo':
        return YahooLoader(config_path)
    elif data_source == 'T-Invest' or data_source == 'Tinkoff':
        return TInvestLoader(config_path)
    else:
        raise ValueError(f"Unsupported data source: {data_source}")


def main():
    """Main function for data loading and processing."""
    logging.basicConfig(level=logging.INFO)
    
    # Use factory function to create appropriate loader
    loader = create_data_loader()
    
    # Download and process data
    prices = loader.load_processed_data()
    
    # Calculate returns
    daily_returns = loader.calculate_returns(prices, 'D')
    monthly_returns = loader.calculate_returns(prices, 'M')
    
    print(f"Data shape: {prices.shape}")
    print(f"Daily returns shape: {daily_returns.shape}")
    print(f"Monthly returns shape: {monthly_returns.shape}")
    print(f"Assets: {list(prices.columns)}")


if __name__ == "__main__":
    main() 