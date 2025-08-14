"""
Data loader module for TSMOM backtest.
Factory functions and main execution logic.
"""

import logging
import yaml
from .base_loader import DataLoader
from .yahoo_futures_loader import YahooFuturesLoader
from .yahoo_crypto_loader import YahooCryptoLoader
from .moex_loader import MoexLoader


def create_data_loader(config_path: str = "config/config.yaml", data_source: str = "YahooFutures") -> DataLoader:
    """
    Factory function to create appropriate data loader based on data source parameter.
    
    Args:
        config_path: Path to configuration file
        data_source: Data source to use ("YahooFutures", "YahooCrypto" or "MOEX")
    
    Returns:
        Appropriate DataLoader instance
    """
    if data_source == 'YahooFutures':
        return YahooFuturesLoader(config_path)
    elif data_source == 'YahooCrypto':
        return YahooCryptoLoader(config_path)
    elif data_source == 'MOEX':
        return MoexLoader(config_path)
    else:
        raise ValueError(f"Unknown data source: {data_source}. Supported sources: YahooFutures, YahooCrypto, MOEX")


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