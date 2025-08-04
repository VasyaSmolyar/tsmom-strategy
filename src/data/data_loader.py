"""
Data loader module for TSMOM backtest.
Handles downloading, cleaning, and processing of financial data.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Optional
import logging
from pathlib import Path
import yaml
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DataLoader(ABC):
    """Abstract base class for data loaders."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize DataLoader with configuration."""
        self.config = self._load_config(config_path)
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def get_asset_universe(self) -> List[str]:
        """Get the complete list of assets from configuration."""
        assets = []
        asset_config = self.config['assets']
        
        for category in asset_config.values():
            if isinstance(category, list):
                assets.extend(category)
        
        return assets
    
    @abstractmethod
    def download_data(self, symbols: Optional[List[str]] = None, 
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Download historical price data for given symbols.
        
        Args:
            symbols: List of symbols to download. If None, uses config assets.
            start_date: Start date for data download. If None, uses config.
            end_date: End date for data download. If None, uses config.
        
        Returns:
            DataFrame with OHLCV data for all symbols
        """
        pass
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the downloaded data.
        
        Args:
            data: Raw price data DataFrame
        
        Returns:
            Cleaned price data DataFrame
        """
        logger.info("Cleaning data...")
        logger.info(f"Initial data shape: {data.shape}")
        logger.info(f"Initial assets: {list(data.columns)}")
        
        # Remove columns with too many missing values
        missing_threshold = 0.5  # 50% missing data threshold (increased from 10%)
        missing_pct = data.isnull().sum() / len(data)
        valid_columns = missing_pct[missing_pct < missing_threshold].index
        data = data[valid_columns]
        
        logger.info(f"After missing data filter: {data.shape}")
        logger.info(f"Assets after missing data filter: {list(data.columns)}")
        
        # Forward fill missing values (up to 10 consecutive days)
        data = data.ffill(limit=10)
        
        # Backward fill remaining missing values at the beginning
        data = data.bfill(limit=5)
        
        # Remove rows with any remaining missing values
        data = data.dropna()
        
        logger.info(f"After forward/backward fill: {data.shape}")
        
        # Remove assets with insufficient data
        min_obs = 252 * 1  # At least 1 year of data (reduced from 2 years)
        valid_assets = data.columns[data.count() >= min_obs]
        data = data[valid_assets]
        
        logger.info(f"Cleaned data shape: {data.shape}")
        logger.info(f"Assets after cleaning: {list(data.columns)}")
        
        # Save cleaned data
        cleaned_file = self.processed_dir / "cleaned_prices.csv"
        data.to_csv(cleaned_file)
        logger.info(f"Saved cleaned data to {cleaned_file}")
        
        return data
    
    def calculate_returns(self, prices: pd.DataFrame, 
                         frequency: str = 'D') -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Args:
            prices: Price DataFrame
            frequency: Return frequency ('D' for daily, 'M' for monthly)
        
        Returns:
            Returns DataFrame
        """
        if frequency == 'D':
            returns = prices.pct_change().dropna()
        elif frequency == 'M':
            # Resample to monthly and calculate returns
            monthly_prices = prices.resample('ME').last()
            returns = monthly_prices.pct_change().dropna()
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")
        
        return returns
    
    def load_processed_data(self) -> pd.DataFrame:
        """Load previously processed data if available."""
        processed_file = self.processed_dir / "cleaned_prices.csv"
        
        if processed_file.exists():
            logger.info(f"Loading processed data from {processed_file}")
            return pd.read_csv(processed_file, index_col=0, parse_dates=True)
        else:
            logger.info("No processed data found. Downloading and processing...")
            raw_data = self.download_data()
            return self.clean_data(raw_data)


class YahooLoader(DataLoader):
    """Yahoo Finance data loader implementation."""
    
    def download_data(self, symbols: Optional[List[str]] = None, 
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Download historical price data for given symbols using Yahoo Finance.
        
        Args:
            symbols: List of symbols to download. If None, uses config assets.
            start_date: Start date for data download. If None, uses config.
            end_date: End date for data download. If None, uses config.
        
        Returns:
            DataFrame with OHLCV data for all symbols
        """
        if symbols is None:
            symbols = self.get_asset_universe()
        
        if start_date is None:
            start_date = self.config['data']['start_date']
        
        if end_date is None:
            end_date = self.config['data']['end_date']
        
        logger.info(f"Downloading data from Yahoo Finance for {len(symbols)} symbols from {start_date} to {end_date}")
        
        # Download data using yfinance
        data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                ticker_data = ticker.history(start=start_date, end=end_date, interval='1d')
                
                if not ticker_data.empty:
                    # Convert index to date only (remove time component)
                    ticker_data.index = ticker_data.index.date
                    data[symbol] = ticker_data['Close']
                    logger.info(f"Downloaded data for {symbol}: {len(ticker_data)} observations")
                else:
                    logger.warning(f"No data found for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error downloading data for {symbol}: {e}")
        
        # Combine all data into a single DataFrame
        if data:
            combined_data = pd.DataFrame(data)
            combined_data.index.name = 'Date'
            
            # Convert index to datetime if it's not already
            if not isinstance(combined_data.index, pd.DatetimeIndex):
                combined_data.index = pd.to_datetime(combined_data.index)
            
            # Save raw data
            raw_file = self.raw_dir / "raw_prices_yahoo.csv"
            combined_data.to_csv(raw_file)
            logger.info(f"Saved raw data to {raw_file}")
            
            return combined_data
        else:
            raise ValueError("No data was successfully downloaded")


class TInvestLoader(DataLoader):
    """T-Invest API data loader implementation (placeholder)."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize T-Invest loader with configuration."""
        super().__init__(config_path)
        # TODO: Add T-Invest API credentials and initialization
        self.api_token = None  # Placeholder for API token
        self.base_url = "https://api.tinvest.ru"  # Placeholder URL
        
    def download_data(self, symbols: Optional[List[str]] = None, 
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Download historical price data for given symbols using T-Invest API.
        
        Args:
            symbols: List of symbols to download. If None, uses config assets.
            start_date: Start date for data download. If None, uses config.
            end_date: End date for data download. If None, uses config.
        
        Returns:
            DataFrame with OHLCV data for all symbols
        """
        if symbols is None:
            symbols = self.get_asset_universe()
        
        if start_date is None:
            start_date = self.config['data']['start_date']
        
        if end_date is None:
            end_date = self.config['data']['end_date']
        
        logger.info(f"Downloading data from T-Invest API for {len(symbols)} symbols from {start_date} to {end_date}")
        
        # TODO: Implement actual T-Invest API calls
        # This is a placeholder implementation that returns empty DataFrame
        logger.warning("T-Invest API integration not yet implemented. Returning empty DataFrame.")
        
        # Create empty DataFrame with proper structure
        empty_data = pd.DataFrame(columns=symbols, index=pd.date_range(start=start_date, end=end_date))
        empty_data.index.name = 'Date'
        
        # Save raw data
        raw_file = self.raw_dir / "raw_prices_tinvest.csv"
        empty_data.to_csv(raw_file)
        logger.info(f"Saved placeholder data to {raw_file}")
        
        return empty_data


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
    elif data_source == 'T-Invest':
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