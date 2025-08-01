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

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data downloading and processing for TSMOM strategy."""
    
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
        if symbols is None:
            symbols = self.get_asset_universe()
        
        if start_date is None:
            start_date = self.config['data']['start_date']
        
        if end_date is None:
            end_date = self.config['data']['end_date']
        
        logger.info(f"Downloading data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        # Download data using yfinance
        data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                ticker_data = ticker.history(start=start_date, end=end_date)
                
                if not ticker_data.empty:
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
            
            # Save raw data
            raw_file = self.raw_dir / "raw_prices.csv"
            combined_data.to_csv(raw_file)
            logger.info(f"Saved raw data to {raw_file}")
            
            return combined_data
        else:
            raise ValueError("No data was successfully downloaded")
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the downloaded data.
        
        Args:
            data: Raw price data DataFrame
        
        Returns:
            Cleaned price data DataFrame
        """
        logger.info("Cleaning data...")
        
        # Remove columns with too many missing values
        missing_threshold = 0.1  # 10% missing data threshold
        missing_pct = data.isnull().sum() / len(data)
        valid_columns = missing_pct[missing_pct < missing_threshold].index
        data = data[valid_columns]
        
        # Forward fill missing values (up to 5 consecutive days)
        data = data.fillna(method='ffill', limit=5)
        
        # Remove rows with any remaining missing values
        data = data.dropna()
        
        # Remove assets with insufficient data
        min_obs = 252 * 2  # At least 2 years of data
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
            monthly_prices = prices.resample('M').last()
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


def main():
    """Main function for data loading and processing."""
    logging.basicConfig(level=logging.INFO)
    
    loader = DataLoader()
    
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