"""
Base data loader module for TSMOM backtest.
Contains abstract base class and common data processing methods.
"""

import pandas as pd
import numpy as np
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