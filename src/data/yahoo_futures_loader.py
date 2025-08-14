"""
Yahoo Finance futures data loader implementation for TSMOM backtest.
Handles downloading historical futures price data from Yahoo Finance.
"""

import pandas as pd
import yfinance as yf
from typing import List, Optional
import logging
from pathlib import Path
from .base_loader import DataLoader

logger = logging.getLogger(__name__)


class YahooFuturesLoader(DataLoader):
    """Yahoo Finance futures data loader implementation."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize Yahoo futures loader with configuration."""
        super().__init__(config_path)
        self._source_suffix = 'yahoo_futures'
    
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
            symbols = self.get_yahoo_futures_asset_universe()
        
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
            raw_file = self.raw_dir / "raw_prices_yahoo_futures.csv"
            combined_data.to_csv(raw_file)
            logger.info(f"Saved raw data to {raw_file}")
            
            return combined_data
        else:
            raise ValueError("No data was successfully downloaded")
    
    def get_yahoo_futures_asset_universe(self) -> List[str]:
        """Get the asset universe suitable for Yahoo Finance futures (exclude Russian and crypto assets)."""
        assets = []
        asset_config = self.config['assets']
        
        # Only include non-Russian, non-crypto assets for Yahoo Futures loader
        for category_name, category in asset_config.items():
            if isinstance(category, list):
                # Skip Russian assets and crypto for Yahoo Futures loader
                if category_name in ['russian_equities', 'cryptocurrencies']:
                    continue
                # Filter out MOEX assets from other categories
                filtered_assets = [asset for asset in category if not asset.endswith('.ME')]
                assets.extend(filtered_assets)
        
        logger.info(f"Yahoo Futures asset universe: {len(assets)} assets - {assets}")
        return assets 