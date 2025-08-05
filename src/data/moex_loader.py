"""
MOEX API data loader implementation for TSMOM backtest.
Handles downloading historical price data from MOEX API.
"""

import pandas as pd
import requests
import json
from typing import List, Optional, Dict
import logging
from pathlib import Path
from datetime import datetime, timedelta
from .base_loader import DataLoader

logger = logging.getLogger(__name__)


class MoexLoader(DataLoader):
    """MOEX API data loader implementation."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize MOEX loader with configuration."""
        super().__init__(config_path)
        self.base_url = "https://iss.moex.com/iss"
        
    def _get_futures_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Download futures data from MOEX API.
        
        Args:
            symbol: Futures symbol (e.g., 'BRF6')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # MOEX ISS API endpoint for futures (add .json to get JSON response)
            url = f"{self.base_url}/engines/futures/markets/forts/securities/{symbol}/candles.json"
            
            params = {
                'from': start_date,
                'till': end_date,
                'interval': 24,  # Daily candles
                'iss.meta': 'off',
                'iss.only': 'data'
            }
            
            logger.info(f"Requesting data for {symbol} from {start_date} to {end_date}")
            logger.info(f"URL: {url}")
            logger.info(f"Params: {params}")
            
            response = requests.get(url, params=params, timeout=30)
            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response headers: {dict(response.headers)}")
            
            response.raise_for_status()
            
            # Log response content for debugging
            logger.info(f"Response content length: {len(response.text)}")
            logger.info(f"Response content preview: {response.text[:200]}")
            
            data = response.json()
            
            if 'candles' not in data or not data['candles'].get('data'):
                logger.warning(f"No data received for {symbol}")
                return pd.DataFrame()
            
            # Parse the response data
            # MOEX API returns data in format: [open, close, high, low, value, volume, begin, end]
            raw_data = data['candles']['data']
            df = pd.DataFrame(raw_data, columns=['open', 'close', 'high', 'low', 'value', 'volume', 'begin', 'end'])
            
            # Convert begin date to datetime and set as index
            df['date'] = pd.to_datetime(df['begin'])
            df.set_index('date', inplace=True)
            
            # Convert to numeric
            for col in ['open', 'close', 'high', 'low', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate typical price for TSMOM strategy
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            
            logger.info(f"Downloaded {len(df)} records for {symbol}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading data for {symbol}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error for {symbol}: {e}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def test_cl_futures_download(self) -> pd.DataFrame:
        """
        Test download of BRF6 futures data (Brent oil futures).
        
        Returns:
            DataFrame with BRF6 futures data
        """
        symbol = "BRF6"  # Brent oil futures
        start_date = "2025-01-01"
        end_date = "2025-01-31"
        
        logger.info(f"Testing download of {symbol} futures data...")
        
        data = self._get_futures_data(symbol, start_date, end_date)
        
        if not data.empty:
            # Save test data
            test_file = self.raw_dir / f"test_{symbol}.csv"
            data.to_csv(test_file)
            logger.info(f"Saved test data to {test_file}")
            
            # Print summary statistics
            logger.info(f"Test data summary for {symbol}:")
            logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
            logger.info(f"Records: {len(data)}")
            logger.info(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
            logger.info(f"Average volume: {data['volume'].mean():.0f}")
            
        return data
    
    def download_data(self, symbols: Optional[List[str]] = None, 
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Download historical price data for given symbols using MOEX API.
        
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
        
        logger.info(f"Downloading data from MOEX API for {len(symbols)} symbols from {start_date} to {end_date}")
        
        all_data = {}
        
        for symbol in symbols:
            logger.info(f"Downloading data for {symbol}...")
            
            # Check if it's a futures symbol (contains '-')
            if '-' in symbol:
                data = self._get_futures_data(symbol, start_date, end_date)
            else:
                # For other symbols, use placeholder for now
                logger.warning(f"Non-futures symbol {symbol} not yet implemented")
                data = pd.DataFrame()
            
            if not data.empty:
                # Use typical price for TSMOM strategy
                all_data[symbol] = data['typical_price'] if 'typical_price' in data.columns else data['close']
        
        if all_data:
            result_df = pd.DataFrame(all_data)
            result_df.index.name = 'Date'
            
            # Save raw data
            raw_file = self.raw_dir / "raw_prices_moex.csv"
            result_df.to_csv(raw_file)
            logger.info(f"Saved data to {raw_file}")
            
            return result_df
        else:
            logger.warning("No data downloaded. Returning empty DataFrame.")
            empty_data = pd.DataFrame(columns=symbols, index=pd.date_range(start=start_date, end=end_date))
            empty_data.index.name = 'Date'
            return empty_data 