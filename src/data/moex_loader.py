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
    
    def get_available_futures(self, asset_type: Optional[str] = None) -> pd.DataFrame:
        """
        Get available futures tickers with trading start and expiration dates.
        
        Args:
            asset_type: Optional filter for asset type (e.g., 'BR' for Brent oil)
        
        Returns:
            DataFrame with futures information including:
            - ticker: Futures ticker
            - asset_name: Asset name
            - trading_start: Trading start date
            - expiration_date: Expiration date
            - status: Contract status
        """
        try:
            # MOEX ISS API endpoint for futures securities
            url = f"{self.base_url}/engines/futures/markets/forts/securities.json"
            
            params = {
                'iss.meta': 'off',
                'iss.only': 'data'
            }
            
            logger.info("Requesting available futures from MOEX API...")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'securities' not in data or not data['securities'].get('data'):
                logger.warning("No futures data received from MOEX API")
                return pd.DataFrame()
            
            # Parse the response data
            # MOEX API returns data with 25 columns
            raw_data = data['securities']['data']
            columns = data['securities']['columns']
            
            # Create DataFrame with futures information
            futures_df = pd.DataFrame(raw_data, columns=columns)
            
            # Rename key columns for easier access
            column_mapping = {
                'SECID': 'ticker',
                'SHORTNAME': 'short_name',
                'SECNAME': 'name',
                'LATNAME': 'lat_name',
                'ASSETCODE': 'asset_code',
                'LASTTRADEDATE': 'last_trade_date',
                'LASTDELDATE': 'expiration_date',
                'SECTYPE': 'sec_type',
                'LOTVOLUME': 'lot_volume',
                'MINSTEP': 'min_step',
                'STEPPRICE': 'step_price',
                'PREVSETTLEPRICE': 'prev_settle_price',
                'LASTSETTLEPRICE': 'last_settle_price',
                'PREVPRICE': 'prev_price',
                'IMTIME': 'update_time'
            }
            
            # Rename columns that exist
            for old_col, new_col in column_mapping.items():
                if old_col in futures_df.columns:
                    futures_df = futures_df.rename(columns={old_col: new_col})
            
            # Filter by asset type if specified
            if asset_type:
                futures_df = futures_df[futures_df['ticker'].str.contains(asset_type, na=False)]
                logger.info(f"Filtered to {len(futures_df)} {asset_type} futures")
            
            # Convert date columns to datetime
            date_columns = ['last_trade_date', 'expiration_date', 'update_time']
            for col in date_columns:
                if col in futures_df.columns:
                    futures_df[col] = pd.to_datetime(futures_df[col], errors='coerce')
            
            # Add current date for filtering
            current_date = pd.Timestamp.now()
            
            # Add status information
            if 'expiration_date' in futures_df.columns:
                futures_df['is_active'] = futures_df['expiration_date'] >= current_date
                
                # Add days to expiration
                futures_df['days_to_expiration'] = (
                    futures_df['expiration_date'] - current_date
                ).dt.days
            else:
                futures_df['is_active'] = False
                futures_df['days_to_expiration'] = None
            
            # Sort by expiration date
            if 'expiration_date' in futures_df.columns:
                futures_df = futures_df.sort_values('expiration_date')
            
            logger.info(f"Retrieved {len(futures_df)} futures contracts")
            
            return futures_df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error requesting futures data: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def get_futures_by_type(self, asset_type: str) -> pd.DataFrame:
        """
        Get futures contracts for specific asset type.
        
        Args:
            asset_type: Asset type (e.g., 'BR' for Brent oil, 'SI' for Silver, 'GD' for Gold)
        
        Returns:
            DataFrame with futures information for the specified asset type
        """
        all_futures = self.get_available_futures()
        
        if all_futures.empty:
            return pd.DataFrame()
        
        # Filter by asset type
        filtered_futures = all_futures[all_futures['ticker'].str.contains(asset_type, na=False)]
        
        if filtered_futures.empty:
            logger.warning(f"No futures found for asset type: {asset_type}")
            return pd.DataFrame()
        
        logger.info(f"Found {len(filtered_futures)} futures for {asset_type}")
        return filtered_futures
    
    def get_active_futures(self, asset_type: Optional[str] = None) -> pd.DataFrame:
        """
        Get currently active futures contracts.
        
        Args:
            asset_type: Optional filter for asset type
        
        Returns:
            DataFrame with active futures information
        """
        futures_df = self.get_available_futures(asset_type)
        
        if futures_df.empty:
            return pd.DataFrame()
        
        # Filter active contracts
        active_futures = futures_df[futures_df['is_active'] == True]
        
        logger.info(f"Found {len(active_futures)} active futures contracts")
        return active_futures
    
    def get_futures_with_data(self, start_date: str, end_date: str, asset_type: Optional[str] = None) -> pd.DataFrame:
        """
        Get futures contracts that have trading data in the specified period.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            asset_type: Optional filter for asset type
        
        Returns:
            DataFrame with futures that have data in the specified period
        """
        futures_df = self.get_available_futures(asset_type)
        
        if futures_df.empty:
            return pd.DataFrame()
        
        # Filter futures that were active during the period
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        period_futures = futures_df[
            (futures_df['last_trade_date'] <= end_dt) & 
            (futures_df['expiration_date'] >= start_dt)
        ]
        
        logger.info(f"Found {len(period_futures)} futures active during {start_date} to {end_date}")
        return period_futures
    
    def get_historical_futures_for_period(self, start_date: str, end_date: str, asset_type: Optional[str] = None) -> pd.DataFrame:
        """
        Get all futures contracts that could have been traded during the specified period.
        This includes both active and expired contracts that were available during the period.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            asset_type: Optional filter for asset type
        
        Returns:
            DataFrame with all futures that could have been traded during the period
        """
        futures_df = self.get_available_futures(asset_type)
        
        if futures_df.empty:
            return pd.DataFrame()
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # More inclusive filter for historical data
        # Include contracts that:
        # 1. Started trading before or during the period AND
        # 2. Expired after or during the period
        # This captures contracts that were available for trading during the period
        
        # If we have trading_start date, use it for more accurate filtering
        if 'trading_start' in futures_df.columns:
            # Convert trading_start to datetime if it's not already
            futures_df['trading_start'] = pd.to_datetime(futures_df['trading_start'], errors='coerce')
            
            historical_futures = futures_df[
                (futures_df['trading_start'] <= end_dt) & 
                (futures_df['expiration_date'] >= start_dt)
            ]
        else:
            # Fallback to using last_trade_date if trading_start is not available
            historical_futures = futures_df[
                (futures_df['last_trade_date'] <= end_dt) & 
                (futures_df['expiration_date'] >= start_dt)
            ]
        
        # Add period information
        historical_futures = historical_futures.copy()
        historical_futures['period_start'] = start_date
        historical_futures['period_end'] = end_date
        
        # Mark if contract was active during the period
        current_date = pd.Timestamp.now()
        historical_futures['was_active_in_period'] = (
            (historical_futures['expiration_date'] >= start_dt) & 
            (historical_futures['expiration_date'] <= end_dt)
        )
        
        logger.info(f"Found {len(historical_futures)} historical futures for period {start_date} to {end_date}")
        return historical_futures 