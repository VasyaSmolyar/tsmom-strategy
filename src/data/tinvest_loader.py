"""
Tinkoff Investments API data loader implementation for TSMOM backtest.
Handles downloading historical price data for Russian futures from Tinkoff API.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import logging
from pathlib import Path
from datetime import datetime, timedelta
import os
from tinkoff.invest import Client, CandleInterval, InstrumentIdType
from tinkoff.invest.services import InstrumentsService, MarketDataService
from .base_loader import DataLoader

logger = logging.getLogger(__name__)


class TInvestLoader(DataLoader):
    """Tinkoff Investments API data loader implementation."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize Tinkoff loader with configuration."""
        super().__init__(config_path)
        self.token = os.getenv('TINKOFF_TOKEN')
        if not self.token:
            raise ValueError("TINKOFF_TOKEN environment variable is required")
        
        # Russian futures mapping to Tinkoff figi codes
        self.futures_mapping = {
            "Si": "FUTSI0624000",      # USD/RUB Futures
            "BR": "FUTBR0624000",      # Brent Oil Futures  
            "RI": "FUTRI0624000",      # RTS Index Futures
            "MX": "FUTMX0624000",      # Moscow Exchange Index Futures
            "GD": "FUTGD0624000",      # Gold Futures
            "SBRF": "FUTSBRF0624000",  # Sberbank Futures
            "GAZR": "FUTGAZR0624000",  # Gazprom Futures
            "LKOH": "FUTLKOH0624000",  # Lukoil Futures
            "NVTK": "FUTNVTK0624000",  # Novatek Futures
            "ROSN": "FUTROSN0624000",  # Rosneft Futures
        }
        
        # Initialize client
        self.client = Client(self.token)
        
    def get_figi_by_ticker(self, ticker: str) -> Optional[str]:
        """Get FIGI code for a given ticker."""
        try:
            with self.client as client:
                instruments: InstrumentsService = client.instruments
                response = instruments.find_instrument(query=ticker)
                
                for instrument in response.instruments:
                    if instrument.ticker == ticker:
                        return instrument.figi
                        
        except Exception as e:
            logger.error(f"Error getting FIGI for ticker {ticker}: {e}")
            
        return None
    
    def get_candles(self, figi: str, from_date: datetime, to_date: datetime) -> pd.DataFrame:
        """Get candles for a specific instrument."""
        try:
            with self.client as client:
                market_data: MarketDataService = client.market_data
                
                # Get candles with daily interval
                response = market_data.get_candles(
                    figi=figi,
                    from_=from_date,
                    to=to_date,
                    interval=CandleInterval.CANDLE_INTERVAL_DAY
                )
                
                if not response.candles:
                    logger.warning(f"No candles found for FIGI {figi}")
                    return pd.DataFrame()
                
                # Convert to DataFrame
                data = []
                for candle in response.candles:
                    data.append({
                        'date': candle.time,
                        'open': candle.open.units + candle.open.nano / 1e9,
                        'high': candle.high.units + candle.high.nano / 1e9,
                        'low': candle.low.units + candle.low.nano / 1e9,
                        'close': candle.close.units + candle.close.nano / 1e9,
                        'volume': candle.volume
                    })
                
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                return df
                
        except Exception as e:
            logger.error(f"Error getting candles for FIGI {figi}: {e}")
            return pd.DataFrame()
    
    def download_data(self, symbols: Optional[List[str]] = None, 
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Download historical price data for Russian futures using Tinkoff API.
        
        Args:
            symbols: List of symbols to download. If None, uses config russian_futures.
            start_date: Start date for data download. If None, uses config.
            end_date: End date for data download. If None, uses config.
        
        Returns:
            DataFrame with OHLCV data for all symbols
        """
        if symbols is None:
            symbols = self.config['assets']['russian_futures']
        
        if start_date is None:
            start_date = self.config['data']['start_date']
        
        if end_date is None:
            end_date = self.config['data']['end_date']
        
        # Convert dates to datetime
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        logger.info(f"Downloading Russian futures data from Tinkoff API for {len(symbols)} symbols from {start_date} to {end_date}")
        
        all_data = {}
        
        for symbol in symbols:
            logger.info(f"Downloading data for {symbol}")
            
            # Try to get FIGI from mapping first
            figi = self.futures_mapping.get(symbol)
            
            # If not in mapping, try to find by ticker
            if not figi:
                figi = self.get_figi_by_ticker(symbol)
            
            if not figi:
                logger.warning(f"Could not find FIGI for symbol {symbol}, skipping")
                continue
            
            # Get candles data
            df = self.get_candles(figi, start_dt, end_dt)
            
            if not df.empty:
                # Create price series (using close price)
                all_data[symbol] = df['close']
                logger.info(f"Downloaded {len(df)} records for {symbol}")
            else:
                logger.warning(f"No data downloaded for {symbol}")
        
        if not all_data:
            logger.error("No data downloaded for any symbols")
            return pd.DataFrame()
        
        # Combine all data into single DataFrame
        result_df = pd.DataFrame(all_data)
        result_df.index.name = 'Date'
        
        # Fill missing values with forward fill, then backward fill
        result_df = result_df.fillna(method='ffill').fillna(method='bfill')
        
        # Save raw data
        raw_file = self.raw_dir / "raw_prices_tinvest.csv"
        result_df.to_csv(raw_file)
        logger.info(f"Saved {len(result_df)} records for {len(result_df.columns)} symbols to {raw_file}")
        
        return result_df
    
    def get_asset_universe(self) -> List[str]:
        """Get list of available assets from configuration."""
        return self.config['assets']['russian_futures']
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate downloaded data quality."""
        if data.empty:
            logger.error("Data is empty")
            return False
        
        # Check for too many missing values
        missing_pct = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        if missing_pct > 0.5:
            logger.warning(f"High percentage of missing data: {missing_pct:.2%}")
        
        # Check for zero or negative prices
        zero_prices = (data <= 0).sum().sum()
        if zero_prices > 0:
            logger.warning(f"Found {zero_prices} zero or negative prices")
        
        logger.info(f"Data validation completed. Shape: {data.shape}, Missing: {missing_pct:.2%}")
        return True 