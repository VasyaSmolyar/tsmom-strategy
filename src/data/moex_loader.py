"""
MOEX API data loader implementation for TSMOM backtest.
Handles downloading historical price data from MOEX API and loading futures data from local files.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
import logging
from pathlib import Path
import glob
import re
from datetime import datetime, timedelta
from .base_loader import DataLoader

logger = logging.getLogger(__name__)


class MoexLoader(DataLoader):
    """MOEX API data loader implementation with futures and index support."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize MOEX loader with configuration."""
        super().__init__(config_path)
        self._source_suffix = 'moex'
        self.api_token = None  # Placeholder for API token
        self.base_url = "https://iss.moex.com/iss"  # MOEX ISS API URL
        self.futures_dir = self.raw_dir / "moex"
        self.imoex_dir = self.raw_dir / "moex" / "imoex"
        
        # Mapping of MOEX ticker symbols to their file names
        self.moex_ticker_mapping = {
            'Si': 'USDRUB',
            'Eu': 'EURUSD', 
            'CNY': 'CNYUSD',
            'GAZR': 'GAZPROM',
            'SBRF': 'SBERBANK',
            'LKOH': 'LUKOIL',
            'ROSN': 'ROSNEFT',
            'VTBR': 'VTB',
            'GMKN': 'NORILSK',
            'RTS': 'RTS_INDEX',
            'MIX': 'MOEX_INDEX',
            'BZ': 'BRENT_OIL',
            'NG': 'NATURAL_GAS',
            'GOLD': 'GOLD_FUTURES',
            'SILV': 'SILVER',
            'PLT': 'PLATINUM'
        }
    
    def get_asset_universe(self) -> List[str]:
        """Get MOEX futures tickers available in the data files."""
        if not self.futures_dir.exists():
            logger.warning(f"Futures directory {self.futures_dir} does not exist")
            return []
        
        # Get all CSV files and extract unique tickers
        csv_files = list(self.futures_dir.glob("*.csv"))
        available_tickers = set()
        
        for filepath in csv_files:
            try:
                ticker, _, _ = self._parse_futures_filename(filepath.name)
                available_tickers.add(ticker)
            except ValueError:
                continue
        
        # Return list of available tickers (these are the actual MOEX symbols)
        return sorted(list(available_tickers))
    
    def get_moex_asset_universe(self) -> List[str]:
        """Get the asset universe suitable for MOEX (only Russian assets and available futures)."""
        # Get available futures tickers from MOEX data files
        available_futures = self.get_asset_universe()
        
        # Start with futures and use set to avoid duplicates
        assets = set(available_futures)
        asset_config = self.config['assets']
        
        # Include Russian assets and MOEX assets for MOEX loader
        for category_name, category in asset_config.items():
            if isinstance(category, list):
                # Only include Russian-specific assets for MOEX
                if category_name in ['russian_equities']:
                    assets.update(category)
                # Include MOEX assets from other categories
                moex_assets = [asset for asset in category if asset.endswith('.ME')]
                assets.update(moex_assets)
        
        # Convert back to sorted list
        assets_list = sorted(list(assets))
        logger.info(f"MOEX asset universe: {len(assets_list)} assets - {assets_list}")
        return assets_list
        
    def _parse_futures_filename(self, filename: str) -> Tuple[str, str, str]:
        """
        Parse futures filename to extract ticker and date range.
        
        Args:
            filename: Filename like "Si_100803_150801.csv"
            
        Returns:
            Tuple of (ticker, start_date, end_date)
        """
        # Remove .csv extension
        name = filename.replace('.csv', '')
        
        # Split by underscore
        parts = name.split('_')
        if len(parts) != 3:
            raise ValueError(f"Invalid filename format: {filename}")
        
        ticker = parts[0]
        start_date = parts[1]
        end_date = parts[2]
        
        return ticker, start_date, end_date
    
    def _parse_moex_date(self, date_str: str) -> datetime:
        """
        Parse MOEX date format (YYMMDD) to datetime.
        
        Args:
            date_str: Date string in format YYMMDD
            
        Returns:
            datetime object
        """
        year = int('20' + date_str[:2])
        month = int(date_str[2:4])
        day = int(date_str[4:6])
        return datetime(year, month, day)
    
    def _load_futures_file(self, filepath: Path) -> pd.DataFrame:
        """
        Load a single futures CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Read CSV with semicolon separator
            df = pd.read_csv(filepath, sep=';', encoding='utf-8')
            
            # Check if file has expected structure
            expected_columns = ['<TICKER>', '<PER>', '<DATE>', '<TIME>', 
                              '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>']
            
            if not all(col in df.columns for col in expected_columns):
                logger.warning(f"File {filepath} has unexpected structure")
                return pd.DataFrame()
            
            # Clean column names
            df.columns = [col.replace('<', '').replace('>', '') for col in df.columns]
            
            # Convert date - try different formats
            try:
                df['DATE'] = pd.to_datetime(df['DATE'], format='%y%m%d')
            except ValueError:
                try:
                    # Try YYYYMMDD format for newer files
                    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d')
                except ValueError:
                    try:
                        # Try as string conversion first 
                        df['DATE'] = df['DATE'].astype(str)
                        # Filter out invalid date strings (too short or non-numeric)
                        df = df[df['DATE'].str.len() >= 6]
                        df = df[df['DATE'].str.isdigit()]
                        # Try parsing again
                        df['DATE'] = pd.to_datetime(df['DATE'], format='%y%m%d', errors='coerce')
                        df = df.dropna(subset=['DATE'])
                    except Exception:
                        logger.warning(f"Could not parse dates in {filepath}")
                        return pd.DataFrame()
            
            # Set date as index
            df.set_index('DATE', inplace=True)
            
            # Filter out dates before 2000 (invalid parsed dates)
            df = df[df.index >= '2000-01-01']
            
            # Convert numeric columns
            numeric_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOL']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Filter out rows with missing data
            df = df.dropna(subset=['CLOSE'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading file {filepath}: {e}")
            return pd.DataFrame()
    
    def _check_data_integrity(self, df: pd.DataFrame, ticker: str) -> Dict[str, any]:
        """
        Check data integrity for a single ticker.
        
        Args:
            df: DataFrame with price data
            ticker: Ticker symbol
            
        Returns:
            Dictionary with integrity check results
        """
        if df.empty:
            return {
                'ticker': ticker,
                'status': 'EMPTY',
                'issues': ['No data available'],
                'data_points': 0,
                'date_range': None,
                'missing_dates': None
            }
        
        issues = []
        
        # Check for missing values
        missing_pct = df.isnull().sum() / len(df)
        for col, pct in missing_pct.items():
            if pct > 0.1:  # More than 10% missing
                issues.append(f"High missing values in {col}: {pct:.1%}")
        
        # Check for price anomalies
        if 'CLOSE' in df.columns:
            # Check for zero or negative prices
            zero_prices = (df['CLOSE'] <= 0).sum()
            if zero_prices > 0:
                issues.append(f"Found {zero_prices} zero/negative prices")
            
            # Check for extreme price changes (>50% in one day)
            price_changes = df['CLOSE'].pct_change().abs()
            extreme_changes = (price_changes > 0.5).sum()
            if extreme_changes > 0:
                issues.append(f"Found {extreme_changes} extreme price changes (>50%)")
        
        # Check for gaps in data
        expected_dates = pd.date_range(df.index.min(), df.index.max(), freq='D')
        missing_dates = expected_dates.difference(df.index)
        if len(missing_dates) > 0:
            issues.append(f"Missing {len(missing_dates)} trading days")
        
        # Check for duplicate dates
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate dates")
        
        status = 'OK' if not issues else 'ISSUES'
        
        return {
            'ticker': ticker,
            'status': status,
            'issues': issues,
            'data_points': len(df),
            'date_range': (df.index.min(), df.index.max()) if not df.empty else None,
            'missing_dates': list(missing_dates) if len(missing_dates) > 0 else None
        }
    
    def load_futures_data(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load futures data from local CSV files.
        
        Args:
            symbols: List of symbols to load. If None, loads all available.
            
        Returns:
            DataFrame with OHLCV data for all symbols
        """
        if not self.futures_dir.exists():
            logger.warning(f"Futures directory {self.futures_dir} does not exist")
            return pd.DataFrame()
        
        # Get all CSV files
        csv_files = list(self.futures_dir.glob("*.csv"))
        if not csv_files:
            logger.warning(f"No CSV files found in {self.futures_dir}")
            return pd.DataFrame()
        
        logger.info(f"Found {len(csv_files)} futures data files")
        
        # Group files by ticker
        ticker_files = {}
        for filepath in csv_files:
            try:
                ticker, start_date, end_date = self._parse_futures_filename(filepath.name)
                if ticker not in ticker_files:
                    ticker_files[ticker] = []
                ticker_files[ticker].append(filepath)
            except ValueError as e:
                logger.warning(f"Skipping file {filepath.name}: {e}")
                continue
        
        # Filter by requested symbols
        if symbols:
            ticker_files = {k: v for k, v in ticker_files.items() if k in symbols}
        
        logger.info(f"Processing {len(ticker_files)} tickers")
        
        # Load and combine data for each ticker
        all_data = {}
        integrity_reports = []
        
        for ticker, files in ticker_files.items():
            logger.info(f"Loading data for {ticker} from {len(files)} files")
            
            # Load all files for this ticker
            ticker_data = []
            for filepath in sorted(files):
                df = self._load_futures_file(filepath)
                if not df.empty:
                    ticker_data.append(df)
            
            if ticker_data:
                # Combine all data for this ticker
                combined_df = pd.concat(ticker_data, axis=0)
                
                # Remove duplicates and sort
                combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
                combined_df = combined_df.sort_index()
                
                # Check integrity
                integrity_report = self._check_data_integrity(combined_df, ticker)
                integrity_reports.append(integrity_report)
                
                if integrity_report['status'] == 'OK':
                    all_data[ticker] = combined_df['CLOSE']  # Use close prices
                    logger.info(f"Successfully loaded {ticker}: {len(combined_df)} data points")
                else:
                    logger.warning(f"Data integrity issues for {ticker}: {integrity_report['issues']}")
                    # Still include the data but log the issues
                    all_data[ticker] = combined_df['CLOSE']
            else:
                logger.warning(f"No valid data found for {ticker}")
        
        # Create final DataFrame
        if all_data:
            result_df = pd.DataFrame(all_data)
            result_df.index.name = 'Date'
            
            # Save raw data
            raw_file = self.raw_dir / "raw_futures_moex.csv"
            result_df.to_csv(raw_file)
            logger.info(f"Saved futures data to {raw_file}")
            
            # Log integrity summary
            self._log_integrity_summary(integrity_reports)
            
            return result_df
        else:
            logger.warning("No valid futures data found")
            return pd.DataFrame()
    
    def load_imoex_data(self) -> pd.DataFrame:
        """
        Load IMOEX index data from local CSV files.
        
        Returns:
            DataFrame with IMOEX price data
        """
        if not self.imoex_dir.exists():
            logger.warning(f"IMOEX directory {self.imoex_dir} does not exist")
            return pd.DataFrame()
        
        # Get all IMOEX CSV files
        csv_files = list(self.imoex_dir.glob("IMOEX_*.csv"))
        if not csv_files:
            logger.warning(f"No IMOEX CSV files found in {self.imoex_dir}")
            return pd.DataFrame()
        
        logger.info(f"Found {len(csv_files)} IMOEX data files")
        
        # Load all files for IMOEX
        imoex_data = []
        for filepath in sorted(csv_files):
            df = self._load_futures_file(filepath)
            if not df.empty:
                imoex_data.append(df)
        
        if imoex_data:
            # Combine all data for IMOEX
            combined_df = pd.concat(imoex_data, axis=0)
            
            # Remove duplicates and sort
            combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
            combined_df = combined_df.sort_index()
            
            # Check integrity
            integrity_report = self._check_data_integrity(combined_df, 'IMOEX')
            
            if integrity_report['status'] == 'OK':
                logger.info(f"Successfully loaded IMOEX: {len(combined_df)} data points")
            else:
                logger.warning(f"Data integrity issues for IMOEX: {integrity_report['issues']}")
            
            # Create DataFrame with IMOEX data
            result_df = pd.DataFrame({'IMOEX': combined_df['CLOSE']})
            result_df.index.name = 'Date'
            
            # Save raw data
            raw_file = self.raw_dir / "raw_imoex_moex.csv"
            result_df.to_csv(raw_file)
            logger.info(f"Saved IMOEX data to {raw_file}")
            
            return result_df
        else:
            logger.warning("No valid IMOEX data found")
            return pd.DataFrame()
    
    def _log_integrity_summary(self, integrity_reports: List[Dict]) -> None:
        """
        Log a summary of data integrity checks.
        
        Args:
            integrity_reports: List of integrity check results
        """
        total_tickers = len(integrity_reports)
        ok_tickers = sum(1 for report in integrity_reports if report['status'] == 'OK')
        issue_tickers = total_tickers - ok_tickers
        
        logger.info(f"Data integrity summary:")
        logger.info(f"  Total tickers: {total_tickers}")
        logger.info(f"  OK: {ok_tickers}")
        logger.info(f"  Issues: {issue_tickers}")
        
        if issue_tickers > 0:
            logger.warning("Tickers with issues:")
            for report in integrity_reports:
                if report['status'] != 'OK':
                    logger.warning(f"  {report['ticker']}: {', '.join(report['issues'])}")
    
    def download_data(self, symbols: Optional[List[str]] = None, 
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Download historical price data for given symbols using MOEX API.
        Also loads futures data and IMOEX index data from local files.
        
        Args:
            symbols: List of symbols to download. If None, uses config assets.
            start_date: Start date for data download. If None, uses config.
            end_date: End date for data download. If None, uses config.
        
        Returns:
            DataFrame with OHLCV data for all symbols
        """
        if symbols is None:
            symbols = self.get_moex_asset_universe()
        
        if start_date is None:
            start_date = self.config['data']['start_date']
        
        if end_date is None:
            end_date = self.config['data']['end_date']
        
        logger.info(f"Loading MOEX data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        # Load futures data from local files
        futures_data = self.load_futures_data(symbols)
        
        # Load IMOEX index data
        imoex_data = self.load_imoex_data()
        
        # Combine futures and IMOEX data
        combined_data = pd.DataFrame()
        
        if not futures_data.empty:
            combined_data = futures_data
            logger.info(f"Successfully loaded futures data with {len(futures_data.columns)} symbols")
        
        if not imoex_data.empty:
            if combined_data.empty:
                combined_data = imoex_data
            else:
                # Merge IMOEX data with futures data
                combined_data = combined_data.join(imoex_data, how='outer')
            logger.info(f"Successfully loaded IMOEX index data")
        
        if not combined_data.empty:
            # Save combined raw data
            raw_file = self.raw_dir / "raw_prices_moex.csv"
            combined_data.to_csv(raw_file)
            logger.info(f"Saved combined data to {raw_file}")
            
            return combined_data
        else:
            logger.warning("No data available, returning empty DataFrame")
            # Create empty DataFrame with proper structure
            empty_data = pd.DataFrame(columns=symbols, index=pd.date_range(start=start_date, end=end_date))
            empty_data.index.name = 'Date'
            
            # Save raw data
            raw_file = self.raw_dir / "raw_prices_moex.csv"
            empty_data.to_csv(raw_file)
            logger.info(f"Saved placeholder data to {raw_file}")
            
            return empty_data 