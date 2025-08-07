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
    """MOEX API data loader implementation with futures support."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize MOEX loader with configuration."""
        super().__init__(config_path)
        self.api_token = None  # Placeholder for API token
        self.base_url = "https://iss.moex.com/iss"  # MOEX ISS API URL
        self.futures_dir = self.raw_dir / "moex"
        
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
            
            # Convert date
            df['DATE'] = pd.to_datetime(df['DATE'], format='%y%m%d')
            
            # Set date as index
            df.set_index('DATE', inplace=True)
            
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
        Also loads futures data from local files.
        
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
        
        logger.info(f"Loading MOEX data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        # Load futures data from local files
        futures_data = self.load_futures_data(symbols)
        
        if not futures_data.empty:
            logger.info(f"Successfully loaded futures data with {len(futures_data.columns)} symbols")
            return futures_data
        else:
            logger.warning("No futures data available, returning empty DataFrame")
            # Create empty DataFrame with proper structure
            empty_data = pd.DataFrame(columns=symbols, index=pd.date_range(start=start_date, end=end_date))
            empty_data.index.name = 'Date'
            
            # Save raw data
            raw_file = self.raw_dir / "raw_prices_moex.csv"
            empty_data.to_csv(raw_file)
            logger.info(f"Saved placeholder data to {raw_file}")
            
            return empty_data 