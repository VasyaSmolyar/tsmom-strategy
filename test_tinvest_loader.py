#!/usr/bin/env python3
"""
Test script for TInvestLoader to verify Russian futures data download.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.tinvest_loader import TInvestLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Test TInvestLoader functionality."""
    
    # Check if TINKOFF_TOKEN is set
    if not os.getenv('TINKOFF_TOKEN'):
        print("ERROR: TINKOFF_TOKEN environment variable is not set")
        print("Please set your Tinkoff API token:")
        print("export TINKOFF_TOKEN='your_token_here'")
        return
    
    try:
        # Initialize loader
        loader = TInvestLoader()
        
        # Test with a smaller date range for faster testing
        start_date = "2023-01-01"
        end_date = "2023-12-31"
        
        print(f"Testing TInvestLoader with date range: {start_date} to {end_date}")
        
        # Download data
        data = loader.download_data(
            symbols=["Si", "BR", "RI"],  # Test with 3 symbols
            start_date=start_date,
            end_date=end_date
        )
        
        if not data.empty:
            print(f"\nSuccessfully downloaded data:")
            print(f"Shape: {data.shape}")
            print(f"Columns: {list(data.columns)}")
            print(f"Date range: {data.index.min()} to {data.index.max()}")
            print(f"Missing values: {data.isnull().sum().sum()}")
            
            # Show first few rows
            print(f"\nFirst 5 rows:")
            print(data.head())
            
            # Validate data
            is_valid = loader.validate_data(data)
            print(f"\nData validation: {'PASSED' if is_valid else 'FAILED'}")
            
        else:
            print("No data downloaded")
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 