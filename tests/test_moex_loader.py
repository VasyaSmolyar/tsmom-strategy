#!/usr/bin/env python3
"""
Test script for MoexLoader with futures data loading and integrity checks.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.moex_loader import MoexLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_moex_loader():
    """Test the MoexLoader with futures data."""
    
    print("Testing MoexLoader with futures data...")
    
    # Initialize loader
    loader = MoexLoader()
    
    # Test loading all futures data
    print("\n1. Loading all available futures data...")
    all_futures = loader.load_futures_data()
    
    if not all_futures.empty:
        print(f"Successfully loaded {len(all_futures.columns)} futures:")
        for ticker in all_futures.columns:
            print(f"  - {ticker}: {len(all_futures[ticker].dropna())} data points")
        
        print(f"\nData shape: {all_futures.shape}")
        print(f"Date range: {all_futures.index.min()} to {all_futures.index.max()}")
        
        # Show sample data
        print("\nSample data (first 5 rows):")
        print(all_futures.head())
        
        # Check for missing values
        missing_pct = all_futures.isnull().sum() / len(all_futures) * 100
        print(f"\nMissing data percentage by ticker:")
        for ticker, pct in missing_pct.items():
            print(f"  {ticker}: {pct:.1f}%")
            
    else:
        print("No futures data loaded")
    
    # Test loading specific symbols
    print("\n2. Loading specific symbols (Si, GOLD)...")
    specific_futures = loader.load_futures_data(symbols=['Si', 'GOLD'])
    
    if not specific_futures.empty:
        print(f"Loaded {len(specific_futures.columns)} specific futures:")
        for ticker in specific_futures.columns:
            print(f"  - {ticker}: {len(specific_futures[ticker].dropna())} data points")
    
    # Test the main download_data method
    print("\n3. Testing main download_data method...")
    main_data = loader.download_data()
    
    if not main_data.empty:
        print(f"Main method loaded {len(main_data.columns)} symbols")
        print(f"Data shape: {main_data.shape}")
    else:
        print("Main method returned empty data")

if __name__ == "__main__":
    test_moex_loader()
