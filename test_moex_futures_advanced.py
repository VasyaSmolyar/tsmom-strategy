#!/usr/bin/env python3
"""
Advanced test script for MOEX futures data download.
Tests multiple futures symbols and data processing.
"""

import sys
import logging
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.moex_loader import MoexLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_multiple_futures():
    """Test download of multiple futures symbols."""
    print("Testing MOEX futures data download for multiple symbols...")
    print("=" * 60)
    
    try:
        # Initialize MOEX loader
        loader = MoexLoader()
        
        # Test symbols
        test_symbols = ["BRF6", "BRG6", "BRH6"]
        start_date = "2025-01-01"
        end_date = "2025-01-31"
        
        all_data = {}
        
        for symbol in test_symbols:
            print(f"\n🔍 Testing {symbol}...")
            
            data = loader._get_futures_data(symbol, start_date, end_date)
            
            if not data.empty:
                print(f"✅ Successfully downloaded {len(data)} records for {symbol}")
                print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
                print(f"   Average volume: {data['volume'].mean():.0f}")
                
                # Use typical price for TSMOM strategy
                all_data[symbol] = data['typical_price']
                
                # Save individual symbol data
                symbol_file = loader.raw_dir / f"test_{symbol}.csv"
                data.to_csv(symbol_file)
                print(f"   Saved to: {symbol_file}")
            else:
                print(f"❌ No data for {symbol}")
        
        if all_data:
            # Create combined DataFrame
            combined_df = pd.DataFrame(all_data)
            combined_df.index.name = 'Date'
            
            # Save combined data
            combined_file = loader.raw_dir / "test_combined_futures.csv"
            combined_df.to_csv(combined_file)
            print(f"\n📊 Combined data saved to: {combined_file}")
            print(f"📈 Combined data shape: {combined_df.shape}")
            print(f"📅 Date range: {combined_df.index.min()} to {combined_df.index.max()}")
            
            # Show correlation matrix
            print("\n📊 Correlation matrix:")
            print(combined_df.corr().round(3))
            
            return combined_df
        else:
            print("❌ No data downloaded for any symbol")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def test_data_processing():
    """Test data processing and cleaning."""
    print("\n" + "=" * 60)
    print("Testing data processing and cleaning...")
    print("=" * 60)
    
    try:
        loader = MoexLoader()
        
        # Download test data
        data = loader.test_cl_futures_download()
        
        if not data.empty:
            print("✅ Data downloaded successfully")
            
            # Test cleaning
            print("\n🧹 Testing data cleaning...")
            cleaned_data = loader.clean_data(data[['close']])  # Use only close prices for cleaning test
            print(f"📊 Original shape: {data.shape}")
            print(f"📊 Cleaned shape: {cleaned_data.shape}")
            
            # Test returns calculation
            print("\n📈 Testing returns calculation...")
            returns = loader.calculate_returns(cleaned_data)
            print(f"📊 Returns shape: {returns.shape}")
            print(f"📊 Returns sample:")
            print(returns.head())
            
            return cleaned_data
        else:
            print("❌ No data to process")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Error during processing test: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def main():
    """Run all tests."""
    print("🚀 Starting advanced MOEX futures tests...")
    
    # Test multiple futures
    futures_data = test_multiple_futures()
    
    # Test data processing
    processed_data = test_data_processing()
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    main() 