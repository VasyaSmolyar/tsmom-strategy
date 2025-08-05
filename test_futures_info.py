#!/usr/bin/env python3
"""
Test script for MOEX futures information methods.
Tests the new methods for getting available futures with trading dates.
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

def test_available_futures():
    """Test getting all available futures."""
    print("Testing get_available_futures()...")
    print("=" * 50)
    
    try:
        loader = MoexLoader()
        
        # Get all available futures
        futures_df = loader.get_available_futures()
        
        if not futures_df.empty:
            print(f"✅ Successfully retrieved {len(futures_df)} futures contracts")
            print(f"📊 DataFrame shape: {futures_df.shape}")
            
            # Show key columns
            key_columns = ['ticker', 'name', 'trading_start', 'expiration_date', 'is_active', 'days_to_expiration']
            display_columns = [col for col in key_columns if col in futures_df.columns]
            
            print(f"\n📋 Sample futures (first 10):")
            print(futures_df[display_columns].head(10))
            
            # Show active vs inactive
            active_count = futures_df['is_active'].sum() if 'is_active' in futures_df.columns else 0
            print(f"\n📈 Active contracts: {active_count}")
            print(f"📉 Inactive contracts: {len(futures_df) - active_count}")
            
            return futures_df
        else:
            print("❌ No futures data retrieved")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def test_futures_by_type():
    """Test getting futures by asset type."""
    print("\nTesting get_futures_by_type()...")
    print("=" * 50)
    
    try:
        loader = MoexLoader()
        
        # Test different asset types
        asset_types = ['BR', 'SI', 'GD', 'CL']
        
        for asset_type in asset_types:
            print(f"\n🔍 Testing {asset_type} futures...")
            
            futures_df = loader.get_futures_by_type(asset_type)
            
            if not futures_df.empty:
                print(f"✅ Found {len(futures_df)} {asset_type} futures")
                
                # Show key information
                key_columns = ['ticker', 'name', 'trading_start', 'expiration_date', 'is_active']
                display_columns = [col for col in key_columns if col in futures_df.columns]
                
                print(f"📋 {asset_type} futures:")
                print(futures_df[display_columns].head())
                
                # Show active contracts
                active_count = futures_df['is_active'].sum() if 'is_active' in futures_df.columns else 0
                print(f"📈 Active {asset_type} contracts: {active_count}")
            else:
                print(f"❌ No {asset_type} futures found")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_active_futures():
    """Test getting active futures."""
    print("\nTesting get_active_futures()...")
    print("=" * 50)
    
    try:
        loader = MoexLoader()
        
        # Get all active futures
        active_futures = loader.get_active_futures()
        
        if not active_futures.empty:
            print(f"✅ Found {len(active_futures)} active futures contracts")
            
            # Show key information
            key_columns = ['ticker', 'name', 'trading_start', 'expiration_date', 'days_to_expiration']
            display_columns = [col for col in key_columns if col in active_futures.columns]
            
            print(f"📋 Active futures (first 10):")
            print(active_futures[display_columns].head(10))
            
            # Show by asset type
            print(f"\n📊 Active futures by asset type:")
            if 'ticker' in active_futures.columns:
                asset_types = active_futures['ticker'].str[:2].value_counts()
                for asset_type, count in asset_types.head(10).items():
                    print(f"   {asset_type}: {count} contracts")
                    
        else:
            print("❌ No active futures found")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_futures_with_data():
    """Test getting futures with data in specific period."""
    print("\nTesting get_futures_with_data()...")
    print("=" * 50)
    
    try:
        loader = MoexLoader()
        
        # Test different periods
        test_periods = [
            ("2025-01-01", "2025-01-31", "BR"),
            ("2024-06-01", "2024-06-30", "BR"),
            ("2025-02-01", "2025-02-28", "SI")
        ]
        
        for start_date, end_date, asset_type in test_periods:
            print(f"\n🔍 Testing {asset_type} futures for {start_date} to {end_date}...")
            
            futures_df = loader.get_futures_with_data(start_date, end_date, asset_type)
            
            if not futures_df.empty:
                print(f"✅ Found {len(futures_df)} {asset_type} futures active during period")
                
                # Show key information
                key_columns = ['ticker', 'name', 'trading_start', 'expiration_date']
                display_columns = [col for col in key_columns if col in futures_df.columns]
                
                print(f"📋 {asset_type} futures for period:")
                print(futures_df[display_columns].head())
            else:
                print(f"❌ No {asset_type} futures found for period")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_specific_futures_data():
    """Test downloading data for specific futures found."""
    print("\nTesting data download for specific futures...")
    print("=" * 50)
    
    try:
        loader = MoexLoader()
        
        # Get active Brent futures
        brent_futures = loader.get_active_futures("BR")
        
        if not brent_futures.empty:
            print(f"✅ Found {len(brent_futures)} active Brent futures")
            
            # Try to download data for the first active contract
            first_ticker = brent_futures.iloc[0]['ticker']
            print(f"\n🔍 Testing data download for {first_ticker}...")
            
            # Get data for recent period
            data = loader._get_futures_data(first_ticker, "2025-01-01", "2025-01-31")
            
            if not data.empty:
                print(f"✅ Successfully downloaded data for {first_ticker}")
                print(f"📊 Data shape: {data.shape}")
                print(f"📅 Date range: {data.index.min()} to {data.index.max()}")
                print(f"💰 Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
                
                # Save test data
                test_file = loader.raw_dir / f"test_{first_ticker}_data.csv"
                data.to_csv(test_file)
                print(f"💾 Saved data to {test_file}")
            else:
                print(f"❌ No data available for {first_ticker}")
        else:
            print("❌ No active Brent futures found")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests."""
    print("🚀 Starting MOEX futures information tests...")
    print("=" * 60)
    
    # Test all methods
    all_futures = test_available_futures()
    test_futures_by_type()
    test_active_futures()
    test_futures_with_data()
    test_specific_futures_data()
    
    print("\n" + "=" * 60)
    print("🎉 All futures information tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    main() 