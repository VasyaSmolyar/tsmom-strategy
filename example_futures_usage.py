#!/usr/bin/env python3
"""
Example usage of MOEX futures information methods.
Demonstrates how to use the new futures information methods.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.moex_loader import MoexLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Example usage of futures information methods."""
    print("🚀 MOEX Futures Information Example")
    print("=" * 50)
    
    # Initialize loader
    loader = MoexLoader()
    
    # Example 1: Get all available futures
    print("\n1️⃣ Getting all available futures...")
    all_futures = loader.get_available_futures()
    print(f"📊 Total futures: {len(all_futures)}")
    
    # Example 2: Get Brent oil futures
    print("\n2️⃣ Getting Brent oil futures...")
    brent_futures = loader.get_futures_by_type("BR")
    print(f"📈 Brent futures found: {len(brent_futures)}")
    if not brent_futures.empty:
        print("📋 Brent futures:")
        for _, row in brent_futures.head().iterrows():
            print(f"   {row['ticker']}: {row['name']} (Expires: {row['expiration_date'].strftime('%Y-%m-%d')})")
    
    # Example 3: Get active futures
    print("\n3️⃣ Getting active futures...")
    active_futures = loader.get_active_futures()
    print(f"📈 Active futures: {len(active_futures)}")
    
    # Example 4: Get futures by asset type
    asset_types = ["BR", "GD", "Si"]  # Brent, Gold, Silver
    print("\n4️⃣ Getting futures by asset type...")
    for asset_type in asset_types:
        futures = loader.get_futures_by_type(asset_type)
        if not futures.empty:
            print(f"📊 {asset_type}: {len(futures)} contracts")
            # Show nearest expiration
            nearest = futures.sort_values('expiration_date').iloc[0]
            print(f"   Nearest: {nearest['ticker']} expires {nearest['expiration_date'].strftime('%Y-%m-%d')}")
        else:
            print(f"📊 {asset_type}: No contracts found")
    
    # Example 5: Get futures with data for specific period
    print("\n5️⃣ Getting futures with data for 2025-01-01 to 2025-01-31...")
    period_futures = loader.get_futures_with_data("2025-01-01", "2025-01-31", "BR")
    print(f"📈 Brent futures active in period: {len(period_futures)}")
    
    # Example 6: Download data for specific futures
    print("\n6️⃣ Downloading data for specific futures...")
    if not brent_futures.empty:
        # Get the nearest Brent future
        nearest_brent = brent_futures.sort_values('expiration_date').iloc[0]
        ticker = nearest_brent['ticker']
        
        print(f"📊 Downloading data for {ticker}...")
        data = loader._get_futures_data(ticker, "2025-01-01", "2025-01-31")
        
        if not data.empty:
            print(f"✅ Successfully downloaded {len(data)} records")
            print(f"💰 Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
            print(f"📈 Average volume: {data['volume'].mean():.0f}")
            
            # Save data
            filename = f"example_{ticker}_data.csv"
            data.to_csv(f"data/raw/{filename}")
            print(f"💾 Saved to data/raw/{filename}")
        else:
            print(f"❌ No data available for {ticker}")
    
    # Example 7: Show summary statistics
    print("\n7️⃣ Summary statistics...")
    print(f"📊 Total futures contracts: {len(all_futures)}")
    print(f"📈 Active contracts: {len(active_futures)}")
    
    # Show top asset types by number of contracts
    if not all_futures.empty:
        asset_counts = all_futures['ticker'].str[:2].value_counts()
        print("\n📊 Top asset types by number of contracts:")
        for asset_type, count in asset_counts.head(10).items():
            print(f"   {asset_type}: {count} contracts")
    
    print("\n🎉 Example completed!")

if __name__ == "__main__":
    main() 