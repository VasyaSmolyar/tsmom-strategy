#!/usr/bin/env python3
"""
Test script for MOEX CL-12.24 futures data download.
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
    """Test CL-12.24 futures data download."""
    print("Testing MOEX CL-12.24 futures data download...")
    print("=" * 50)
    
    try:
        # Initialize MOEX loader
        loader = MoexLoader()
        
        # Test CL-12.24 futures download
        data = loader.test_cl_futures_download()
        
        if not data.empty:
            print("\n✅ Successfully downloaded CL-12.24 futures data!")
            print(f"📊 Data shape: {data.shape}")
            print(f"📅 Date range: {data.index.min()} to {data.index.max()}")
            print(f"💰 Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
            print(f"📈 Average volume: {data['volume'].mean():.0f}")
            
            # Show first few rows
            print("\n📋 First 5 rows:")
            print(data.head())
            
            # Show last few rows
            print("\n📋 Last 5 rows:")
            print(data.tail())
            
        else:
            print("❌ No data received for CL-12.24")
            print("This might be due to:")
            print("- Symbol not found in MOEX API")
            print("- Date range not available")
            print("- API endpoint issues")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 