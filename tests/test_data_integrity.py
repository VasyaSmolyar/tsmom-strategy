#!/usr/bin/env python3
"""
Test script for data integrity checks in MoexLoader.
"""

import sys
import logging
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.moex_loader import MoexLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_data_integrity():
    """Test data integrity checks."""
    
    print("Testing data integrity checks...")
    
    # Initialize loader
    loader = MoexLoader()
    
    # Test with a specific ticker that has good data
    print("\n1. Testing data integrity for Si (Silver futures)...")
    
    # Load Si data specifically
    si_data = loader.load_futures_data(symbols=['Si'])
    
    if not si_data.empty:
        print(f"Si data shape: {si_data.shape}")
        print(f"Date range: {si_data.index.min()} to {si_data.index.max()}")
        
        # Check for missing values
        missing_count = si_data['Si'].isnull().sum()
        total_count = len(si_data)
        missing_pct = (missing_count / total_count) * 100
        
        print(f"Missing values: {missing_count} out of {total_count} ({missing_pct:.1f}%)")
        
        # Check for price anomalies
        si_prices = si_data['Si'].dropna()
        zero_prices = (si_prices <= 0).sum()
        extreme_changes = (si_prices.pct_change().abs() > 0.5).sum()
        
        print(f"Zero/negative prices: {zero_prices}")
        print(f"Extreme price changes (>50%): {extreme_changes}")
        
        # Show price statistics
        print(f"\nPrice statistics:")
        print(f"  Min: {si_prices.min():.2f}")
        print(f"  Max: {si_prices.max():.2f}")
        print(f"  Mean: {si_prices.mean():.2f}")
        print(f"  Std: {si_prices.std():.2f}")
        
        # Check for gaps in data
        expected_dates = pd.date_range(si_data.index.min(), si_data.index.max(), freq='D')
        missing_dates = expected_dates.difference(si_data.index)
        print(f"\nMissing trading days: {len(missing_dates)}")
        
        if len(missing_dates) > 0:
            print("Sample missing dates:")
            for date in missing_dates[:10]:
                print(f"  {date.strftime('%Y-%m-%d')}")
            if len(missing_dates) > 10:
                print(f"  ... and {len(missing_dates) - 10} more")
    
    # Test with a ticker that has poor data coverage
    print("\n2. Testing data integrity for NG (Natural Gas futures)...")
    
    ng_data = loader.load_futures_data(symbols=['NG'])
    
    if not ng_data.empty:
        print(f"NG data shape: {ng_data.shape}")
        print(f"Date range: {ng_data.index.min()} to {ng_data.index.max()}")
        
        missing_count = ng_data['NG'].isnull().sum()
        total_count = len(ng_data)
        missing_pct = (missing_count / total_count) * 100
        
        print(f"Missing values: {missing_count} out of {total_count} ({missing_pct:.1f}%)")
        
        # Show data coverage by year
        ng_prices = ng_data['NG'].dropna()
        yearly_coverage = ng_prices.groupby(ng_prices.index.year).count()
        print(f"\nData coverage by year:")
        for year, count in yearly_coverage.items():
            print(f"  {year}: {count} days")
    
    # Test data quality metrics
    print("\n3. Testing overall data quality metrics...")
    
    all_futures = loader.load_futures_data()
    
    if not all_futures.empty:
        print(f"Overall data quality summary:")
        print(f"  Total tickers: {len(all_futures.columns)}")
        print(f"  Total time period: {all_futures.index.min()} to {all_futures.index.max()}")
        print(f"  Total trading days: {len(all_futures)}")
        
        # Calculate coverage statistics
        coverage_stats = []
        for ticker in all_futures.columns:
            data_points = all_futures[ticker].dropna().count()
            total_days = len(all_futures)
            coverage_pct = (data_points / total_days) * 100
            coverage_stats.append({
                'ticker': ticker,
                'data_points': data_points,
                'coverage_pct': coverage_pct
            })
        
        # Sort by coverage
        coverage_stats.sort(key=lambda x: x['coverage_pct'], reverse=True)
        
        print(f"\nCoverage by ticker (sorted by coverage):")
        for stat in coverage_stats:
            print(f"  {stat['ticker']}: {stat['data_points']} points ({stat['coverage_pct']:.1f}%)")
        
        # Identify best and worst performers
        best_ticker = coverage_stats[0]
        worst_ticker = coverage_stats[-1]
        
        print(f"\nBest coverage: {best_ticker['ticker']} ({best_ticker['coverage_pct']:.1f}%)")
        print(f"Worst coverage: {worst_ticker['ticker']} ({worst_ticker['coverage_pct']:.1f}%)")

if __name__ == "__main__":
    test_data_integrity()
