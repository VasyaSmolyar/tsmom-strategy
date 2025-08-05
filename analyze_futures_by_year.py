#!/usr/bin/env python3
"""
Script to analyze MOEX futures data by year and create detailed reports.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def analyze_futures_by_year():
    """Analyze futures data by year and create reports."""
    
    # Load the futures table
    futures_file = Path("data/moex_futures/moex_futures_table.csv")
    
    if not futures_file.exists():
        print("❌ Futures table not found. Please run scrape_moex_futures.py first.")
        return
    
    print("📊 Analyzing MOEX futures data by year...")
    
    # Load data
    df = pd.read_csv(futures_file)
    
    # Create analysis directory
    analysis_dir = Path("data/moex_futures/analysis")
    analysis_dir.mkdir(exist_ok=True)
    
    # 1. Year distribution analysis
    print("\n📈 Year Distribution Analysis:")
    year_stats = df.groupby('year').agg({
        'asset_code': 'count',
        'asset_name': 'nunique'
    }).rename(columns={
        'asset_code': 'total_contracts',
        'asset_name': 'unique_assets'
    })
    
    print(year_stats)
    
    # 2. Asset type distribution by year
    print("\n📊 Asset Type Distribution by Year:")
    asset_year_stats = df.groupby(['year', 'asset_code']).size().unstack(fill_value=0)
    print(asset_year_stats.head(10))
    
    # 3. Create detailed year report
    year_report_file = analysis_dir / "year_analysis.md"
    with open(year_report_file, 'w', encoding='utf-8') as f:
        f.write("# MOEX Futures Analysis by Year\n\n")
        f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Year Overview\n\n")
        f.write("| Year | Total Contracts | Unique Assets |\n")
        f.write("|------|----------------|---------------|\n")
        for year, row in year_stats.iterrows():
            f.write(f"| {year} | {row['total_contracts']} | {row['unique_assets']} |\n")
        
        f.write("\n## Asset Types by Year\n\n")
        f.write("| Asset Code | Asset Name | 2010-2019 | 2020-2029 | 2030 |\n")
        f.write("|------------|------------|-----------|-----------|------|\n")
        
        # Group years into decades
        df['decade'] = df['year'].apply(lambda x: '2010-2019' if 2010 <= x <= 2019 else 
                                       '2020-2029' if 2020 <= x <= 2029 else '2030')
        
        asset_decade_stats = df.groupby(['asset_code', 'decade']).size().unstack(fill_value=0)
        
        # Create asset name mapping
        asset_names = df.groupby('asset_code')['asset_name'].first()
        
        for asset_code in asset_decade_stats.index:
            asset_name = asset_names[asset_code]
            decade_2010s = asset_decade_stats.loc[asset_code, '2010-2019'] if '2010-2019' in asset_decade_stats.columns else 0
            decade_2020s = asset_decade_stats.loc[asset_code, '2020-2029'] if '2020-2029' in asset_decade_stats.columns else 0
            year_2030 = asset_decade_stats.loc[asset_code, '2030'] if '2030' in asset_decade_stats.columns else 0
            
            f.write(f"| {asset_code} | {asset_name} | {decade_2010s} | {decade_2020s} | {year_2030} |\n")
    
    print(f"💾 Saved year analysis to {year_report_file}")
    
    # 4. Create sample contracts for each year
    sample_contracts_file = analysis_dir / "sample_contracts_by_year.md"
    with open(sample_contracts_file, 'w', encoding='utf-8') as f:
        f.write("# Sample Contracts by Year\n\n")
        
        for year in sorted(df['year'].unique()):
            f.write(f"## {year}\n\n")
            
            year_data = df[df['year'] == year]
            
            # Show sample contracts for each asset type
            for asset_code in ['MX', 'BR', 'GD', 'SI', 'NG']:
                asset_contracts = year_data[year_data['asset_code'] == asset_code]
                if not asset_contracts.empty:
                    f.write(f"### {asset_code} ({asset_contracts.iloc[0]['asset_name']})\n\n")
                    
                    # Show first 6 months
                    sample_months = asset_contracts.head(6)
                    for _, contract in sample_months.iterrows():
                        f.write(f"- **{contract['short_code']}** ({contract['month_name']} {year}): {contract['long_code']}\n")
                    
                    f.write("\n")
    
    print(f"💾 Saved sample contracts to {sample_contracts_file}")
    
    # 5. Create statistics summary
    stats_summary = {
        'total_contracts': len(df),
        'total_years': len(df['year'].unique()),
        'total_assets': len(df['asset_code'].unique()),
        'total_months': len(df['month_code'].unique()),
        'year_range': f"{df['year'].min()} - {df['year'].max()}",
        'contracts_per_year': len(df) // len(df['year'].unique()),
        'contracts_per_asset': len(df) // len(df['asset_code'].unique())
    }
    
    stats_file = analysis_dir / "statistics_summary.md"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("# MOEX Futures Statistics Summary\n\n")
        
        f.write("## Overall Statistics\n\n")
        for key, value in stats_summary.items():
            f.write(f"- **{key.replace('_', ' ').title()}**: {value}\n")
        
        f.write("\n## Year Distribution\n\n")
        for year in sorted(df['year'].unique()):
            year_count = len(df[df['year'] == year])
            f.write(f"- **{year}**: {year_count} contracts\n")
        
        f.write("\n## Asset Type Distribution\n\n")
        asset_counts = df['asset_code'].value_counts()
        for asset_code, count in asset_counts.items():
            asset_name = df[df['asset_code'] == asset_code]['asset_name'].iloc[0]
            f.write(f"- **{asset_code}** ({asset_name}): {count} contracts\n")
    
    print(f"💾 Saved statistics summary to {stats_file}")
    
    # 6. Print summary to console
    print(f"\n📊 ANALYSIS SUMMARY:")
    print(f"   Total contracts: {stats_summary['total_contracts']}")
    print(f"   Year range: {stats_summary['year_range']}")
    print(f"   Contracts per year: {stats_summary['contracts_per_year']}")
    print(f"   Asset types: {stats_summary['total_assets']}")
    
    print(f"\n📈 Year Distribution:")
    for year in sorted(df['year'].unique()):
        year_count = len(df[df['year'] == year])
        print(f"   {year}: {year_count} contracts")
    
    return {
        'year_report': str(year_report_file),
        'sample_contracts': str(sample_contracts_file),
        'statistics': str(stats_file),
        'total_contracts': stats_summary['total_contracts'],
        'year_range': stats_summary['year_range']
    }

if __name__ == "__main__":
    analyze_futures_by_year() 