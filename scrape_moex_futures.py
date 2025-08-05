#!/usr/bin/env python3
"""
Script to scrape futures information from MOEX website and create a comprehensive table.
Based on the content from https://www.moex.com/s205
"""

import requests
import pandas as pd
import re
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class MoexFuturesScraper:
    """Scraper for MOEX futures information."""
    
    def __init__(self):
        self.base_url = "https://www.moex.com/s205"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def scrape_futures_info(self):
        """Scrape futures information from MOEX website."""
        try:
            print(f"🔍 Scraping futures information from {self.base_url}...")
            
            response = self.session.get(self.base_url)
            response.raise_for_status()
            
            content = response.text
            
            # Extract futures information from the content
            futures_data = self._parse_futures_content(content)
            
            return futures_data
            
        except Exception as e:
            print(f"❌ Error scraping MOEX website: {e}")
            return None
    
    def _parse_futures_content(self, content):
        """Parse the HTML content to extract futures information."""
        futures_info = {
            'contract_codes': [],
            'asset_types': [],
            'month_codes': [],
            'year_codes': [],
            'examples': []
        }
        
        # Extract asset type codes from the content
        # Based on the website content, we can see various asset codes
        asset_codes = {
            'MX': 'MOEX',
            'MM': 'MOEX_Mini',
            'MY': 'MOEX_CNY',
            'RI': 'RTS',
            'BR': 'Brent',
            'GD': 'Gold',
            'SI': 'Silver',
            'CL': 'WTI',
            'NG': 'Natural_Gas',
            'SB': 'Sugar',
            'CO': 'Copper',
            'AL': 'Aluminum',
            'NI': 'Nickel',
            'ZN': 'Zinc',
            'SN': 'Tin',
            'PB': 'Lead',
            'CU': 'Copper_Alt',
            'AU': 'Gold_Alt',
            'AG': 'Silver_Alt',
            'PL': 'Platinum',
            'PA': 'Palladium',
            'RU': 'Ruble',
            'US': 'USD',
            'EU': 'EUR',
            'CN': 'CNY'
        }
        
        # Month codes from the website
        month_codes = {
            'F': 'Январь',
            'G': 'Февраль', 
            'H': 'Март',
            'J': 'Апрель',
            'K': 'Май',
            'M': 'Июнь',
            'N': 'Июль',
            'Q': 'Август',
            'U': 'Сентябрь',
            'V': 'Октябрь',
            'X': 'Ноябрь',
            'Z': 'Декабрь'
        }
        
        # Year codes
        year_codes = {
            '0': '2010',
            '1': '2011',
            '2': '2012',
            '3': '2013',
            '4': '2014',
            '5': '2015',
            '6': '2016',
            '7': '2017',
            '8': '2018',
            '9': '2019',
            'A': '2020',
            'B': '2021',
            'C': '2022',
            'D': '2023',
            'E': '2024',
            'F': '2025',
            'G': '2026',
            'H': '2027',
            'I': '2028',
            'J': '2029',
            'K': '2030'
        }
        
        # Create comprehensive futures table
        futures_table = []
        
        # Generate examples for different asset types
        for asset_code, asset_name in asset_codes.items():
            for year_code, year in year_codes.items():
                for month_code, month_name in month_codes.items():
                    # Create short code example
                    short_code = f"{asset_code}{month_code}{year_code}"
                    
                    # Create long code example (based on website format)
                    long_code = f"{asset_code}-{month_code}.{year_code}"
                    
                    futures_table.append({
                        'asset_code': asset_code,
                        'asset_name': asset_name,
                        'month_code': month_code,
                        'month_name': month_name,
                        'year_code': year_code,
                        'year': year,
                        'short_code': short_code,
                        'long_code': long_code,
                        'example_ticker': short_code
                    })
        
        return {
            'futures_table': pd.DataFrame(futures_table),
            'asset_codes': asset_codes,
            'month_codes': month_codes,
            'year_codes': year_codes
        }
    
    def create_futures_summary(self, futures_data):
        """Create a summary of futures information."""
        if not futures_data:
            return None
            
        summary = {
            'total_contracts': len(futures_data['futures_table']),
            'asset_types': len(futures_data['asset_codes']),
            'months': len(futures_data['month_codes']),
            'years': len(futures_data['year_codes']),
            'asset_breakdown': {},
            'month_breakdown': {},
            'year_breakdown': {}
        }
        
        # Asset type breakdown
        df = futures_data['futures_table']
        asset_counts = df['asset_code'].value_counts()
        for asset_code, count in asset_counts.items():
            asset_name = futures_data['asset_codes'].get(asset_code, 'Unknown')
            summary['asset_breakdown'][asset_code] = {
                'name': asset_name,
                'count': count
            }
        
        # Month breakdown
        month_counts = df['month_code'].value_counts()
        for month_code, count in month_counts.items():
            month_name = futures_data['month_codes'].get(month_code, 'Unknown')
            summary['month_breakdown'][month_code] = {
                'name': month_name,
                'count': count
            }
        
        # Year breakdown
        year_counts = df['year_code'].value_counts()
        for year_code, count in year_counts.items():
            year = futures_data['year_codes'].get(year_code, 'Unknown')
            summary['year_breakdown'][year_code] = {
                'year': year,
                'count': count
            }
        
        return summary
    
    def save_futures_data(self, futures_data, output_dir="data/moex_futures"):
        """Save futures data to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main futures table
        csv_file = output_path / "moex_futures_table.csv"
        futures_data['futures_table'].to_csv(csv_file, index=False)
        print(f"💾 Saved futures table to {csv_file}")
        
        # Save asset codes
        asset_codes_file = output_path / "asset_codes.csv"
        asset_df = pd.DataFrame([
            {'code': code, 'name': name} 
            for code, name in futures_data['asset_codes'].items()
        ])
        asset_df.to_csv(asset_codes_file, index=False)
        print(f"💾 Saved asset codes to {asset_codes_file}")
        
        # Save month codes
        month_codes_file = output_path / "month_codes.csv"
        month_df = pd.DataFrame([
            {'code': code, 'name': name} 
            for code, name in futures_data['month_codes'].items()
        ])
        month_df.to_csv(month_codes_file, index=False)
        print(f"💾 Saved month codes to {month_codes_file}")
        
        # Save year codes
        year_codes_file = output_path / "year_codes.csv"
        year_df = pd.DataFrame([
            {'code': code, 'year': year} 
            for code, year in futures_data['year_codes'].items()
        ])
        year_df.to_csv(year_codes_file, index=False)
        print(f"💾 Saved year codes to {year_codes_file}")
        
        # Create summary report
        summary = self.create_futures_summary(futures_data)
        if summary:
            summary_file = output_path / "futures_summary.md"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("# MOEX Futures Summary\n\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("## Overview\n\n")
                f.write(f"- Total contracts: {summary['total_contracts']}\n")
                f.write(f"- Asset types: {summary['asset_types']}\n")
                f.write(f"- Months: {summary['months']}\n")
                f.write(f"- Years: {summary['years']}\n\n")
                
                f.write("## Asset Types\n\n")
                for code, info in summary['asset_breakdown'].items():
                    f.write(f"- **{code}** ({info['name']}): {info['count']} contracts\n")
                
                f.write("\n## Months\n\n")
                for code, info in summary['month_breakdown'].items():
                    f.write(f"- **{code}** ({info['name']}): {info['count']} contracts\n")
                
                f.write("\n## Years\n\n")
                for code, info in summary['year_breakdown'].items():
                    f.write(f"- **{code}** ({info['year']}): {info['count']} contracts\n")
            
            print(f"💾 Saved summary report to {summary_file}")
        
        return {
            'csv_file': str(csv_file),
            'asset_codes_file': str(asset_codes_file),
            'month_codes_file': str(month_codes_file),
            'year_codes_file': str(year_codes_file),
            'summary_file': str(summary_file) if summary else None
        }

def main():
    """Main function to scrape and save MOEX futures data."""
    print("🚀 Starting MOEX futures scraping...")
    print("=" * 60)
    
    scraper = MoexFuturesScraper()
    
    # Scrape futures information
    futures_data = scraper.scrape_futures_info()
    
    if futures_data:
        print(f"✅ Successfully scraped futures data")
        print(f"📊 Total contracts: {len(futures_data['futures_table'])}")
        print(f"📈 Asset types: {len(futures_data['asset_codes'])}")
        
        # Display sample data
        print(f"\n📋 Sample futures contracts:")
        sample_df = futures_data['futures_table'].head(10)
        print(sample_df[['asset_code', 'asset_name', 'short_code', 'long_code']].to_string(index=False))
        
        # Save data
        saved_files = scraper.save_futures_data(futures_data)
        
        print(f"\n🎉 Successfully saved MOEX futures data!")
        print(f"📁 Output directory: data/moex_futures/")
        print(f"📄 Files created: {len(saved_files)}")
        
        # Show summary
        summary = scraper.create_futures_summary(futures_data)
        if summary:
            print(f"\n📊 Summary:")
            print(f"   Total contracts: {summary['total_contracts']}")
            print(f"   Asset types: {summary['asset_types']}")
            print(f"   Months: {summary['months']}")
            print(f"   Years: {summary['years']}")
            
            print(f"\n📈 Top asset types:")
            for code, info in list(summary['asset_breakdown'].items())[:5]:
                print(f"   {code} ({info['name']}): {info['count']} contracts")
        
    else:
        print("❌ Failed to scrape futures data")

if __name__ == "__main__":
    main() 