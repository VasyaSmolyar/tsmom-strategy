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

def test_quarterly_futures_2022_2025():
    """Test loading available futures by quarters for 2022-2025 years."""
    print("\nTesting quarterly futures availability for 2022-2025...")
    print("=" * 70)
    
    try:
        loader = MoexLoader()
        
        # Define quarters for 2022-2025
        quarters = {
            "2022": [
                ("2022-01-01", "2022-03-31", "Q1 2022"),
                ("2022-04-01", "2022-06-30", "Q2 2022"),
                ("2022-07-01", "2022-09-30", "Q3 2022"),
                ("2022-10-01", "2022-12-31", "Q4 2022")
            ],
            "2023": [
                ("2023-01-01", "2023-03-31", "Q1 2023"),
                ("2023-04-01", "2023-06-30", "Q2 2023"),
                ("2023-07-01", "2023-09-30", "Q3 2023"),
                ("2023-10-01", "2023-12-31", "Q4 2023")
            ],
            "2024": [
                ("2024-01-01", "2024-03-31", "Q1 2024"),
                ("2024-04-01", "2024-06-30", "Q2 2024"),
                ("2024-07-01", "2024-09-30", "Q3 2024"),
                ("2024-10-01", "2024-12-31", "Q4 2024")
            ],
            "2025": [
                ("2025-01-01", "2025-03-31", "Q1 2025"),
                ("2025-04-01", "2025-06-30", "Q2 2025"),
                ("2025-07-01", "2025-09-30", "Q3 2025"),
                ("2025-10-01", "2025-12-31", "Q4 2025")
            ]
        }
        
        # Test different asset types
        asset_types = ['BR', 'SI', 'GD', 'CL', 'NG']  # Brent, Silver, Gold, Crude Oil, Natural Gas
        
        results_summary = {}
        
        for year, year_quarters in quarters.items():
            print(f"\n📅 {year} год:")
            print("-" * 40)
            
            year_results = {}
            
            for start_date, end_date, quarter_name in year_quarters:
                print(f"\n🔍 {quarter_name} ({start_date} - {end_date}):")
                
                quarter_results = {}
                
                for asset_type in asset_types:
                    try:
                        # Get historical futures available during this quarter
                        futures_df = loader.get_historical_futures_for_period(start_date, end_date, asset_type)
                        
                        if not futures_df.empty:
                            # Count active and expired contracts
                            active_count = futures_df['is_active'].sum() if 'is_active' in futures_df.columns else 0
                            total_count = len(futures_df)
                            expired_count = total_count - active_count
                            
                            print(f"   ✅ {asset_type}: {total_count} контрактов (активных: {active_count}, истекших: {expired_count})")
                            
                            # Show sample tickers
                            if 'ticker' in futures_df.columns:
                                sample_tickers = futures_df['ticker'].head(5).tolist()
                                print(f"      Примеры: {', '.join(sample_tickers)}")
                            
                            # Show expiration dates for better understanding
                            if 'expiration_date' in futures_df.columns:
                                exp_dates = futures_df['expiration_date'].dt.strftime('%Y-%m').value_counts().head(3)
                                print(f"      Истечения: {', '.join([f'{date}({count})' for date, count in exp_dates.items()])}")
                            
                            quarter_results[asset_type] = {
                                'total': total_count,
                                'active': active_count,
                                'expired': expired_count,
                                'tickers': futures_df['ticker'].tolist() if 'ticker' in futures_df.columns else []
                            }
                        else:
                            print(f"   ❌ {asset_type}: нет данных")
                            quarter_results[asset_type] = {
                                'total': 0,
                                'active': 0,
                                'expired': 0,
                                'tickers': []
                            }
                            
                    except Exception as e:
                        print(f"   ❌ {asset_type}: ошибка - {e}")
                        quarter_results[asset_type] = {
                            'total': 0,
                            'active': 0,
                            'expired': 0,
                            'tickers': [],
                            'error': str(e)
                        }
                
                year_results[quarter_name] = quarter_results
            
            results_summary[year] = year_results
        
        # Print summary statistics
        print("\n" + "=" * 70)
        print("📊 СВОДНАЯ СТАТИСТИКА ПО КВАРТАЛАМ 2022-2025:")
        print("=" * 70)
        
        for year, year_data in results_summary.items():
            print(f"\n📅 {year} год:")
            print("-" * 40)
            
            for quarter, quarter_data in year_data.items():
                print(f"\n🔍 {quarter}:")
                
                total_contracts = sum(data['total'] for data in quarter_data.values())
                total_active = sum(data['active'] for data in quarter_data.values())
                total_expired = sum(data['expired'] for data in quarter_data.values())
                
                print(f"   📈 Всего контрактов: {total_contracts}")
                print(f"   📊 Активных контрактов: {total_active}")
                print(f"   📉 Истекших контрактов: {total_expired}")
                
                # Show breakdown by asset type
                for asset_type, data in quarter_data.items():
                    if data['total'] > 0:
                        print(f"      {asset_type}: {data['total']} ({data['active']} активных, {data['expired']} истекших)")
        
        return results_summary
        
    except Exception as e:
        print(f"❌ Ошибка в тесте квартальных фьючерсов: {e}")
        import traceback
        traceback.print_exc()
        return {}

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

def test_historical_vs_active_futures():
    """Test comparison between historical and active futures methods."""
    print("\nTesting historical vs active futures methods...")
    print("=" * 70)
    
    try:
        loader = MoexLoader()
        
        # Test periods
        test_periods = [
            ("2023-01-01", "2023-03-31", "Q1 2023"),
            ("2024-01-01", "2024-03-31", "Q1 2024"),
            ("2025-01-01", "2025-03-31", "Q1 2025")
        ]
        
        asset_types = ['BR', 'GD', 'NG']
        
        for start_date, end_date, period_name in test_periods:
            print(f"\n🔍 {period_name} ({start_date} - {end_date}):")
            print("-" * 50)
            
            for asset_type in asset_types:
                print(f"\n📊 {asset_type} futures:")
                
                # Test historical method
                historical_futures = loader.get_historical_futures_for_period(start_date, end_date, asset_type)
                historical_count = len(historical_futures)
                
                # Test active method (old approach)
                active_futures = loader.get_futures_with_data(start_date, end_date, asset_type)
                active_count = len(active_futures)
                
                print(f"   📈 Исторических контрактов: {historical_count}")
                print(f"   📊 Активных контрактов: {active_count}")
                
                if historical_count > 0:
                    # Show some details about historical contracts
                    if 'ticker' in historical_futures.columns:
                        sample_tickers = historical_futures['ticker'].head(3).tolist()
                        print(f"      Примеры исторических: {', '.join(sample_tickers)}")
                    
                    if 'expiration_date' in historical_futures.columns:
                        # Show expiration date range
                        min_exp = historical_futures['expiration_date'].min()
                        max_exp = historical_futures['expiration_date'].max()
                        print(f"      Диапазон истечений: {min_exp.strftime('%Y-%m')} - {max_exp.strftime('%Y-%m')}")
                
                if active_count > 0:
                    if 'ticker' in active_futures.columns:
                        sample_tickers = active_futures['ticker'].head(3).tolist()
                        print(f"      Примеры активных: {', '.join(sample_tickers)}")
                
                print(f"   📊 Разница: {historical_count - active_count} дополнительных исторических контрактов")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в тесте сравнения методов: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_historical_futures_download():
    """Test downloading historical futures data directly."""
    print("\nTesting historical futures data download...")
    print("=" * 70)
    
    try:
        loader = MoexLoader()
        
        # Test historical futures that might have existed
        # These are examples of what futures might have been available in the past
        historical_futures_to_test = {
            "2022": {
                "BR": ["BRH2", "BRJ2", "BRK2", "BRM2", "BRN2", "BRQ2", "BRU2", "BRV2", "BRX2", "BRZ2"],
                "GD": ["GDH2", "GDJ2", "GDK2", "GDM2", "GDN2", "GDQ2", "GDU2", "GDV2", "GDX2", "GDZ2"],
                "NG": ["NGH2", "NGJ2", "NGK2", "NGM2", "NGN2", "NGQ2", "NGU2", "NGV2", "NGX2", "NGZ2"]
            },
            "2023": {
                "BR": ["BRH3", "BRJ3", "BRK3", "BRM3", "BRN3", "BRQ3", "BRU3", "BRV3", "BRX3", "BRZ3"],
                "GD": ["GDH3", "GDJ3", "GDK3", "GDM3", "GDN3", "GDQ3", "GDU3", "GDV3", "GDX3", "GDZ3"],
                "NG": ["NGH3", "NGJ3", "NGK3", "NGM3", "NGN3", "NGQ3", "NGU3", "NGV3", "NGX3", "NGZ3"]
            },
            "2024": {
                "BR": ["BRH4", "BRJ4", "BRK4", "BRM4", "BRN4", "BRQ4", "BRU4", "BRV4", "BRX4", "BRZ4"],
                "GD": ["GDH4", "GDJ4", "GDK4", "GDM4", "GDN4", "GDQ4", "GDU4", "GDV4", "GDX4", "GDZ4"],
                "NG": ["NGH4", "NGJ4", "NGK4", "NGM4", "NGN4", "NGQ4", "NGU4", "NGV4", "NGX4", "NGZ4"]
            }
        }
        
        results = {}
        
        for year, asset_types in historical_futures_to_test.items():
            print(f"\n📅 {year} год:")
            print("-" * 40)
            
            year_results = {}
            
            for asset_type, tickers in asset_types.items():
                print(f"\n🔍 {asset_type} futures:")
                
                asset_results = {
                    'available': [],
                    'with_data': [],
                    'no_data': []
                }
                
                # Test first 3 tickers to avoid too many API calls
                test_tickers = tickers[:3]
                
                for ticker in test_tickers:
                    try:
                        # Try to get data for a specific period in that year
                        if year == "2022":
                            start_date = "2022-01-01"
                            end_date = "2022-01-31"
                        elif year == "2023":
                            start_date = "2023-01-01"
                            end_date = "2023-01-31"
                        elif year == "2024":
                            start_date = "2024-01-01"
                            end_date = "2024-01-31"
                        
                        print(f"   🔍 Testing {ticker} for {start_date} to {end_date}...")
                        
                        data = loader._get_futures_data(ticker, start_date, end_date)
                        
                        if not data.empty:
                            print(f"      ✅ {ticker}: {len(data)} записей, цена: ${data['close'].min():.2f}-${data['close'].max():.2f}")
                            asset_results['with_data'].append(ticker)
                        else:
                            print(f"      ❌ {ticker}: нет данных")
                            asset_results['no_data'].append(ticker)
                            
                    except Exception as e:
                        print(f"      ❌ {ticker}: ошибка - {e}")
                        asset_results['no_data'].append(ticker)
                
                year_results[asset_type] = asset_results
                
                # Summary for this asset type
                total_tested = len(test_tickers)
                with_data = len(asset_results['with_data'])
                no_data = len(asset_results['no_data'])
                
                print(f"   📊 {asset_type}: {with_data}/{total_tested} с данными")
            
            results[year] = year_results
        
        # Print summary
        print("\n" + "=" * 70)
        print("📊 СВОДКА ПО ИСТОРИЧЕСКИМ ДАННЫМ:")
        print("=" * 70)
        
        for year, year_data in results.items():
            print(f"\n📅 {year} год:")
            for asset_type, asset_data in year_data.items():
                with_data = len(asset_data['with_data'])
                no_data = len(asset_data['no_data'])
                total = with_data + no_data
                
                print(f"   {asset_type}: {with_data}/{total} контрактов с данными")
                
                if asset_data['with_data']:
                    print(f"      Доступные: {', '.join(asset_data['with_data'])}")
        
        return results
        
    except Exception as e:
        print(f"❌ Ошибка в тесте загрузки исторических данных: {e}")
        import traceback
        traceback.print_exc()
        return {}

def test_save_historical_futures_data():
    """Test saving historical futures data by days."""
    print("\nTesting saving historical futures data by days...")
    print("=" * 70)
    
    try:
        loader = MoexLoader()
        
        # Define historical periods to test
        historical_periods = {
            "2022": {
                "BR": ["BRH2", "BRJ2", "BRK2"],
                "GD": ["GDH2"],
                "NG": ["NGH2", "NGJ2", "NGK2"]
            },
            "2023": {
                "BR": ["BRH3", "BRJ3", "BRK3"],
                "GD": ["GDH3"],
                "NG": ["NGH3", "NGJ3", "NGK3"]
            },
            "2024": {
                "BR": ["BRH4", "BRJ4", "BRK4"],
                "GD": ["GDH4"],
                "NG": ["NGH4", "NGJ4", "NGK4"]
            }
        }
        
        # Create data directory if it doesn't exist
        data_dir = Path("data/historical_futures")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        for year, asset_types in historical_periods.items():
            print(f"\n📅 {year} год:")
            print("-" * 40)
            
            year_dir = data_dir / str(year)
            year_dir.mkdir(exist_ok=True)
            
            for asset_type, tickers in asset_types.items():
                print(f"\n🔍 {asset_type} futures:")
                
                asset_dir = year_dir / asset_type
                asset_dir.mkdir(exist_ok=True)
                
                for ticker in tickers:
                    try:
                        # Define period for this ticker (January of the year)
                        start_date = f"{year}-01-01"
                        end_date = f"{year}-01-31"
                        
                        print(f"   🔍 Загружаю {ticker} для {start_date} - {end_date}...")
                        
                        # Download data
                        data = loader._get_futures_data(ticker, start_date, end_date)
                        
                        if not data.empty:
                            # Save to CSV
                            csv_file = asset_dir / f"{ticker}_daily.csv"
                            data.to_csv(csv_file)
                            
                            # Save summary info
                            summary_file = asset_dir / f"{ticker}_summary.txt"
                            with open(summary_file, 'w', encoding='utf-8') as f:
                                f.write(f"Ticker: {ticker}\n")
                                f.write(f"Asset Type: {asset_type}\n")
                                f.write(f"Year: {year}\n")
                                f.write(f"Period: {start_date} to {end_date}\n")
                                f.write(f"Records: {len(data)}\n")
                                f.write(f"Date Range: {data.index.min()} to {data.index.max()}\n")
                                f.write(f"Price Range: ${data['close'].min():.2f} - ${data['close'].max():.2f}\n")
                                f.write(f"Average Volume: {data['volume'].mean():.0f}\n")
                                f.write(f"Columns: {list(data.columns)}\n")
                            
                            saved_files.append({
                                'ticker': ticker,
                                'asset_type': asset_type,
                                'year': year,
                                'csv_file': str(csv_file),
                                'summary_file': str(summary_file),
                                'records': len(data),
                                'price_range': f"${data['close'].min():.2f}-${data['close'].max():.2f}"
                            })
                            
                            print(f"      ✅ {ticker}: {len(data)} записей, цена: ${data['close'].min():.2f}-${data['close'].max():.2f}")
                            print(f"      💾 Сохранено в: {csv_file}")
                            
                        else:
                            print(f"      ❌ {ticker}: нет данных")
                            
                    except Exception as e:
                        print(f"      ❌ {ticker}: ошибка - {e}")
        
        # Create summary report
        summary_report = data_dir / "historical_data_summary.md"
        with open(summary_report, 'w', encoding='utf-8') as f:
            f.write("# Исторические данные фьючерсов\n\n")
            f.write(f"Дата создания: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Сводка по годам\n\n")
            
            for year in historical_periods.keys():
                year_files = [f for f in saved_files if f['year'] == year]
                f.write(f"### {year} год\n\n")
                
                for asset_type in ['BR', 'GD', 'NG']:
                    asset_files = [f for f in year_files if f['asset_type'] == asset_type]
                    if asset_files:
                        f.write(f"#### {asset_type} ({len(asset_files)} контрактов)\n\n")
                        for file_info in asset_files:
                            f.write(f"- **{file_info['ticker']}**: {file_info['records']} записей, {file_info['price_range']}\n")
                        f.write("\n")
            
            f.write("## Статистика\n\n")
            f.write(f"- Всего сохранено файлов: {len(saved_files)}\n")
            f.write(f"- Общее количество записей: {sum(f['records'] for f in saved_files)}\n")
            
            # Asset type statistics
            asset_stats = {}
            for file_info in saved_files:
                asset_type = file_info['asset_type']
                if asset_type not in asset_stats:
                    asset_stats[asset_type] = {'count': 0, 'records': 0}
                asset_stats[asset_type]['count'] += 1
                asset_stats[asset_type]['records'] += file_info['records']
            
            f.write("### По типам активов\n\n")
            for asset_type, stats in asset_stats.items():
                f.write(f"- **{asset_type}**: {stats['count']} контрактов, {stats['records']} записей\n")
        
        print(f"\n📊 СВОДКА СОХРАНЕННЫХ ДАННЫХ:")
        print("=" * 50)
        print(f"📁 Директория: {data_dir}")
        print(f"📄 Всего файлов: {len(saved_files)}")
        print(f"📊 Общее количество записей: {sum(f['records'] for f in saved_files)}")
        
        # Show statistics by asset type
        asset_stats = {}
        for file_info in saved_files:
            asset_type = file_info['asset_type']
            if asset_type not in asset_stats:
                asset_stats[asset_type] = {'count': 0, 'records': 0}
            asset_stats[asset_type]['count'] += 1
            asset_stats[asset_type]['records'] += file_info['records']
        
        print(f"\n📈 По типам активов:")
        for asset_type, stats in asset_stats.items():
            print(f"   {asset_type}: {stats['count']} контрактов, {stats['records']} записей")
        
        print(f"\n📋 Отчет сохранен в: {summary_report}")
        
        return saved_files
        
    except Exception as e:
        print(f"❌ Ошибка в сохранении исторических данных: {e}")
        import traceback
        traceback.print_exc()
        return []

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
    test_quarterly_futures_2022_2025()
    test_historical_vs_active_futures()
    test_historical_futures_download()
    test_save_historical_futures_data()
    
    print("\n" + "=" * 60)
    print("🎉 All futures information tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    main() 