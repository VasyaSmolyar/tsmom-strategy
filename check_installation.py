#!/usr/bin/env python3
"""
Installation check script for TSMOM Backtest Project.
Run this script to verify that all dependencies are installed correctly.
"""

import sys
import importlib
from pathlib import Path

def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ required.")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True

def check_dependencies():
    """Check required dependencies."""
    print("\nChecking dependencies...")
    
    required_packages = [
        'pandas',
        'numpy', 
        'matplotlib',
        'seaborn',
        'yfinance',
        'scipy',
        'statsmodels',
        'pyyaml',
        'pytest'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} - OK")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install missing packages with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies installed successfully!")
        return True

def check_project_structure():
    """Check project structure."""
    print("\nChecking project structure...")
    
    required_files = [
        'config/config.yaml',
        'src/main.py',
        'src/data/data_loader.py',
        'src/strategy/tsmom_strategy.py',
        'src/analysis/performance_analyzer.py',
        'src/utils/helpers.py',
        'requirements.txt',
        'README.md',
        'run_backtest.py'
    ]
    
    required_dirs = [
        'data',
        'data/raw',
        'data/processed',
        'src',
        'src/data',
        'src/strategy',
        'src/analysis',
        'src/utils',
        'notebooks',
        'reports',
        'tests',
        'config',
        'docs'
    ]
    
    missing_files = []
    missing_dirs = []
    
    # Check files
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
            print(f"❌ {file_path} - MISSING")
        else:
            print(f"✅ {file_path} - OK")
    
    # Check directories
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
            print(f"❌ {dir_path}/ - MISSING")
        else:
            print(f"✅ {dir_path}/ - OK")
    
    if missing_files or missing_dirs:
        print(f"\n❌ Missing files: {missing_files}")
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("\n✅ Project structure is complete!")
        return True

def check_imports():
    """Check if project modules can be imported."""
    print("\nChecking module imports...")
    
    # Add src to path
    sys.path.append(str(Path(__file__).parent / "src"))
    
    modules_to_test = [
        'data.data_loader',
        'strategy.tsmom_strategy', 
        'analysis.performance_analyzer',
        'utils.helpers'
    ]
    
    failed_imports = []
    
    for module in modules_to_test:
        try:
            importlib.import_module(module)
            print(f"✅ {module} - OK")
        except ImportError as e:
            print(f"❌ {module} - FAILED: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed imports: {failed_imports}")
        return False
    else:
        print("\n✅ All modules can be imported successfully!")
        return True

def test_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")
    
    try:
        # Test configuration loading
        from src.utils.helpers import load_config
        config = load_config('config/config.yaml')
        print("✅ Configuration loading - OK")
        
        # Test data loader initialization
        from src.data.data_loader import DataLoader
        loader = DataLoader()
        print("✅ Data loader initialization - OK")
        
        # Test strategy initialization
        from src.strategy.tsmom_strategy import TSMOMStrategy
        strategy = TSMOMStrategy()
        print("✅ Strategy initialization - OK")
        
        # Test analyzer initialization
        from src.analysis.performance_analyzer import PerformanceAnalyzer
        analyzer = PerformanceAnalyzer()
        print("✅ Performance analyzer initialization - OK")
        
        print("\n✅ Basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Basic functionality test failed: {e}")
        return False

def main():
    """Run all checks."""
    print("TSMOM Backtest Project - Installation Check")
    print("=" * 50)
    
    checks = [
        check_python_version,
        check_dependencies,
        check_project_structure,
        check_imports,
        test_basic_functionality
    ]
    
    results = []
    for check in checks:
        results.append(check())
    
    print("\n" + "=" * 50)
    print("INSTALLATION CHECK SUMMARY")
    print("=" * 50)
    
    if all(results):
        print("🎉 ALL CHECKS PASSED!")
        print("\n✅ Your TSMOM Backtest Project is ready to use!")
        print("\nNext steps:")
        print("1. Run the backtest: python run_backtest.py")
        print("2. Explore the Jupyter notebook: jupyter notebook notebooks/tsmom_analysis.ipynb")
        print("3. Read the documentation: docs/USAGE.md")
    else:
        print("❌ SOME CHECKS FAILED!")
        print("\nPlease fix the issues above before proceeding.")
        print("\nCommon solutions:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Check Python version: python --version")
        print("3. Verify project structure")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 