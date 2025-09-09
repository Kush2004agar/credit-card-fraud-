#!/usr/bin/env python3
"""
Simple Project Test Script
==========================

This script tests the project structure without requiring external dependencies.
"""

import os
import sys
from pathlib import Path

def test_project_structure():
    """Test if the project structure is correct"""
    print("🔍 Testing Credit Card Fraud Detection Project Structure")
    print("=" * 60)
    
    # Check directories
    required_dirs = [
        'data',
        'notebooks',
        'src',
        'src/data',
        'src/models',
        'src/visualization',
        'src/utils',
        'models',
        'results',
        'docs',
        'tests',
        'scripts',
        'logs'
    ]
    
    print("📁 Checking directory structure...")
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory}/")
        else:
            print(f"❌ {directory}/ - MISSING")
    
    print()
    
    # Check key files
    required_files = [
        'data/creditcard.csv',
        'src/main.py',
        'src/dashboard.py',
        'src/api.py',
        'requirements.txt',
        'README.md',
        'config.py',
        'showcase.py'
    ]
    
    print("📋 Checking key files...")
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({size:,} bytes)")
        else:
            print(f"❌ {file_path} - MISSING")
    
    print()
    
    # Check Python files
    python_files = [
        'src/main.py',
        'src/dashboard.py',
        'src/api.py',
        'showcase.py'
    ]
    
    print("🐍 Checking Python files...")
    for py_file in python_files:
        if os.path.exists(py_file):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = len(content.split('\n'))
                    print(f"✅ {py_file} ({lines} lines)")
            except Exception as e:
                print(f"⚠️ {py_file} - Error reading: {e}")
        else:
            print(f"❌ {py_file} - MISSING")
    
    print()
    
    # Check data file
    print("📊 Checking dataset...")
    data_file = 'data/creditcard.csv'
    if os.path.exists(data_file):
        size = os.path.getsize(data_file)
        size_mb = size / (1024 * 1024)
        print(f"✅ Dataset found: {size_mb:.1f} MB")
        
        # Try to read first few lines
        try:
            with open(data_file, 'r') as f:
                first_line = f.readline().strip()
                if first_line:
                    print(f"✅ Dataset readable, first line: {first_line[:50]}...")
                else:
                    print("⚠️ Dataset appears to be empty")
        except Exception as e:
            print(f"⚠️ Error reading dataset: {e}")
    else:
        print("❌ Dataset not found")
    
    print()
    
    # Summary
    print("🎯 Project Status Summary")
    print("=" * 40)
    
    # Count existing vs missing
    existing_dirs = sum(1 for d in required_dirs if os.path.exists(d))
    existing_files = sum(1 for f in required_files if os.path.exists(f))
    
    print(f"📁 Directories: {existing_dirs}/{len(required_dirs)} ✅")
    print(f"📋 Files: {existing_files}/{len(required_files)} ✅")
    
    if existing_dirs == len(required_dirs) and existing_files == len(required_files):
        print("\n🎉 All project components are present!")
        print("🚀 Your project is ready for showcase!")
    else:
        print(f"\n⚠️ {len(required_dirs) - existing_dirs} directories and {len(required_files) - existing_files} files are missing.")
        print("🔧 Please run the setup script again or check for errors.")
    
    print()
    
    # Next steps
    print("🚀 Next Steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Test the system: python showcase.py")
    print("3. Run analysis: python src/main.py")
    print("4. Launch dashboard: streamlit run src/dashboard.py")
    print("5. Start API: python src/api.py")

def test_basic_functionality():
    """Test basic functionality without external dependencies"""
    print("🧪 Testing Basic Functionality")
    print("=" * 40)
    
    # Test if we can read the main script
    main_script = 'src/main.py'
    if os.path.exists(main_script):
        try:
            with open(main_script, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Check for key components
                checks = [
                    ("CreditCardFraudDetector class", "class CreditCardFraudDetector"),
                    ("load_data method", "def load_data"),
                    ("train_models method", "def train_models"),
                    ("evaluate_models method", "def evaluate_models"),
                    ("Machine learning imports", "from sklearn"),
                    ("Pandas import", "import pandas"),
                    ("Numpy import", "import numpy")
                ]
                
                print("🔍 Checking code components...")
                for check_name, check_string in checks:
                    if check_string in content:
                        print(f"✅ {check_name}")
                    else:
                        print(f"❌ {check_name}")
                
        except Exception as e:
            print(f"❌ Error reading main script: {e}")
    else:
        print("❌ Main script not found")
    
    print()

def main():
    """Main test function"""
    print("🚀 Credit Card Fraud Detection Project - Structure Test")
    print("=" * 70)
    print()
    
    # Test project structure
    test_project_structure()
    
    # Test basic functionality
    test_basic_functionality()
    
    print("🎯 Test completed!")
    print("💡 Check the results above to ensure your project is ready.")

if __name__ == "__main__":
    main()
