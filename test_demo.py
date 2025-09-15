#!/usr/bin/env python3
"""
Test script for the demo application
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    try:
        import flask
        import pandas
        import joblib
        print("✓ All required packages can be imported")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_app_creation():
    """Test if the Flask app can be created"""
    try:
        from app import app
        print("✓ Flask app can be created")
        return True
    except Exception as e:
        print(f"✗ App creation error: {e}")
        return False

def test_data_files():
    """Test if data files exist"""
    data_files = [
        "data/processed/auto_annotated_large.csv",
        "data/processed/utterances_full.jsonl"
    ]
    
    all_exist = True
    for file_path in data_files:
        if Path(file_path).exists():
            print(f"✓ {file_path} exists")
        else:
            print(f"⚠ {file_path} not found (demo will use mock data)")
            all_exist = False
    
    return all_exist

def test_models():
    """Test if models exist"""
    model_dirs = [
        "experiments/runs/baseline_auto_annotated",
        "experiments/runs/sentence_transformer"
    ]
    
    found_models = []
    for model_dir in model_dirs:
        if Path(model_dir).exists():
            print(f"✓ {model_dir} exists")
            found_models.append(model_dir)
        else:
            print(f"⚠ {model_dir} not found (demo will use mock predictions)")
    
    return len(found_models) > 0

def main():
    """Run all tests"""
    print("Testing Demo Application...")
    print("=" * 40)
    
    tests = [
        ("Package Imports", test_imports),
        ("App Creation", test_app_creation),
        ("Data Files", test_data_files),
        ("Trained Models", test_models)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append(result)
    
    print("\n" + "=" * 40)
    print("Test Summary:")
    
    if all(results):
        print("✓ All tests passed! Demo should work perfectly.")
    else:
        print("⚠ Some tests failed, but demo should still work with limited functionality.")
    
    print("\nTo start the demo:")
    print("  python run_demo.py")
    print("  Then visit: http://localhost:8080")

if __name__ == "__main__":
    main()
