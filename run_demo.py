#!/usr/bin/env python3
"""
Parliamentary Debate Analysis - Demo Web Application
CSC 4792 - Team 16

This script starts the web application for demonstrating the motion-utterance
classification system.
"""

import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import flask
        import pandas
        import joblib
        print("✓ Required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def check_data_files():
    """Check if required data files exist"""
    required_files = [
        "data/processed/auto_annotated_large.csv",
        "data/processed/utterances_full.jsonl"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("✗ Missing required data files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease run the data preparation pipeline first:")
        print("  python -m src.parse.segment --in data/raw/ --order-papers-dir data/interim/ --out data/processed/")
        print("  python -m src.label.auto_annotate --in data/processed/utterances_full.jsonl --out data/processed/auto_annotated_large.csv")
        return False
    
    print("✓ Required data files found")
    return True

def check_models():
    """Check if trained models exist"""
    model_dirs = [
        "experiments/runs/baseline_auto_annotated",
        "experiments/runs/sentence_transformer"
    ]
    
    found_models = []
    for model_dir in model_dirs:
        if Path(model_dir).exists():
            found_models.append(model_dir)
    
    if not found_models:
        print("⚠ No trained models found")
        print("The demo will work with mock predictions")
        print("To train models, run:")
        print("  python -m src.models.train_baselines --in data/processed/ --out experiments/runs/baseline/")
        print("  python -m src.models.train_sentence_transformer --in data/processed/ --out experiments/runs/sentence_transformer/")
    else:
        print(f"✓ Found trained models in: {', '.join(found_models)}")
    
    return True

def main():
    """Main function to start the demo"""
    print("=" * 60)
    print("Parliamentary Debate Analysis - Demo Application")
    print("CSC 4792 - Team 16")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check data files
    if not check_data_files():
        sys.exit(1)
    
    # Check models
    check_models()
    
    print("\n" + "=" * 60)
    print("Starting web application...")
    print("=" * 60)
    print("The demo will be available at: http://localhost:8080")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Start the Flask application
        from app import app
        app.run(debug=True, host='0.0.0.0', port=8080)
    except KeyboardInterrupt:
        print("\n\nDemo stopped by user")
    except Exception as e:
        print(f"\nError starting demo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
