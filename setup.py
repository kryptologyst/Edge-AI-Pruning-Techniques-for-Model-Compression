#!/usr/bin/env python3
"""Setup script for Edge AI Pruning project."""

import os
import subprocess
import sys
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 10):
        print("Error: Python 3.10 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✓ Python version: {sys.version.split()[0]}")


def create_directories():
    """Create necessary directories."""
    directories = [
        "data/raw",
        "data/processed", 
        "checkpoints",
        "logs",
        "assets",
        "exports",
        "tests",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def install_dependencies():
    """Install project dependencies."""
    print("Installing dependencies...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        sys.exit(1)


def run_basic_tests():
    """Run basic tests to verify installation."""
    print("Running basic tests...")
    
    try:
        subprocess.run([sys.executable, "test_basic.py"], check=True)
        print("✓ Basic tests passed")
    except subprocess.CalledProcessError as e:
        print(f"✗ Basic tests failed: {e}")
        print("You may need to install additional dependencies")
        return False
    
    return True


def main():
    """Main setup function."""
    print("=" * 60)
    print("EDGE AI PRUNING PROJECT SETUP")
    print("=" * 60)
    
    # Check Python version
    check_python_version()
    
    # Create directories
    print("\nCreating directories...")
    create_directories()
    
    # Install dependencies
    print("\nInstalling dependencies...")
    install_dependencies()
    
    # Run basic tests
    print("\nRunning basic tests...")
    tests_passed = run_basic_tests()
    
    print("\n" + "=" * 60)
    if tests_passed:
        print("✓ SETUP COMPLETED SUCCESSFULLY!")
        print("\nNext steps:")
        print("1. Run training: python train.py")
        print("2. Launch demo: python run_demo.py")
        print("3. Analyze results: python scripts/analyze_results.py analyze assets/results_summary.json")
    else:
        print("⚠ SETUP COMPLETED WITH WARNINGS")
        print("Some tests failed. Check the output above for details.")
    
    print("\nFor more information, see README.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
