#!/usr/bin/env python3
"""Script to run the Streamlit demo."""

import subprocess
import sys
from pathlib import Path


def main():
    """Run the Streamlit demo."""
    demo_path = Path(__file__).parent / "demo" / "app.py"
    
    if not demo_path.exists():
        print(f"Demo file not found: {demo_path}")
        sys.exit(1)
    
    print("Starting Edge AI Pruning Demo...")
    print("The demo will open in your default web browser.")
    print("Press Ctrl+C to stop the demo.")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(demo_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\nDemo stopped by user.")
    except Exception as e:
        print(f"Error running demo: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
