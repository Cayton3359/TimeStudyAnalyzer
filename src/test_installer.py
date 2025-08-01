"""
Simple launcher for Time Study Analyzer Installer
Run this to test the installer without building an exe
"""

import sys
import os

# Add current directory to path so we can import the installer
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import and run the installer
from one_click_installer import main

if __name__ == "__main__":
    print("Starting Time Study Analyzer Installer...")
    main()
