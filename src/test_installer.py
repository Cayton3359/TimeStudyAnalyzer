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
try:
    # Try direct import first
    from one_click_installer import main
except ImportError:
    try:
        # Try importing as module from file path
        import importlib.util
        installer_path = os.path.join(current_dir, "one_click_installer.py")
        if os.path.exists(installer_path):
            spec = importlib.util.spec_from_file_location("one_click_installer", installer_path)
            installer_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(installer_module)
            main = installer_module.main
        else:
            raise ImportError("one_click_installer.py not found")
    except Exception as e:
        print(f"Error: Could not import 'one_click_installer.py': {e}")
        print(f"Current directory: {current_dir}")
        print("Files in current directory:")
        for file in os.listdir(current_dir):
            print(f"  {file}")
        sys.exit(1)

if __name__ == "__main__":
    print("Starting Time Study Analyzer Installer...")
    main()
