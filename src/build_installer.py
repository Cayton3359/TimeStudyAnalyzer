"""
Build script to create executable installer for Time Study Analyzer
This creates a standalone .exe file that users can run to install everything
"""

import os
import subprocess
import sys

def install_pyinstaller():
    """Install PyInstaller if not already installed"""
    try:
        import PyInstaller
        print("✓ PyInstaller already installed")
        return True
    except ImportError:
        print("Installing PyInstaller...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
            print("✓ PyInstaller installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("✗ Failed to install PyInstaller")
            return False

def build_installer():
    """Build the installer executable"""
    if not install_pyinstaller():
        return False
    
    # PyInstaller command to create a single executable
    cmd = [
        "pyinstaller",
        "--onefile",                    # Create single .exe file
        "--windowed",                   # No console window (GUI only)
        "--name=TimeStudyAnalyzer_Installer",  # Output filename
        "--icon=NONE",                  # No icon for now
        "--add-data=one_click_installer.py;.",  # Include the installer script
        "one_click_installer.py"
    ]
    
    try:
        print("Building installer executable...")
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, cwd=os.path.dirname(__file__), 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Installer executable created successfully!")
            print("Check the 'dist' folder for TimeStudyAnalyzer_Installer.exe")
            return True
        else:
            print(f"✗ Build failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Error building installer: {e}")
        return False

def main():
    print("=== Time Study Analyzer Installer Builder ===")
    print("This will create a standalone installer executable")
    print()
    
    if build_installer():
        print()
        print("SUCCESS!")
        print("The installer executable has been created.")
        print("You can now distribute TimeStudyAnalyzer_Installer.exe")
        print("Users just need to run it to install everything automatically.")
    else:
        print()
        print("FAILED!")
        print("Could not create the installer executable.")

if __name__ == "__main__":
    main()
