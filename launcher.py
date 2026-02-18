"""
Time Study Analyzer - All-in-One Launcher
Single EXE that contains everything and launches the GUI
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

def extract_bundled_files():
    """Extract bundled files to a temporary directory"""
    # Get the directory where this EXE is running from
    if getattr(sys, 'frozen', False):
        # Running as compiled EXE
        bundle_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    else:
        # Running as script
        bundle_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create a user data directory
    app_data_dir = os.path.join(os.environ.get('APPDATA', ''), 'TimeStudyAnalyzer')
    os.makedirs(app_data_dir, exist_ok=True)
    
    # Copy templates if they don't exist
    templates_dir = os.path.join(app_data_dir, 'templates')
    if not os.path.exists(templates_dir):
        bundled_templates = os.path.join(bundle_dir, 'templates')
        if os.path.exists(bundled_templates):
            shutil.copytree(bundled_templates, templates_dir)
    
    # Copy PowerPoint file
    powerpoint_dest = os.path.join(app_data_dir, 'Que Cards.pptx')
    if not os.path.exists(powerpoint_dest):
        bundled_ppt = os.path.join(bundle_dir, 'Que Cards.pptx')
        if os.path.exists(bundled_ppt):
            shutil.copy2(bundled_ppt, powerpoint_dest)
    
    # Copy user guide
    guide_dest = os.path.join(app_data_dir, 'User-Guide.md')
    bundled_guide = os.path.join(bundle_dir, 'User-Guide.md')
    if os.path.exists(bundled_guide):
        shutil.copy2(bundled_guide, guide_dest)
    
    return app_data_dir

def check_tesseract():
    """Check if Tesseract OCR is installed"""
    try:
        subprocess.run(['tesseract', '--version'], 
                      capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def show_setup_dialog():
    """Show setup dialog for first-time users"""
    root = tk.Tk()
    root.withdraw()  # Hide main window
    
    # Extract files first
    app_data_dir = extract_bundled_files()
    
    # Check for Tesseract
    if not check_tesseract():
        result = messagebox.askyesno(
            "Time Study Analyzer - Setup Required",
            "Welcome to Time Study Analyzer!\n\n" +
            "This software requires Tesseract OCR for text recognition.\n\n" +
            "Would you like to:\n" +
            "• Install Tesseract OCR automatically, OR\n" +
            "• Download it manually from GitHub?\n\n" +
            "Click 'Yes' for automatic install, 'No' to download manually.",
            icon='question'
        )
        
        if result:
            # Try automatic install
            try:
                import urllib.request
                tesseract_url = "https://github.com/UB-Mannheim/tesseract/releases/download/v5.3.3/tesseract-ocr-w64-setup-5.3.3.20231005.exe"
                
                with tempfile.NamedTemporaryFile(suffix='.exe', delete=False) as tmp_file:
                    messagebox.showinfo("Downloading", "Downloading Tesseract OCR installer...\nThis may take a few minutes.")
                    urllib.request.urlretrieve(tesseract_url, tmp_file.name)
                    
                    # Run installer
                    subprocess.run([tmp_file.name], check=True)
                    os.unlink(tmp_file.name)
                    
                    messagebox.showinfo("Success", "Tesseract OCR installed successfully!\nTime Study Analyzer will now launch.")
                    
            except Exception as e:
                messagebox.showerror("Installation Failed", 
                                   f"Automatic installation failed: {str(e)}\n\n" +
                                   "Please install Tesseract OCR manually from:\n" +
                                   "https://github.com/UB-Mannheim/tesseract/wiki")
                return False
        else:
            # Manual download
            import webbrowser
            webbrowser.open("https://github.com/UB-Mannheim/tesseract/wiki")
            messagebox.showinfo("Manual Installation", 
                              "Opening Tesseract download page in your browser.\n\n" +
                              "After installing Tesseract OCR, run this program again.")
            return False
    
    # Show welcome message with file locations
    messagebox.showinfo(
        "Time Study Analyzer - Ready!",
        f"Time Study Analyzer is ready to use!\n\n" +
        f"Your files are located at:\n{app_data_dir}\n\n" +
        f"• Print 'Que Cards.pptx' for colored cards\n" +
        f"• See 'User-Guide.md' for instructions\n\n" +
        f"The application will now launch."
    )
    
    root.destroy()
    return True

def main():
    """Main entry point"""
    # Check if this is the first run or if setup is needed
    app_data_dir = extract_bundled_files()
    
    # For first-time users, show setup dialog
    setup_file = os.path.join(app_data_dir, '.setup_complete')
    if not os.path.exists(setup_file):
        if show_setup_dialog():
            # Mark setup as complete
            with open(setup_file, 'w') as f:
                f.write('Setup completed')
        else:
            return  # User cancelled or setup failed
    
    # Launch the main GUI application
    try:
        # Import and run the main application
        import importlib.util
        
        # Load the main window module
        if getattr(sys, 'frozen', False):
            # Running as EXE - import the bundled module
            from src.gui.main_window import MainWindow
            from PyQt5.QtWidgets import QApplication
            
            app = QApplication(sys.argv)
            window = MainWindow()
            window.show()
            sys.exit(app.exec_())
        else:
            # Running as script - use the regular import
            import main
            try:
                main.main()
            except KeyboardInterrupt:
                # Exit cleanly if the terminal sends Ctrl+C / SIGINT.
                return
            
    except Exception as e:
        import tkinter as tk
        from tkinter import messagebox
        
        root = tk.Tk()
        root.withdraw()
        
        messagebox.showerror("Launch Error", 
                           f"Failed to launch Time Study Analyzer:\n{str(e)}\n\n" +
                           f"Please check that all files are present in:\n{app_data_dir}")

if __name__ == "__main__":
    main()
