"""
Time Study Analyzer - One Click Installer
==========================================
This installer will:
1. Check for Python installation
2. Download the latest code from GitHub
3. Install all required dependencies
4. Create a desktop shortcut
5. Launch the application

Author: GitHub Copilot Assistant
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import shutil
import winreg
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
import threading
import json

class TimeStudyInstaller:
    def __init__(self):
        self.repo_url = "https://github.com/Cayton3359/timestudyanalyzer"
        self.install_dir = os.path.join(os.path.expanduser("~"), "TimeStudyAnalyzer")
        self.desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        self.progress_var = None
        self.status_var = None
        
    def setup_gui(self):
        """Create the installer GUI"""
        self.root = tk.Tk()
        self.root.title("Time Study Analyzer - One Click Installer")
        self.root.geometry("500x400")
        self.root.resizable(False, False)
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Time Study Analyzer Installer", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Description
        desc_text = """This installer will:
• Check for Python installation
• Download the latest code from GitHub
• Install all required dependencies
• Create a desktop shortcut
• Launch the application

Installation directory: """ + self.install_dir
        
        desc_label = ttk.Label(main_frame, text=desc_text, justify="left")
        desc_label.grid(row=1, column=0, columnspan=2, pady=(0, 20), sticky="w")
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, 
                                     maximum=100, length=400)
        progress_bar.grid(row=2, column=0, columnspan=2, pady=(0, 10), sticky="ew")
        
        # Status label
        self.status_var = tk.StringVar(value="Ready to install...")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.grid(row=3, column=0, columnspan=2, pady=(0, 20))
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=2)
        
        # Install button
        self.install_btn = ttk.Button(button_frame, text="Install Time Study Analyzer", 
                                     command=self.start_installation, style="Accent.TButton")
        self.install_btn.pack(side="left", padx=(0, 10))
        
        # Exit button
        exit_btn = ttk.Button(button_frame, text="Exit", command=self.root.quit)
        exit_btn.pack(side="left")
        
        # Log text area
        log_frame = ttk.LabelFrame(main_frame, text="Installation Log", padding="10")
        log_frame.grid(row=5, column=0, columnspan=2, pady=(20, 0), sticky="ew")
        
        self.log_text = tk.Text(log_frame, height=8, width=60)
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def log_message(self, message):
        """Add message to log"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update()
        
    def update_progress(self, value, status):
        """Update progress bar and status"""
        self.progress_var.set(value)
        self.status_var.set(status)
        self.root.update()
        
    def check_python(self):
        """Check if Python is installed"""
        self.log_message("Checking Python installation...")
        try:
            result = subprocess.run([sys.executable, "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                python_version = result.stdout.strip()
                self.log_message(f"✓ Found {python_version}")
                return True
            else:
                self.log_message("✗ Python not found")
                return False
        except Exception as e:
            self.log_message(f"✗ Error checking Python: {e}")
            return False
            
    def check_git(self):
        """Check if git is available"""
        try:
            result = subprocess.run(["git", "--version"], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
            
    def download_from_github(self):
        """Download the repository from GitHub"""
        self.log_message("Downloading from GitHub...")
        
        # Remove existing directory
        if os.path.exists(self.install_dir):
            self.log_message(f"Removing existing installation: {self.install_dir}")
            shutil.rmtree(self.install_dir)
        
        try:
            if self.check_git():
                # Use git clone if available
                self.log_message("Using git to clone repository...")
                result = subprocess.run([
                    "git", "clone", self.repo_url + ".git", self.install_dir
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.log_message("✓ Repository cloned successfully")
                    return True
                else:
                    self.log_message(f"Git clone failed: {result.stderr}")
                    # Fall back to ZIP download
            
            # Download as ZIP file
            self.log_message("Downloading repository as ZIP...")
            zip_url = self.repo_url + "/archive/refs/heads/main.zip"
            zip_path = os.path.join(os.path.expanduser("~"), "timestudyanalyzer.zip")
            
            urllib.request.urlretrieve(zip_url, zip_path)
            self.log_message("✓ ZIP file downloaded")
            
            # Extract ZIP
            self.log_message("Extracting files...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.expanduser("~"))
            
            # Rename extracted folder
            extracted_folder = os.path.join(os.path.expanduser("~"), "timestudyanalyzer-main")
            if os.path.exists(extracted_folder):
                os.rename(extracted_folder, self.install_dir)
                self.log_message("✓ Files extracted successfully")
            
            # Clean up ZIP file
            os.remove(zip_path)
            return True
            
        except Exception as e:
            self.log_message(f"✗ Download failed: {e}")
            return False
            
    def install_dependencies(self):
        """Install Python dependencies"""
        self.log_message("Installing Python dependencies...")
        
        requirements_file = os.path.join(self.install_dir, "requirements.txt")
        if not os.path.exists(requirements_file):
            self.log_message("✗ requirements.txt not found")
            return False
            
        try:
            # Upgrade pip first
            self.log_message("Upgrading pip...")
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                         check=True, capture_output=True)
            
            # Install requirements
            self.log_message("Installing packages...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", requirements_file
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.log_message("✓ All dependencies installed successfully")
                return True
            else:
                self.log_message(f"✗ Installation failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.log_message(f"✗ Error installing dependencies: {e}")
            return False
            
    def create_desktop_shortcut(self):
        """Create desktop shortcut"""
        self.log_message("Creating desktop shortcut...")
        
        try:
            # Create batch file to run the application
            batch_content = f'''@echo off
cd /d "{self.install_dir}"
"{sys.executable}" main.py
pause'''
            
            batch_path = os.path.join(self.install_dir, "run_time_study.bat")
            with open(batch_path, 'w') as f:
                f.write(batch_content)
            
            # Create shortcut on desktop
            shortcut_path = os.path.join(self.desktop_path, "Time Study Analyzer.lnk")
            
            # Use PowerShell to create shortcut
            ps_script = f'''
$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("{shortcut_path}")
$Shortcut.TargetPath = "{batch_path}"
$Shortcut.WorkingDirectory = "{self.install_dir}"
$Shortcut.Description = "Time Study Analyzer Application"
$Shortcut.Save()
'''
            
            result = subprocess.run([
                "powershell", "-Command", ps_script
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.log_message("✓ Desktop shortcut created")
                return True
            else:
                self.log_message(f"Shortcut creation failed: {result.stderr}")
                # Create a simple batch file on desktop as fallback
                desktop_batch = os.path.join(self.desktop_path, "Time Study Analyzer.bat")
                with open(desktop_batch, 'w') as f:
                    f.write(batch_content)
                self.log_message("✓ Desktop batch file created as fallback")
                return True
                
        except Exception as e:
            self.log_message(f"✗ Error creating shortcut: {e}")
            return False
            
    def test_installation(self):
        """Test if the installation works"""
        self.log_message("Testing installation...")
        
        main_py = os.path.join(self.install_dir, "main.py")
        if not os.path.exists(main_py):
            self.log_message("✗ main.py not found")
            return False
            
        try:
            # Test import of main modules
            test_script = f'''
import sys
sys.path.insert(0, "{self.install_dir}")
try:
    import main
    print("SUCCESS: Main module imports correctly")
except Exception as e:
    print(f"ERROR: {{e}}")
'''
            
            result = subprocess.run([sys.executable, "-c", test_script], 
                                  capture_output=True, text=True, cwd=self.install_dir)
            
            if "SUCCESS" in result.stdout:
                self.log_message("✓ Installation test passed")
                return True
            else:
                self.log_message(f"✗ Installation test failed: {result.stdout} {result.stderr}")
                return False
                
        except Exception as e:
            self.log_message(f"✗ Error testing installation: {e}")
            return False
            
    def launch_application(self):
        """Launch the Time Study Analyzer"""
        self.log_message("Launching Time Study Analyzer...")
        
        try:
            main_py = os.path.join(self.install_dir, "main.py")
            subprocess.Popen([sys.executable, main_py], cwd=self.install_dir)
            self.log_message("✓ Application launched!")
            return True
        except Exception as e:
            self.log_message(f"✗ Error launching application: {e}")
            return False
            
    def installation_thread(self):
        """Run installation in separate thread"""
        try:
            self.install_btn.config(state="disabled")
            
            # Step 1: Check Python
            self.update_progress(10, "Checking Python...")
            if not self.check_python():
                messagebox.showerror("Error", "Python is not installed or not accessible!")
                return
                
            # Step 2: Download from GitHub
            self.update_progress(25, "Downloading from GitHub...")
            if not self.download_from_github():
                messagebox.showerror("Error", "Failed to download from GitHub!")
                return
                
            # Step 3: Install dependencies
            self.update_progress(50, "Installing dependencies...")
            if not self.install_dependencies():
                messagebox.showerror("Error", "Failed to install dependencies!")
                return
                
            # Step 4: Create shortcut
            self.update_progress(75, "Creating desktop shortcut...")
            self.create_desktop_shortcut()
            
            # Step 5: Test installation
            self.update_progress(90, "Testing installation...")
            if not self.test_installation():
                messagebox.showwarning("Warning", "Installation test failed, but continuing...")
                
            # Step 6: Complete
            self.update_progress(100, "Installation complete!")
            
            # Ask if user wants to launch now
            if messagebox.askyesno("Installation Complete", 
                                 "Installation completed successfully!\n\n"
                                 "Would you like to launch Time Study Analyzer now?"):
                self.launch_application()
                
            messagebox.showinfo("Success", 
                              f"Time Study Analyzer has been installed to:\n{self.install_dir}\n\n"
                              "You can launch it using the desktop shortcut or by running main.py")
                              
        except Exception as e:
            self.log_message(f"✗ Installation failed: {e}")
            messagebox.showerror("Installation Failed", f"An error occurred: {e}")
        finally:
            self.install_btn.config(state="normal")
            
    def start_installation(self):
        """Start installation in background thread"""
        thread = threading.Thread(target=self.installation_thread)
        thread.daemon = True
        thread.start()
        
    def run(self):
        """Run the installer GUI"""
        self.setup_gui()
        self.root.mainloop()

def main():
    """Main entry point"""
    installer = TimeStudyInstaller()
    installer.run()

if __name__ == "__main__":
    main()
