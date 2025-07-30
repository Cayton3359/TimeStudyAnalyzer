import sys
import os
import importlib.util
import subprocess
import requests
import json
from PyQt5.QtCore import QThread, pyqtSignal

# Direct import method - no package needed
analyzer_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           "analyzer", "analyze_cards.py")

# Import the module directly from file
spec = importlib.util.spec_from_file_location("analyze_cards", analyzer_path)
analyze_cards = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze_cards)

# Rest of your imports remain the same
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QPushButton, 
                            QLabel, QGroupBox, QGridLayout, QLineEdit, 
                            QFileDialog, QSlider, QCheckBox, QMessageBox, QButtonGroup, 
                            QHBoxLayout, QRadioButton, QApplication)
from PyQt5.QtCore import Qt

class UpdateChecker(QThread):
    """Background thread to check for updates"""
    update_available = pyqtSignal(str, str, str)  # current_version, latest_version, download_url
    
    def __init__(self, current_version):
        super().__init__()
        self.current_version = current_version
        
    def run(self):
        try:
            url = "https://api.github.com/repos/cayton3359/TimeStudyAnalyzer/releases/latest"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                latest_release = response.json()
                latest_version = latest_release['tag_name']
                download_url = latest_release['html_url']
                
                if latest_version != self.current_version:
                    self.update_available.emit(self.current_version, latest_version, download_url)
        except Exception:
            # Silently fail - don't interrupt the GUI
            pass

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Time Study Analyzer {analyze_cards.__version__}")
        self.setGeometry(100, 100, 600, 450)  # Smaller window size
        
        # Check for updates in background
        self.update_checker = UpdateChecker(analyze_cards.__version__)
        self.update_checker.update_available.connect(self.show_update_notification)
        self.update_checker.start()
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)  # Smaller margins
        main_layout.setSpacing(8)  # Reduce spacing between widgets
        
        # Create input fields for folder paths
        path_group = QGroupBox("File Locations")
        path_layout = QGridLayout()
        path_layout.setContentsMargins(7, 12, 7, 7)  # Smaller internal margins
        path_layout.setVerticalSpacing(5)  # Less space between rows
        
        # Input folder
        path_layout.addWidget(QLabel("Video Folder:"), 0, 0)
        self.input_folder = QLineEdit(analyze_cards.root_folder)
        path_layout.addWidget(self.input_folder, 0, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_input)
        path_layout.addWidget(browse_btn, 0, 2)
        
        # Output file
        path_layout.addWidget(QLabel("Output Excel:"), 1, 0)
        self.output_file = QLineEdit(analyze_cards.output_excel)
        path_layout.addWidget(self.output_file, 1, 1)
        save_btn = QPushButton("Browse...")
        save_btn.clicked.connect(self.browse_output)
        path_layout.addWidget(save_btn, 1, 2)
        
        path_group.setLayout(path_layout)
        main_layout.addWidget(path_group)
        
        # Resolution toggle buttons
        resolution_group = QGroupBox("Video Resolution")
        resolution_layout = QHBoxLayout()  # Horizontal layout for buttons side by side
        resolution_layout.setContentsMargins(7, 12, 7, 7)  # Smaller internal margins
        
        # Create button group for mutual exclusion
        self.resolution_buttons = QButtonGroup(self)
        
        # 360p radio button
        self.btn_360p = QRadioButton("360p")
        self.btn_360p.setChecked(analyze_cards.RESOLUTION_SETTING == "360p")  # Set initial state
        self.resolution_buttons.addButton(self.btn_360p)
        
        # 720p radio button
        self.btn_720p = QRadioButton("720p")
        self.btn_720p.setChecked(analyze_cards.RESOLUTION_SETTING == "720p")  # Set initial state
        self.resolution_buttons.addButton(self.btn_720p)
        
        # 1080p radio button
        self.btn_1080p = QRadioButton("1080p")
        self.btn_1080p.setChecked(analyze_cards.RESOLUTION_SETTING == "1080p")  # Set initial state
        self.resolution_buttons.addButton(self.btn_1080p)
        
        # Connect button clicks
        self.btn_360p.toggled.connect(lambda checked: checked and self.toggle_resolution("360p"))
        self.btn_720p.toggled.connect(lambda checked: checked and self.toggle_resolution("720p"))
        self.btn_1080p.toggled.connect(lambda checked: checked and self.toggle_resolution("1080p"))
        
        # Add buttons to layout
        resolution_layout.addWidget(self.btn_360p)
        resolution_layout.addWidget(self.btn_720p)
        resolution_layout.addWidget(self.btn_1080p)
        resolution_layout.addStretch(1)  # Push advanced button to the right

        # Add advanced button in-line with resolution buttons
        self.advanced_button = QPushButton("▶ Advanced")
        self.advanced_button.setCheckable(True)
        self.advanced_button.setFixedWidth(120)  # Make it smaller
        self.advanced_button.clicked.connect(self.toggle_advanced_options)
        resolution_layout.addWidget(self.advanced_button)

        resolution_group.setLayout(resolution_layout)
        main_layout.addWidget(resolution_group)
        
        # Create container for advanced options
        self.advanced_container = QWidget()
        self.advanced_container.setVisible(False)  # Initially hidden
        advanced_layout = QVBoxLayout(self.advanced_container)
        advanced_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins for cleaner look
        advanced_layout.setSpacing(5)  # Less space between elements
        main_layout.addWidget(self.advanced_container)

        # Optional: Parameters group
        param_group = QGroupBox("Detection Parameters")
        param_layout = QGridLayout()
        param_layout.setContentsMargins(7, 12, 7, 7)
        param_layout.setVerticalSpacing(3)  # Minimal vertical spacing
        
        # Min area ratio
        param_layout.addWidget(QLabel("Min Area Ratio:"), 0, 0)
        self.area_slider = QSlider(Qt.Horizontal)
        self.area_slider.setMinimum(5)
        self.area_slider.setMaximum(30)
        self.area_slider.setValue(int(analyze_cards.min_area_ratio * 100))
        self.area_slider.setTickPosition(QSlider.TicksBelow)
        self.area_slider.setTickInterval(5)
        self.area_slider.setFixedWidth(150)  # Shorter sliders
        param_layout.addWidget(self.area_slider, 0, 1)
        
        # Resize factor
        param_layout.addWidget(QLabel("Resize Factor:"), 1, 0)
        self.resize_slider = QSlider(Qt.Horizontal)
        self.resize_slider.setMinimum(10)
        self.resize_slider.setMaximum(100)
        self.resize_slider.setValue(int(analyze_cards.RESIZE_FACTOR * 100))
        self.resize_slider.setTickPosition(QSlider.TicksBelow)
        self.resize_slider.setTickInterval(10)
        self.resize_slider.setFixedWidth(150)
        param_layout.addWidget(self.resize_slider, 1, 1)
        
        # Frame skip control
        param_layout.addWidget(QLabel("Frame Skip:"), 2, 0)
        self.frame_skip_slider = QSlider(Qt.Horizontal)
        self.frame_skip_slider.setMinimum(1)
        self.frame_skip_slider.setMaximum(10)
        self.frame_skip_slider.setValue(analyze_cards.frame_skip)
        self.frame_skip_slider.setTickPosition(QSlider.TicksBelow)
        self.frame_skip_slider.setTickInterval(1)
        self.frame_skip_slider.setFixedWidth(150)
        param_layout.addWidget(self.frame_skip_slider, 2, 1)

        # In __init__, add these label references
        # Min area ratio status
        self.area_label = QLabel(f"{analyze_cards.min_area_ratio:.2f}")
        param_layout.addWidget(self.area_label, 0, 2)
        self.area_status = QLabel("(Recommended)")
        self.area_status.setStyleSheet("color: green;")
        param_layout.addWidget(self.area_status, 0, 3)

        # Resize factor status
        self.resize_label = QLabel(f"{analyze_cards.RESIZE_FACTOR:.2f}")
        param_layout.addWidget(self.resize_label, 1, 2)
        self.resize_status = QLabel("(Recommended)")
        self.resize_status.setStyleSheet("color: green;")
        param_layout.addWidget(self.resize_status, 1, 3)

        # Frame skip status
        self.frame_skip_label = QLabel(f"{analyze_cards.frame_skip}")
        param_layout.addWidget(self.frame_skip_label, 2, 2)
        self.frame_skip_status = QLabel("(Recommended)")
        self.frame_skip_status.setStyleSheet("color: green;")
        param_layout.addWidget(self.frame_skip_status, 2, 3)
        
        param_group.setLayout(param_layout)
        advanced_layout.addWidget(param_group)
        
        # Debug options - move this into advanced container
        debug_group = QGroupBox("Debug Options")
        debug_layout = QVBoxLayout()
        debug_layout.setContentsMargins(7, 12, 7, 7)
        debug_layout.setSpacing(5)
        self.debug_check = QCheckBox("Debug Mode")
        self.debug_check.setChecked(analyze_cards.DEBUG)
        debug_layout.addWidget(self.debug_check)
        self.verbose_check = QCheckBox("Verbose Output")
        self.verbose_check.setChecked(analyze_cards.VERBOSE)
        debug_layout.addWidget(self.verbose_check)
        debug_group.setLayout(debug_layout)
        advanced_layout.addWidget(debug_group)
        
        # Add spacer before run button for better spacing
        main_layout.addStretch(1)

        # Make run button centered and smaller
        run_button_container = QHBoxLayout()
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.setFixedWidth(150)  # Narrower
        self.run_btn.setFixedHeight(35)  # Slightly less tall
        self.run_btn.setStyleSheet("QPushButton { font-weight: bold; background-color: #4CAF50; color: white; }")
        self.run_btn.clicked.connect(self.run_analysis)
        run_button_container.addStretch(1)  # Push button to center
        run_button_container.addWidget(self.run_btn)
        run_button_container.addStretch(1)  # Push button to center
        main_layout.addLayout(run_button_container)
        
        # Connect sliders to update functions
        self.area_slider.valueChanged.connect(self.update_area_value)
        self.resize_slider.valueChanged.connect(self.update_resize_value)
        self.frame_skip_slider.valueChanged.connect(self.update_frame_skip)
        
        # Initialize UI with current settings
        self.apply_resolution_settings(analyze_cards.RESOLUTION_SETTING)
    
    def browse_input(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Video Folder", self.input_folder.text())
        if folder:
            self.input_folder.setText(folder)
    
    def browse_output(self):
        file, _ = QFileDialog.getSaveFileName(self, "Select Output Excel File", 
                                           self.output_file.text(), "Excel Files (*.xlsx)")
        if file:
            self.output_file.setText(file)
    
    def toggle_resolution(self, resolution):
        """Toggle between 360p, 720p, and 1080p settings"""
        # Apply all recommended settings for this resolution
        self.apply_resolution_settings(resolution)
        
    def apply_resolution_settings(self, resolution=None):
        """Apply all recommended settings based on the selected resolution"""
        # If no resolution specified, get it from the radio buttons
        if resolution is None:
            if self.btn_360p.isChecked():
                resolution = "360p"
            elif self.btn_720p.isChecked():
                resolution = "720p"
            elif self.btn_1080p.isChecked():
                resolution = "1080p"
            else:
                resolution = "360p"  # Default fallback
        
        # Get appropriate settings dictionary
        if resolution == "360p":
            settings = analyze_cards.SETTINGS_360P
        elif resolution == "720p":
            settings = analyze_cards.SETTINGS_720P
        elif resolution == "1080p":
            settings = analyze_cards.SETTINGS_1080P
        else:
            settings = analyze_cards.SETTINGS_360P  # Default fallback
        
        # Update sliders to match recommended settings
        self.area_slider.setValue(int(settings["min_area_ratio"] * 100))
        self.resize_slider.setValue(int(settings["resize_factor"] * 100))
        self.frame_skip_slider.setValue(settings["frame_skip"])
        
        # Update UI labels
        self.area_label.setText(f"{settings['min_area_ratio']:.2f}")
        self.resize_label.setText(f"{settings['resize_factor']:.2f}")
        self.frame_skip_label.setText(f"{settings['frame_skip']}")
        
        # Set radio buttons to match
        self.btn_360p.setChecked(resolution == "360p")
        self.btn_720p.setChecked(resolution == "720p")
        self.btn_1080p.setChecked(resolution == "1080p")

        # Reset all status labels to "Recommended"
        self.area_status.setText("(Recommended)")
        self.area_status.setStyleSheet("color: green;")
        self.resize_status.setText("(Recommended)")
        self.resize_status.setStyleSheet("color: green;")
        self.frame_skip_status.setText("(Recommended)")
        self.frame_skip_status.setStyleSheet("color: green;")

    def run_analysis(self):
        # Get the output file path
        output_file = self.output_file.text()
        
        # Update settings from the GUI
        analyze_cards.root_folder = self.input_folder.text()
        analyze_cards.output_excel = output_file
        
        # Get selected resolution
        if self.btn_360p.isChecked():
            analyze_cards.RESOLUTION_SETTING = "360p"
        elif self.btn_720p.isChecked():
            analyze_cards.RESOLUTION_SETTING = "720p"
        elif self.btn_1080p.isChecked():
            analyze_cards.RESOLUTION_SETTING = "1080p"
        
        analyze_cards.DEBUG = self.debug_check.isChecked() if hasattr(self, 'debug_check') else False
        analyze_cards.VERBOSE = self.verbose_check.isChecked() if hasattr(self, 'verbose_check') else False
        
        # Update settings based on resolution selection
        if analyze_cards.RESOLUTION_SETTING == "360p":
            analyze_cards.settings = analyze_cards.SETTINGS_360P
        elif analyze_cards.RESOLUTION_SETTING == "720p":
            analyze_cards.settings = analyze_cards.SETTINGS_720P
        elif analyze_cards.RESOLUTION_SETTING == "1080p":
            analyze_cards.settings = analyze_cards.SETTINGS_1080P
            
        # Apply settings
        analyze_cards.min_area_ratio = analyze_cards.settings["min_area_ratio"]
        analyze_cards.debounce_secs = analyze_cards.settings["debounce_secs"]
        analyze_cards.min_frames = analyze_cards.settings["min_frames"]
        analyze_cards.RESIZE_FACTOR = analyze_cards.settings["resize_factor"]
        analyze_cards.frame_skip = analyze_cards.settings["frame_skip"]
        analyze_cards.W_TEMPLATE = analyze_cards.settings["w_template"]
        analyze_cards.W_HUE = analyze_cards.settings["w_hue"]
        analyze_cards.W_OCR = analyze_cards.settings["w_ocr"]
        analyze_cards.W_EDGE = analyze_cards.settings["w_edge"]
        analyze_cards.TEMPLATE_STRONG = analyze_cards.settings["template_strong"]
        analyze_cards.HUE_STRONG = analyze_cards.settings["hue_strong"]
        analyze_cards.OCR_STRONG = analyze_cards.settings["ocr_strong"]
        analyze_cards.EDGE_STRONG = analyze_cards.settings["edge_strong"]
        
        # Run the analysis
        try:
            # Show a "Processing..." message
            self.run_btn.setText("Processing...")
            self.run_btn.setEnabled(False)
            QApplication.processEvents()  # Update the UI
            
            # Run the analysis
            analyze_cards.main()
            
            # Reset the button
            self.run_btn.setText("Run Analysis")
            self.run_btn.setEnabled(True)
            
            # Show success message
            QMessageBox.information(self, "Success", "Analysis completed successfully!")
            
            # Open the Excel file if it exists
            if os.path.exists(output_file):
                try:
                    # For Windows
                    os.startfile(output_file)
                except AttributeError:
                    # For non-Windows platforms
                    subprocess.run(['xdg-open', output_file], check=True)
                
        except Exception as e:
            # Reset the button
            self.run_btn.setText("Run Analysis")
            self.run_btn.setEnabled(True)
            
            # Show error message
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
            # Print full error information to console for debugging
            import traceback
            traceback.print_exc()
    
    def update_area_value(self, value):
        """Update the min area ratio value label when slider changes."""
        ratio = value / 100.0
        self.area_label.setText(f"{ratio:.2f}")
        
        # Check if this matches the recommended value for current resolution
        if self.btn_360p.isChecked():
            settings = analyze_cards.SETTINGS_360P
        elif self.btn_720p.isChecked():
            settings = analyze_cards.SETTINGS_720P
        elif self.btn_1080p.isChecked():
            settings = analyze_cards.SETTINGS_1080P
        else:
            settings = analyze_cards.SETTINGS_360P  # Default
            
        if abs(ratio - settings["min_area_ratio"]) < 0.01:  # Small tolerance for float comparison
            self.area_status.setText("(Recommended)")
            self.area_status.setStyleSheet("color: green;")
        else:
            self.area_status.setText("(Custom)")
            self.area_status.setStyleSheet("color: blue;")
    
    def update_resize_value(self, value):
        """Update the resize factor value label when slider changes."""
        factor = value / 100.0
        self.resize_label.setText(f"{factor:.2f}")
        
        # Check if this matches the recommended value for current resolution
        if self.btn_360p.isChecked():
            settings = analyze_cards.SETTINGS_360P
        elif self.btn_720p.isChecked():
            settings = analyze_cards.SETTINGS_720P
        elif self.btn_1080p.isChecked():
            settings = analyze_cards.SETTINGS_1080P
        else:
            settings = analyze_cards.SETTINGS_360P  # Default
            
        if abs(factor - settings["resize_factor"]) < 0.01:  # Small tolerance for float comparison
            self.resize_status.setText("(Recommended)")
            self.resize_status.setStyleSheet("color: green;")
        else:
            self.resize_status.setText("(Custom)")
            self.resize_status.setStyleSheet("color: blue;")

    def update_frame_skip(self, value):
        """Update the frame skip value label when slider changes."""
        self.frame_skip_label.setText(f"{value}")
        
        # Check if this matches the recommended value for current resolution
        if self.btn_360p.isChecked():
            settings = analyze_cards.SETTINGS_360P
        elif self.btn_720p.isChecked():
            settings = analyze_cards.SETTINGS_720P
        elif self.btn_1080p.isChecked():
            settings = analyze_cards.SETTINGS_1080P
        else:
            settings = analyze_cards.SETTINGS_360P  # Default
            
        if value == settings["frame_skip"]:
            self.frame_skip_status.setText("(Recommended)")
            self.frame_skip_status.setStyleSheet("color: green;")
        else:
            self.frame_skip_status.setText("(Custom)")
            self.frame_skip_status.setStyleSheet("color: blue;")

    def toggle_advanced_options(self):
        """Show or hide advanced options"""
        is_visible = not self.advanced_container.isVisible()
        self.advanced_container.setVisible(is_visible)
        
        # Update button text to show state
        if is_visible:
            self.advanced_button.setText("▼ Hide Advanced Options")
        else:
            self.advanced_button.setText("▶ Show Advanced Options")
    
    def show_update_notification(self, current_version, latest_version, download_url):
        """Show update notification dialog"""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Update Available")
        msg.setText(f"A new version of Time Study Analyzer is available!")
        msg.setInformativeText(f"Current version: {current_version}\nLatest version: {latest_version}\n\nWould you like to download the update?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.Yes)
        
        if msg.exec_() == QMessageBox.Yes:
            import webbrowser
            webbrowser.open(download_url)