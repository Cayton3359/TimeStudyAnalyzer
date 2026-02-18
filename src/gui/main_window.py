import sys
import os
import importlib.util
import subprocess
import json
import multiprocessing
import time
import threading
from typing import Any
from PyQt5.QtCore import QThread, pyqtSignal

DEFAULT_VERSION = "v1.0.0"
DEFAULT_RESOLUTION = "360p"
DEFAULT_ROOT_FOLDER = r"D:\record"
DEFAULT_OUTPUT_EXCEL = r"M:\Engineering\Lindsey\12. Code\Time Study Camera\TimeStudyAnalyzer\output\card_events_summary.xlsx"
DEFAULT_DEBUG = False
DEFAULT_VERBOSE = False

DEFAULT_SETTINGS_360P = {
    "min_area_ratio": 0.15,
    "debounce_secs": 10.0,
    "min_frames": 2,
    "frame_skip": 4,
    "resize_factor": 0.4,
    "w_template": 0.2,
    "w_hue": 0.1,
    "w_ocr": 0.6,
    "w_edge": 0.1,
    "template_strong": 0.4,
    "hue_strong": 0.6,
    "ocr_strong": 0.5,
    "edge_strong": 0.6
}
DEFAULT_SETTINGS_720P = {
    "min_area_ratio": 0.20,
    "debounce_secs": 10.0,
    "min_frames": 2,
    "frame_skip": 4,
    "resize_factor": 0.25,
    "w_template": 0.2,
    "w_hue": 0.1,
    "w_ocr": 0.6,
    "w_edge": 0.1,
    "template_strong": 0.4,
    "hue_strong": 0.6,
    "ocr_strong": 0.5,
    "edge_strong": 0.6
}
DEFAULT_SETTINGS_1080P = {
    "min_area_ratio": 0.20,
    "debounce_secs": 10.0,
    "min_frames": 2,
    "frame_skip": 4,
    "resize_factor": 0.17,
    "w_template": 0.2,
    "w_hue": 0.1,
    "w_ocr": 0.6,
    "w_edge": 0.1,
    "template_strong": 0.4,
    "hue_strong": 0.6,
    "ocr_strong": 0.5,
    "edge_strong": 0.6
}

def _load_analyzer() -> Any:
    analyzer_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "analyzer",
        "analyze_cards_fast.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_cards_fast", analyzer_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load analyzer module from {analyzer_path}")
    analyzer: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    return analyzer

# Rest of your imports remain the same
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QPushButton, 
                            QLabel, QGroupBox, QGridLayout, QLineEdit, 
                            QFileDialog, QSlider, QCheckBox, QMessageBox, QButtonGroup, 
                            QHBoxLayout, QRadioButton, QApplication, QProgressBar)
from PyQt5.QtCore import Qt

class UpdateChecker(QThread):
    """Background thread to check for updates"""
    update_available = pyqtSignal(str, str, str)  # current_version, latest_version, download_url
    
    def __init__(self, current_version):
        super().__init__()
        self.current_version = current_version
        
    def run(self):
        try:
            import requests
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

class AnalysisWorker(QThread):
    """Background thread to run analysis and emit progress updates"""
    progress_updated = pyqtSignal(int, float, int)  # progress percentage, estimated_total_seconds, total_videos
    analysis_finished = pyqtSignal()
    analysis_error = pyqtSignal(str)  # error message
    
    SECONDS_PER_VIDEO = 30  # Assumed processing time per video
    
    def __init__(self, analyzer, root_folder):
        super().__init__()
        self.analyzer = analyzer
        self.root_folder = root_folder
        
    def run(self):
        try:
            # Count total video files
            video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
            total_videos = 0
            video_files = []
            
            for root, dirs, files in os.walk(self.root_folder):
                for file in files:
                    if file.lower().endswith(video_extensions):
                        total_videos += 1
                        video_files.append(os.path.join(root, file))
            
            if total_videos == 0:
                self.progress_updated.emit(100, 0, 0)
                self.analysis_finished.emit()
                return
            
            # Get number of CPU cores available for processing
            num_cores = multiprocessing.cpu_count()
            
            # Calculate estimated total processing time
            # Each core processes 1 video in 30 seconds
            # With N cores, we can process N videos in 30 seconds (parallel processing)
            estimated_total_seconds = (total_videos * self.SECONDS_PER_VIDEO) / num_cores
            
            # Emit initial progress
            self.progress_updated.emit(0, estimated_total_seconds, total_videos)
            
            # Start tracking elapsed time
            start_time = time.time()
            
            # Run the main analysis in a separate thread to allow progress updates
            analysis_thread = AnalysisThread(self.analyzer)
            analysis_thread.start()
            
            # Monitor elapsed time and emit progress updates
            while analysis_thread.is_alive():
                elapsed_time = time.time() - start_time
                # Calculate progress as percentage of estimated time, capped at 95%
                progress = min(int((elapsed_time / estimated_total_seconds) * 100), 95)
                self.progress_updated.emit(progress, estimated_total_seconds, total_videos)
                time.sleep(0.5)  # Update progress every 500ms
            
            # Wait for thread to finish
            analysis_thread.join()
            
            # Check if there was an error in the analysis thread
            if analysis_thread.error:
                raise Exception(analysis_thread.error)
            
            # Emit completion
            self.progress_updated.emit(100, estimated_total_seconds, total_videos)
            self.analysis_finished.emit()
            
        except Exception as e:
            self.analysis_error.emit(str(e))


class AnalysisThread(threading.Thread):
    """Helper thread to run the analyzer's main() method"""
    def __init__(self, analyzer):
        super().__init__()
        self.analyzer = analyzer
        self.error = None
    
    def run(self):
        try:
            self.analyzer.main()
        except Exception as e:
            self.error = str(e)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.analyzer = None
        self.setWindowTitle(f"Time Study Analyzer {DEFAULT_VERSION}")
        self.setGeometry(100, 100, 600, 450)  # Smaller window size
        
        # Set tooltip styling for better visibility
        self.setStyleSheet("""
            QToolTip {
                background-color: #f0f0f0;
                color: #333;
                border: 2px solid #4CAF50;
                border-radius: 5px;
                padding: 5px;
                font-size: 11px;
            }
        """)
        
        # Check for updates in background
        self.update_checker = UpdateChecker(DEFAULT_VERSION)
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
        self.input_folder = QLineEdit(DEFAULT_ROOT_FOLDER)
        path_layout.addWidget(self.input_folder, 0, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_input)
        path_layout.addWidget(browse_btn, 0, 2)
        
        # Output file
        path_layout.addWidget(QLabel("Output Excel:"), 1, 0)
        self.output_file = QLineEdit(DEFAULT_OUTPUT_EXCEL)
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
        self.btn_360p.setChecked(DEFAULT_RESOLUTION == "360p")  # Set initial state
        self.resolution_buttons.addButton(self.btn_360p)
        
        # 720p radio button
        self.btn_720p = QRadioButton("720p")
        self.btn_720p.setChecked(DEFAULT_RESOLUTION == "720p")  # Set initial state
        self.resolution_buttons.addButton(self.btn_720p)
        
        # 1080p radio button
        self.btn_1080p = QRadioButton("1080p")
        self.btn_1080p.setChecked(DEFAULT_RESOLUTION == "1080p")  # Set initial state
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
        advanced_layout.setContentsMargins(5, 5, 5, 5)
        advanced_layout.setSpacing(10)
        main_layout.addWidget(self.advanced_container)

        # ===== FAST-PASS TUNABLES =====
        fast_pass_group = QGroupBox("Fast-Pass Tunables")
        fast_pass_layout = QGridLayout()
        fast_pass_layout.setContentsMargins(7, 12, 7, 7)
        fast_pass_layout.setVerticalSpacing(5)
        fast_pass_layout.setHorizontalSpacing(10)
        
        # Helper function to create slider rows
        def create_slider_row(layout, row, label_text, tooltip, min_val, max_val, default_val, 
                             is_float=True, decimals=2, slider_width=150):
            label = QLabel(label_text)
            label.setToolTip(tooltip)
            layout.addWidget(label, row, 0)
            
            slider = QSlider(Qt.Orientation.Horizontal)
            if is_float:
                slider.setMinimum(int(min_val * (10 ** decimals)))
                slider.setMaximum(int(max_val * (10 ** decimals)))
                slider.setValue(int(default_val * (10 ** decimals)))
            else:
                slider.setMinimum(int(min_val))
                slider.setMaximum(int(max_val))
                slider.setValue(int(default_val))
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setFixedWidth(slider_width)
            layout.addWidget(slider, row, 1)
            
            value_label = QLabel(f"{default_val:.{decimals}f}" if is_float else str(int(default_val)))
            layout.addWidget(value_label, row, 2)
            
            return slider, value_label
        
        row = 0
        # NEAR_HIT_FOCUS_SECS
        self.near_hit_focus_slider, self.near_hit_focus_label = create_slider_row(
            fast_pass_layout, row, "Near Hit Focus Time (sec):",
            "Duration to process every frame after a near hit is detected",
            0.1, 5.0, 1.0, is_float=True, decimals=1)
        self.near_hit_focus_slider.valueChanged.connect(self.update_near_hit_focus)
        
        row += 1
        # POST_CONFIRM_FOCUS_SECS
        self.post_confirm_focus_slider, self.post_confirm_focus_label = create_slider_row(
            fast_pass_layout, row, "Post Confirmation Focus Time (sec):",
            "Duration to process every frame after a card is confirmed",
            0.1, 5.0, 1.0, is_float=True, decimals=1)
        self.post_confirm_focus_slider.valueChanged.connect(self.update_post_confirm_focus)
        
        row += 1
        # NEAR_HIT_DEBOUNCE_SECS
        self.near_hit_debounce_slider, self.near_hit_debounce_label = create_slider_row(
            fast_pass_layout, row, "Near Hit Debounce (sec):",
            "Minimum spacing between near-hit log entries",
            0.5, 5.0, 2.0, is_float=True, decimals=1)
        self.near_hit_debounce_slider.valueChanged.connect(self.update_near_hit_debounce)
        
        row += 1
        # BASELINE_SKIP_STOP
        self.baseline_skip_stop_slider, self.baseline_skip_stop_label = create_slider_row(
            fast_pass_layout, row, "Baseline Skip (STOP):",
            "Process 1 of every N frames after STOP card",
            1, 16, 8, is_float=False)
        self.baseline_skip_stop_slider.valueChanged.connect(self.update_baseline_skip_stop)
        
        row += 1
        # BASELINE_SKIP_OTHER
        self.baseline_skip_other_slider, self.baseline_skip_other_label = create_slider_row(
            fast_pass_layout, row, "Baseline Skip (Other):",
            "Process 1 of every N frames after other cards",
            1, 16, 8, is_float=False)
        self.baseline_skip_other_slider.valueChanged.connect(self.update_baseline_skip_other)
        
        row += 1
        # NEAR_HIT_SUPPRESSION_WINDOW_SECS
        self.near_hit_suppression_slider, self.near_hit_suppression_label = create_slider_row(
            fast_pass_layout, row, "Near Hit Suppression Window (sec):",
            "Suppress near hits if confirmed within this window",
            0.5, 10.0, 3.0, is_float=True, decimals=1)
        self.near_hit_suppression_slider.valueChanged.connect(self.update_near_hit_suppression)
        
        row += 1
        # GLARE_CONFIRMATION_DEBOUNCE_SECS
        self.glare_debounce_slider, self.glare_debounce_label = create_slider_row(
            fast_pass_layout, row, "Glare Confirmation Debounce (sec):",
            "Remove glare entries if confirmation occurs within this window",
            0.5, 10.0, 4.0, is_float=True, decimals=1)
        self.glare_debounce_slider.valueChanged.connect(self.update_glare_debounce)
        
        row += 1
        # OCR_FUZZY
        self.ocr_fuzzy_slider, self.ocr_fuzzy_label = create_slider_row(
            fast_pass_layout, row, "OCR Fuzzy Threshold:",
            "Accept fuzzy OCR when hue is strong (0.0-1.0)",
            0.0, 1.0, 0.55, is_float=True, decimals=2)
        self.ocr_fuzzy_slider.valueChanged.connect(self.update_ocr_fuzzy)
        
        row += 1
        # LOOSE_HUE_FOCUS
        self.loose_hue_focus_slider, self.loose_hue_focus_label = create_slider_row(
            fast_pass_layout, row, "Loose Hue Focus Threshold:",
            "Loose hue trigger for focus mode (0.0-1.0)",
            0.0, 1.0, 0.30, is_float=True, decimals=2)
        self.loose_hue_focus_slider.valueChanged.connect(self.update_loose_hue_focus)
        
        fast_pass_group.setLayout(fast_pass_layout)
        advanced_layout.addWidget(fast_pass_group)
        
        # ===== DETECTION STRENGTH THRESHOLDS =====
        detection_group = QGroupBox("Detection Strength Thresholds")
        detection_layout = QGridLayout()
        detection_layout.setContentsMargins(7, 12, 7, 7)
        detection_layout.setVerticalSpacing(5)
        detection_layout.setHorizontalSpacing(10)
        
        row = 0
        # OCR_STRONG
        self.ocr_strong_slider, self.ocr_strong_label = create_slider_row(
            detection_layout, row, "OCR Strong Confidence:",
            "Confidence threshold for exact OCR match (0.0-1.0)",
            0.0, 1.0, 0.35, is_float=True, decimals=2)
        self.ocr_strong_slider.valueChanged.connect(self.update_ocr_strong)
        
        row += 1
        # HUE_STRONG
        self.hue_strong_slider, self.hue_strong_label = create_slider_row(
            detection_layout, row, "Hue Strong Threshold:",
            "Hue strength threshold for detection (0.0-1.0)",
            0.0, 1.0, 0.35, is_float=True, decimals=2)
        self.hue_strong_slider.valueChanged.connect(self.update_hue_strong)
        
        row += 1
        # TEMPLATE_STRONG
        self.template_strong_slider, self.template_strong_label = create_slider_row(
            detection_layout, row, "Template Strong Threshold:",
            "Template matching sensitivity (0.0-1.0)",
            0.0, 1.0, 0.20, is_float=True, decimals=2)
        self.template_strong_slider.valueChanged.connect(self.update_template_strong)
        
        detection_group.setLayout(detection_layout)
        advanced_layout.addWidget(detection_group)
        
        # ===== CONFIRMATION REQUIREMENTS =====
        confirm_group = QGroupBox("Confirmation Requirements")
        confirm_layout = QGridLayout()
        confirm_layout.setContentsMargins(7, 12, 7, 7)
        confirm_layout.setVerticalSpacing(5)
        confirm_layout.setHorizontalSpacing(10)
        
        row = 0
        # MIN_AREA_RATIO
        self.min_area_ratio_slider, self.min_area_ratio_label = create_slider_row(
            confirm_layout, row, "Min Area Ratio:",
            "Quads must cover at least this % of frame",
            0.01, 0.5, 0.08, is_float=True, decimals=2)
        self.min_area_ratio_slider.valueChanged.connect(self.update_min_area_ratio)
        
        row += 1
        # MIN_FRAMES
        self.min_frames_slider, self.min_frames_label = create_slider_row(
            confirm_layout, row, "Min Consecutive Frames:",
            "Minimum consecutive frames required for confirmation",
            1, 10, 1, is_float=False)
        self.min_frames_slider.valueChanged.connect(self.update_min_frames)
        
        row += 1
        # CONFIRMATION_DEBOUNCE_SECS
        self.confirmation_debounce_slider, self.confirmation_debounce_label = create_slider_row(
            confirm_layout, row, "Confirmation Debounce (sec):",
            "Seconds between same-card confirmations",
            1.0, 20.0, 5.0, is_float=True, decimals=1)
        self.confirmation_debounce_slider.valueChanged.connect(self.update_confirmation_debounce)
        
        confirm_group.setLayout(confirm_layout)
        advanced_layout.addWidget(confirm_group)
        
        # Debug options - move this into advanced container
        debug_group = QGroupBox("Debug Options")
        debug_layout = QVBoxLayout()
        debug_layout.setContentsMargins(7, 12, 7, 7)
        debug_layout.setSpacing(5)
        self.debug_check = QCheckBox("Debug Mode")
        self.debug_check.setChecked(DEFAULT_DEBUG)
        self.debug_check.setToolTip("Enable debug mode to save intermediate images and detailed logs for troubleshooting")
        debug_layout.addWidget(self.debug_check)
        self.verbose_check = QCheckBox("Verbose Output")
        self.verbose_check.setChecked(DEFAULT_VERBOSE)
        self.verbose_check.setToolTip("Enable verbose output to see detailed processing information in the console")
        debug_layout.addWidget(self.verbose_check)
        debug_group.setLayout(debug_layout)
        advanced_layout.addWidget(debug_group)
        
        # Add spacer before run button for better spacing
        main_layout.addStretch(1)
        
        # Add progress bar (hidden initially)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)

        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 5px;
                background-color: #f0f0f0;
                height: 25px;
                text-align: center;
                color: #333;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 4px;
            }
        """)
        main_layout.addWidget(self.progress_bar)
        
        # Add progress label (hidden initially)
        self.progress_label = QLabel("Processing...")
        self.progress_label.setVisible(False)
        self.progress_label.setStyleSheet("color: #666; font-size: 12px;")
        main_layout.addWidget(self.progress_label)

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
        
        # Initialize UI with current settings
        self.apply_resolution_settings(DEFAULT_RESOLUTION)
        
        # Store the original window size (before showing) for later restoration
        self.original_size = None  # Will be set when window is first shown
        
        # Analysis worker thread
        self.analysis_worker = None
    
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

    def _get_settings(self, resolution):
        analyzer = self.analyzer
        if analyzer is not None:
            if resolution == "360p":
                return analyzer.SETTINGS_360P
            if resolution == "720p":
                return analyzer.SETTINGS_720P
            if resolution == "1080p":
                return analyzer.SETTINGS_1080P
        if resolution == "360p":
            return DEFAULT_SETTINGS_360P
        if resolution == "720p":
            return DEFAULT_SETTINGS_720P
        if resolution == "1080p":
            return DEFAULT_SETTINGS_1080P
        return DEFAULT_SETTINGS_360P
        
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
        
        # Set radio buttons to match
        self.btn_360p.setChecked(resolution == "360p")
        self.btn_720p.setChecked(resolution == "720p")
        self.btn_1080p.setChecked(resolution == "1080p")

    def run_analysis(self):
        # Change button text immediately to show user that analysis is starting
        self.run_btn.setText("Running...")
        self.run_btn.setEnabled(False)
        QApplication.processEvents()  # Force UI update immediately
        
        # Get the output file path
        output_file = self.output_file.text()

        if self.analyzer is None:
            try:
                self.analyzer = _load_analyzer()
                self.setWindowTitle(f"Time Study Analyzer {self.analyzer.__version__}")
            except Exception as e:
                self.run_btn.setText("Run Analysis")
                self.run_btn.setEnabled(True)
                QMessageBox.critical(self, "Error", f"Failed to load analyzer: {str(e)}")
                return
        
        # Update settings from the GUI
        analyze_cards = self.analyzer
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
        
        # Apply all tunable values from sliders
        # Fast-pass tunables
        analyze_cards.NEAR_HIT_FOCUS_SECS = self.near_hit_focus_slider.value() / 10.0
        analyze_cards.POST_CONFIRM_FOCUS_SECS = self.post_confirm_focus_slider.value() / 10.0
        analyze_cards.NEAR_HIT_DEBOUNCE_SECS = self.near_hit_debounce_slider.value() / 10.0
        analyze_cards.BASELINE_SKIP_STOP = self.baseline_skip_stop_slider.value()
        analyze_cards.BASELINE_SKIP_OTHER = self.baseline_skip_other_slider.value()
        analyze_cards.NEAR_HIT_SUPPRESSION_WINDOW_SECS = self.near_hit_suppression_slider.value() / 10.0
        analyze_cards.GLARE_CONFIRMATION_DEBOUNCE_SECS = self.glare_debounce_slider.value() / 10.0
        analyze_cards.OCR_FUZZY = self.ocr_fuzzy_slider.value() / 100.0
        analyze_cards.LOOSE_HUE_FOCUS = self.loose_hue_focus_slider.value() / 100.0
        
        # Detection strength thresholds
        analyze_cards.OCR_STRONG = self.ocr_strong_slider.value() / 100.0
        analyze_cards.HUE_STRONG = self.hue_strong_slider.value() / 100.0
        analyze_cards.TEMPLATE_STRONG = self.template_strong_slider.value() / 100.0
        
        # Confirmation requirements
        analyze_cards.MIN_AREA_RATIO = self.min_area_ratio_slider.value() / 100.0
        analyze_cards.MIN_FRAMES = self.min_frames_slider.value()
        analyze_cards.CONFIRMATION_DEBOUNCE_SECS = self.confirmation_debounce_slider.value() / 10.0
        
        # Update settings based on resolution selection
        if analyze_cards.RESOLUTION_SETTING == "360p":
            analyze_cards.settings = getattr(analyze_cards, "SETTINGS_360P", DEFAULT_SETTINGS_360P)
        elif analyze_cards.RESOLUTION_SETTING == "720p":
            analyze_cards.settings = getattr(analyze_cards, "SETTINGS_720P", DEFAULT_SETTINGS_720P)
        elif analyze_cards.RESOLUTION_SETTING == "1080p":
            analyze_cards.settings = getattr(analyze_cards, "SETTINGS_1080P", DEFAULT_SETTINGS_1080P)
            
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
        
        # Create and start the analysis worker thread
        self.analysis_worker = AnalysisWorker(analyze_cards, self.input_folder.text())
        self.analysis_worker.progress_updated.connect(self.on_progress_updated)
        self.analysis_worker.analysis_finished.connect(self.on_analysis_finished)
        self.analysis_worker.analysis_error.connect(self.on_analysis_error)
        
        # Show progress bar and label, disable run button
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0%")
        self.run_btn.setEnabled(False)
        
        # Start the worker thread
        self.analysis_worker.start()
    
    def on_progress_updated(self, value, estimated_total_seconds, total_videos):
        """Update progress bar when worker emits progress signal"""
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"{value}%")
        
        # Calculate estimated videos completed based on progress percentage
        videos_completed = int((value / 100) * total_videos) if total_videos > 0 else 0
        
        # Calculate remaining time
        if value > 0 and value < 100:
            elapsed_per_percent = estimated_total_seconds / 100
            remaining_seconds = (100 - value) * elapsed_per_percent
            
            # Convert to hours and minutes
            hours = int(remaining_seconds // 3600)
            minutes = int((remaining_seconds % 3600) // 60)
            
            # Format time string
            if hours > 0:
                time_str = f"{hours}h {minutes}m"
            else:
                time_str = f"{minutes}m"
            
            self.progress_label.setText(f"{videos_completed}/{total_videos} videos | Est. {time_str} remaining")
        else:
            self.progress_label.setText(f"{videos_completed}/{total_videos} videos")
        
        QApplication.processEvents()
    
    def on_analysis_finished(self):
        """Handle analysis completion"""
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("100%")
        self.progress_label.setText("Complete!")
        self.run_btn.setText("Run Analysis")
        self.run_btn.setEnabled(True)
        
        output_file = self.output_file.text()
        
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
        
        # Hide progress bar after a short delay
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
    
    def on_analysis_error(self, error_message):
        """Handle analysis errors"""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.run_btn.setText("Run Analysis")
        self.run_btn.setEnabled(True)
        
        QMessageBox.critical(self, "Error", f"An error occurred: {error_message}")
        import traceback
        traceback.print_exc()
    
    def update_near_hit_focus(self, value):
        """Update Near Hit Focus Time value."""
        actual_value = value / 10.0
        self.near_hit_focus_label.setText(f"{actual_value:.1f}")

    def update_post_confirm_focus(self, value):
        """Update Post Confirmation Focus Time value."""
        actual_value = value / 10.0
        self.post_confirm_focus_label.setText(f"{actual_value:.1f}")

    def update_near_hit_debounce(self, value):
        """Update Near Hit Debounce value."""
        actual_value = value / 10.0
        self.near_hit_debounce_label.setText(f"{actual_value:.1f}")

    def update_baseline_skip_stop(self, value):
        """Update Baseline Skip (STOP) value."""
        self.baseline_skip_stop_label.setText(f"{value}")

    def update_baseline_skip_other(self, value):
        """Update Baseline Skip (Other) value."""
        self.baseline_skip_other_label.setText(f"{value}")

    def update_near_hit_suppression(self, value):
        """Update Near Hit Suppression Window value."""
        actual_value = value / 10.0
        self.near_hit_suppression_label.setText(f"{actual_value:.1f}")

    def update_glare_debounce(self, value):
        """Update Glare Confirmation Debounce value."""
        actual_value = value / 10.0
        self.glare_debounce_label.setText(f"{actual_value:.1f}")

    def update_ocr_fuzzy(self, value):
        """Update OCR Fuzzy Threshold value."""
        actual_value = value / 100.0
        self.ocr_fuzzy_label.setText(f"{actual_value:.2f}")

    def update_loose_hue_focus(self, value):
        """Update Loose Hue Focus Threshold value."""
        actual_value = value / 100.0
        self.loose_hue_focus_label.setText(f"{actual_value:.2f}")

    def update_ocr_strong(self, value):
        """Update OCR Strong Confidence value."""
        actual_value = value / 100.0
        self.ocr_strong_label.setText(f"{actual_value:.2f}")

    def update_hue_strong(self, value):
        """Update Hue Strong Threshold value."""
        actual_value = value / 100.0
        self.hue_strong_label.setText(f"{actual_value:.2f}")

    def update_template_strong(self, value):
        """Update Template Strong Threshold value."""
        actual_value = value / 100.0
        self.template_strong_label.setText(f"{actual_value:.2f}")

    def update_min_area_ratio(self, value):
        """Update Min Area Ratio value."""
        actual_value = value / 100.0
        self.min_area_ratio_label.setText(f"{actual_value:.2f}")

    def update_min_frames(self, value):
        """Update Min Frames value."""
        self.min_frames_label.setText(f"{value}")

    def update_confirmation_debounce(self, value):
        """Update Confirmation Debounce value."""
        actual_value = value / 10.0
        self.confirmation_debounce_label.setText(f"{actual_value:.1f}")

    def update_area_value(self, value):
        """Update the min area ratio value label when slider changes."""
        ratio = value / 100.0
        self.area_label.setText(f"{ratio:.2f}")
        
        # Check if this matches the recommended value for current resolution
        if self.btn_360p.isChecked():
            settings = self._get_settings("360p")
        elif self.btn_720p.isChecked():
            settings = self._get_settings("720p")
        elif self.btn_1080p.isChecked():
            settings = self._get_settings("1080p")
        else:
            settings = self._get_settings("360p")
            
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
            settings = self._get_settings("360p")
        elif self.btn_720p.isChecked():
            settings = self._get_settings("720p")
        elif self.btn_1080p.isChecked():
            settings = self._get_settings("1080p")
        else:
            settings = self._get_settings("360p")
            
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
            settings = self._get_settings("360p")
        elif self.btn_720p.isChecked():
            settings = self._get_settings("720p")
        elif self.btn_1080p.isChecked():
            settings = self._get_settings("1080p")
        else:
            settings = self._get_settings("360p")
            
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
            # Expand height to show all sliders, but keep original width
            self.adjustSize()
            if self.original_size is not None:
                self.resize(self.original_size.width(), self.height())
        else:
            self.advanced_button.setText("▶ Show Advanced Options")
            # Restore to original window size
            if self.original_size is not None:
                self.resize(self.original_size)
    
    def showEvent(self, event):
        """Capture the original window size when first shown"""
        super().showEvent(event)
        # Store the original size on first show (after layout is complete)
        if self.original_size is None:
            self.original_size = self.size()
    
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