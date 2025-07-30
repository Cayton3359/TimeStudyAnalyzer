# Time Study Analyzer

A computer vision-based system for analyzing time study videos using colored cards to track different work states.

## Features

- **Multi-resolution support**: 360p, 720p, and 1080p video processing
- **Real-time card detection**: Detects START, STOP, DOWN, LOOK, and MEETING cards
- **Intelligent processing**: Uses OCR, template matching, color detection, and edge detection
- **Excel reporting**: Generates comprehensive reports with timing summaries
- **GUI interface**: User-friendly PyQt5 interface for easy operation
- **Multi-threaded processing**: Optimized for performance using 60% of available CPU cores

## Installation

### Prerequisites

- Python 3.7+
- OpenCV
- PyQt5
- Tesseract OCR
- Required Python packages (install via `pip install -r requirements.txt`)

### Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Install Tesseract OCR from https://github.com/UB-Mannheim/tesseract/wiki
4. Place your card templates in the `templates/` folder
5. Update paths in the configuration as needed

## Usage

### GUI Mode
```bash
python main.py
```

### Command Line Mode
```bash
python src/analyzer/analyze_cards.py
```

## Configuration

### Resolution Settings
- **360p**: Optimized for speed with 40% frame resize
- **720p**: Balanced performance with 25% frame resize  
- **1080p**: High quality with 17% frame resize (equivalent processing to 720p)

### Detection Parameters
- **Min Area Ratio**: Minimum card size as percentage of frame
- **Debounce Time**: Minimum seconds between same card detections
- **Frame Skip**: Process every Nth frame for performance
- **OCR Confidence**: Threshold for text recognition accuracy

## Card Types

- **START**: Green cards indicating work start
- **STOP**: Red cards indicating work stop
- **DOWN**: Orange cards indicating downtime
- **LOOK**: Blue cards indicating inspection/looking
- **MEETING**: Purple cards indicating meetings

## Output

The system generates Excel reports with:
1. **Summary Statistics**: Total time for each work state
2. **LOOK Events**: Detailed list of all inspection events
3. **Raw Detections**: Complete chronological list of all card detections

## Directory Structure

```
TimeStudyAnalyzer/
├── src/
│   ├── analyzer/
│   │   └── analyze_cards.py    # Core analysis engine
│   └── gui/
│       └── main_window.py      # PyQt5 GUI interface
├── templates/                  # Card template images
├── output/                     # Generated reports
├── main.py                     # GUI launcher
└── README.md
```

## Performance

- **Multi-threaded**: Uses up to 60% of CPU cores for optimal performance
- **Optimized processing**: Frame skipping and resize factors for speed
- **Memory efficient**: Streaming video processing without loading entire files

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

[Add your license information here]

## Author

Lindsey - Engineering Department
