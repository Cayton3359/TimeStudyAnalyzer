# Time Study Analyzer - Complete Setup Guide

## 📦 What's Included

This package contains everything you need to run time study analysis:

- **TimeStudyAnalyzer-GUI.exe** - Easy-to-use graphical interface
- **TimeStudyAnalyzer-CLI.exe** - Command line version for advanced users
- **Que Cards.pptx** - PowerPoint file with printable colored cards
- **templates/** - Card detection templates (already configured)
- **README.md** - Detailed documentation
- **User-Guide.md** - This setup guide

## 🚀 Quick Start (5 Minutes)

### Step 1: Print the Cards
1. Open **Que Cards.pptx** in PowerPoint
2. Print on **color printer** (cards must be in color)
3. Cut out the cards along the lines
4. You now have: START (green), STOP (red), DOWN (orange), LOOK (blue), MEETING (purple)

### Step 2: Install Tesseract OCR
**Required for text recognition:**
1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install with default settings
3. Restart your computer

### Step 3: Setup Video Recording
1. Create folder: `D:\record\` (or choose your own location)
2. Record videos in this structure:
   ```
   D:\record\
   ├── 20250730\          (Date: YYYYMMDD)
   │   ├── 08\            (Hour: 24-hour format)
   │   │   ├── 00.mp4     (Minute the recording started)
   │   │   ├── 01.mp4
   │   │   └── 02.mp4
   │   └── 09\
   │       ├── 15.mp4
   │       └── 30.mp4
   ```

### Step 4: Run Analysis
1. **Double-click** `TimeStudyAnalyzer-GUI.exe`
2. **Set Video Folder** to your recording location (e.g., `D:\record\`)
3. **Choose resolution** (360p for speed, 1080p for quality)
4. **Click "Run Analysis"**
5. **Results** will be saved as Excel file with timing summaries

## 🎯 How to Use the Cards

### During Work Recording:
- **START (Green)** - Show when beginning work
- **STOP (Red)** - Show when stopping work  
- **DOWN (Orange)** - Show during downtime/waiting
- **LOOK (Blue)** - Show when inspecting/looking
- **MEETING (Purple)** - Show during meetings/discussions

### Best Practices:
- Hold cards clearly in front of camera for 2-3 seconds
- Ensure good lighting on the cards
- Cards should be at least 20% of the camera frame
- Wait 10 seconds between showing the same card

## 📊 Understanding Results

The Excel report contains 3 tabs:

1. **Summary Statistics** - Total time spent in each state
2. **LOOK Events** - Detailed list of all inspection activities  
3. **Raw Detections** - Complete chronological log with video links

## ⚙️ Advanced Options

### Resolution Settings:
- **360p** - Fastest processing, good for long recordings
- **720p** - Balanced speed and accuracy
- **1080p** - Best accuracy, slower processing

### Performance:
- Uses 60% of CPU cores automatically
- Processing time: ~1-2 minutes per hour of video
- Memory usage: ~500MB per hour of video

## 🔧 Troubleshooting

### "No detections found"
- Check lighting on cards
- Ensure cards are large enough in frame
- Verify Tesseract OCR is installed
- Try 720p or 1080p resolution

### "Can't open video files"
- Check folder structure matches: `YYYYMMDD\HH\MM.mp4`
- Ensure video files are .mp4 format
- Verify folder path is correct

### Slow processing
- Use 360p resolution for faster processing
- Close other programs to free up CPU
- Process smaller time ranges

### Update notifications
- App automatically checks for updates
- Click "Yes" to download latest version
- Replace old .exe files with new ones

## 📞 Support

- **GitHub Issues**: https://github.com/cayton3359/TimeStudyAnalyzer/issues
- **Documentation**: See README.md for technical details
- **Updates**: App will notify when new versions are available

## 📋 System Requirements

- **OS**: Windows 10/11
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 1GB free space
- **Camera**: Any video recording device
- **Printer**: Color printer for cards

---

**Version**: 1.0.0  
**Last Updated**: July 2025  
**Author**: Lindsey - Engineering Department
