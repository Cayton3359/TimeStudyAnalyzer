import PyInstaller.__main__
import os
import sys

# Get absolute path to project root
project_root = os.path.dirname(os.path.abspath(__file__))

# Run PyInstaller
PyInstaller.__main__.run([
    os.path.join(project_root, 'main.py'),  # Main script is now in root
    '--name=TimeStudyAnalyzer',
    '--onefile',
    '--windowed',
    f'--distpath={os.path.join(project_root, "dist")}',
    f'--workpath={os.path.join(project_root, "build")}',
    f'--add-data={os.path.join(project_root, "templates")}{os.pathsep}templates',
    '--clean',
])

print(f"Build complete! Executable is in: {os.path.join(project_root, 'dist')}")