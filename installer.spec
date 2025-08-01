# -*- mode: python ; coding: utf-8 -*-
import os

block_cipher = None

# Define the installer analysis
installer_a = Analysis(
    ['installer.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('dist/TimeStudyAnalyzer-Complete/TimeStudyAnalyzer-GUI.exe', '.'),
        ('dist/TimeStudyAnalyzer-Complete/TimeStudyAnalyzer-CLI.exe', '.'),
        ('templates', 'templates'),
        ('Que Cards.pptx', '.'),
        ('User-Guide.md', '.'),
        ('README.md', '.'),
    ],
    hiddenimports=[
        'tkinter',
        'tkinter.ttk',
        'tkinter.messagebox',
        'urllib.request',
        'zipfile',
        'shutil',
        'tempfile',
        'threading',
        'webbrowser',
        'subprocess',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Create the installer executable
installer_pyz = PYZ(installer_a.pure, installer_a.zipped_data, cipher=block_cipher)

installer_exe = EXE(
    installer_pyz,
    installer_a.scripts,
    installer_a.binaries,
    installer_a.zipfiles,
    installer_a.datas,
    [],
    name='TimeStudyAnalyzer-Setup',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='templates/icon.ico' if os.path.exists('templates/icon.ico') else None,
)
