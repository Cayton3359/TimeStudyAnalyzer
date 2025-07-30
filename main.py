import sys
import os
import importlib.util

# Direct import method - no package needed
main_window_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                            "src", "gui", "main_window.py")

# Import the module directly from file
spec = importlib.util.spec_from_file_location("main_window", main_window_path)
main_window = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_window)

# Get the MainWindow class from the loaded module
MainWindow = main_window.MainWindow

from PyQt5.QtWidgets import QApplication

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()