import sys
from PyQt6.QtWidgets import QApplication
from src.ui.main_window import MainWindow  # Import the MainWindow class

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Create and show the main window
    window = MainWindow()
    window.show()

    # Execute the application
    sys.exit(app.exec())
