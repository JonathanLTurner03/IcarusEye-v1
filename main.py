import sys
from PyQt6.QtWidgets import QApplication
from src.ui.main_window import MainWindow
from src.ui.loading_screen import LoadingScreen
import time


def main():
    app = QApplication(sys.argv)

    # Show loading screen
    loading_screen = LoadingScreen()
    loading_screen.show()

    # Initialize the main window
    main_window = MainWindow()

    time.sleep(1)
    # Hide loading screen and show main window
    loading_screen.close()
    main_window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
