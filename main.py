import sys
from PyQt6.QtWidgets import QApplication
from src.ui.main_window import MainWindow
from src.ui.loading_screen import LoadingScreen
import time

import cProfile
import pstats


def main():
    app = QApplication(sys.argv)

    # Show loading screen
    loading_screen = LoadingScreen()
    loading_screen.show()

    # Initialize the main window
    main_window = MainWindow()
    screen_size = app.primaryScreen().size()
    main_window.resize(int(screen_size.width()/1.25), int(screen_size.height()/1.5))
    main_window.setWindowTitle("IcarusEye - Drone Object Detection System")

    time.sleep(1)
    # Hide loading screen and show main window
    loading_screen.close()
    main_window.show()

    app.exec()


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    profiler.dump_stats('profile_data.prof')

    stats = pstats.Stats('profile_data.prof')
    stats.sort_stats('cumulative').print_stats()

