from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QApplication
from PyQt6.QtCore import Qt


class LoadingScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setFixedSize(300, 200)
        self.setStyleSheet("background-color: #2E2E2E; color: white;")

        layout = QVBoxLayout()
        self.label = QLabel("Loading, please wait...")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)
        self.setLayout(layout)