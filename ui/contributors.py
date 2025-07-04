import os
import sys
import shutil
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.abspath("."))

from PySide6.QtWidgets import QMainWindow, QApplication
from PySide6.QtWidgets import QMessageBox, QFileDialog
from PySide6.QtWidgets import QDialog
from qt_material import apply_stylesheet

from ui.designer.ui_contributors import Ui_Contributors
from source.solution.solution import Solution


class ContributorsWindow(QDialog, Ui_Contributors):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        # label size is 15
        self.lbl_title.setStyleSheet(
            "font-size: 30px; font-weight: bold; color: rgb(0, 255, 255);"
        )

        self.lbl_chzzk.setStyleSheet(
            "font-size: 20px; color: rgb(0, 247, 163); background-color: rgb(0, 0, 0);"
            "padding-top: 5px; padding-bottom: 5px;"
        )
        self.lbl_youtube.setStyleSheet(
            "font-size: 20px; color: rgb(255, 255, 255); background-color: rgb(254, 0, 50);"
            "padding-top: 5px; padding-bottom: 5px;"
        )
        self.lbl_git.setStyleSheet(
            "font-size: 20px; color: rgb(255, 255, 255); background-color: rgb(18, 20, 23);"
            "padding-top: 5px; padding-bottom: 5px;"
        )
