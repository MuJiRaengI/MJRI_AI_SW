import os
import sys
from PySide6.QtWidgets import QMainWindow, QApplication
from ui.designer.ui_main import Ui_main_window
from qt_material import apply_stylesheet

from ui.main_ui import MainWindow

from source.envs import *

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch

GPU_MEMORY = 8
torch.cuda.set_per_process_memory_fraction(GPU_MEMORY / 12, device=0)


class MJRIAISW:
    def __init__(self):
        super().__init__()
        self.ui = None
        self.solutions = {}

    def show(self):
        app = QApplication(sys.argv)
        apply_stylesheet(app, theme="dark_cyan.xml")
        self.ui = MainWindow()
        self.ui.show()
        sys.exit(app.exec())


if __name__ == "__main__":
    app = MJRIAISW()
    app.show()
