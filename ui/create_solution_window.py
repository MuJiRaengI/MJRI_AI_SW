import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath("."))

from PySide6.QtWidgets import QMainWindow, QApplication
from PySide6.QtWidgets import QMessageBox, QFileDialog
from PySide6.QtWidgets import QDialog
from qt_material import apply_stylesheet

from ui.designer.ui_create_solution import Ui_CreateSolutionWindow


class CreateSolutionWindow(QDialog, Ui_CreateSolutionWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.ledit_solution_path.setText(os.getcwd())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme="dark_cyan.xml")
    window = CreateSolutionWindow()
    window.show()
    sys.exit(app.exec())
