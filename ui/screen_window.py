import os
import sys

sys.path.append(os.path.abspath("."))

from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QDialog
from qt_material import apply_stylesheet

from ui.designer.ui_screen import Ui_screen


class Screen(QDialog, Ui_screen):
    def __init__(self, parent=None, h=0, w=0):
        super().__init__(parent)
        self.setupUi(self)
        if h and w:
            self.resize(h, w)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme="dark_cyan.xml")
    window = Screen()
    window.show()
    sys.exit(app.exec())
