import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath("."))

from PySide6.QtWidgets import QMainWindow, QApplication
from PySide6.QtWidgets import QMessageBox, QFileDialog
from PySide6.QtWidgets import QDialog
from qt_material import apply_stylesheet

from ui.designer.ui_main import Ui_main_window
from ui.create_solution_window import CreateSolutionWindow


class MainWindow(QMainWindow, Ui_main_window):
    def __init__(self, solution=None):
        super().__init__()
        self.setupUi(self)
        self.solution = solution
        self.btn_new_solution.clicked.connect(self.create_solution)

        self.create_solution_window = CreateSolutionWindow()

    def create_solution(self):
        if self.solution:
            self.create_solution_window.exec()
        else:
            self.show_info_message("No Solution", "No solution provided to create.")

    def show_info_message(self, title, message):
        QMessageBox.information(self, title, message)

    def get_save_file_path(self, title="Save File", file_filter="All Files (*.*)"):
        file_path, _ = QFileDialog.getSaveFileName(self, title, "", file_filter)
        return file_path

    def get_folder_path(self, title="Select Folder"):
        folder_path = QFileDialog.getExistingDirectory(self, title, "")
        return folder_path


if __name__ == "__main__":
    import sys
    from source.solution.solution import Solution

    app = QApplication(sys.argv)
    apply_stylesheet(app, theme="dark_cyan.xml")
    window = MainWindow(Solution())
    window.show()
    sys.exit(app.exec())
