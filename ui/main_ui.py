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
from ui.base_tab_wdgt import WdgtBaseTab

from source.solution.solution import Solution


class MainWindow(QMainWindow, Ui_main_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.btn_new_solution.clicked.connect(self.slot_create_solution)
        self.btn_open_solution.clicked.connect(self.slot_open_solution)

        self.create_solution_window = CreateSolutionWindow()

    def find_tab_by_solution_path(self, solution):
        """solution의 전체 경로와 일치하는 탭이 있으면 인덱스를 반환, 없으면 -1 반환"""
        solution_path = str(
            (solution.root / solution.name / solution.json_name).resolve()
        )
        for i in range(self.tblw_main.count()):
            tab_widget = self.tblw_main.widget(i)
            if hasattr(tab_widget, "solution"):
                tab_solution = tab_widget.solution
                tab_path = str(
                    (
                        tab_solution.root / tab_solution.name / tab_solution.json_name
                    ).resolve()
                )
                if tab_path == solution_path:
                    return i
        return -1

    def slot_create_solution(self):
        solution = Solution()
        dialog = CreateSolutionWindow(solution=solution)
        if dialog.exec() == QDialog.Accepted:
            index = self.find_tab_by_solution_path(solution)
            if index != -1:
                self.tblw_main.setCurrentIndex(index)
                return
            solution_name = (
                solution.name if isinstance(solution.name, str) else str(solution.name)
            )
            self.tblw_main.insertTab(
                self.tblw_main.count(),
                WdgtBaseTab(self.tblw_main, solution),
                solution_name,
            )
            self.tblw_main.setCurrentIndex(self.tblw_main.count() - 1)

    def slot_open_solution(self):
        file_path = self.get_open_file_path(
            title="Open Solution", file_filter="JSON Files (*.json)"
        )
        if file_path:
            solution = Solution.load_json(file_path)
            index = self.find_tab_by_solution_path(solution)
            if index != -1:
                self.tblw_main.setCurrentIndex(index)
                return
            solution_name = (
                solution.name if isinstance(solution.name, str) else str(solution.name)
            )
            self.tblw_main.insertTab(
                self.tblw_main.count(),
                WdgtBaseTab(self.tblw_main, solution),
                solution_name,
            )
            self.tblw_main.setCurrentIndex(self.tblw_main.count() - 1)

    def show_info_message(self, title, message):
        QMessageBox.information(self, title, message)

    def get_save_file_path(self, title="Save File", file_filter="All Files (*.*)"):
        file_path, _ = QFileDialog.getSaveFileName(self, title, "", file_filter)
        return file_path

    def get_folder_path(self, title="Select Folder"):
        folder_path = QFileDialog.getExistingDirectory(self, title, "")
        return folder_path

    def get_open_file_path(self, title="Open File", file_filter="All Files (*.*)"):
        file_path, _ = QFileDialog.getOpenFileName(self, title, "", file_filter)
        return file_path


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    apply_stylesheet(app, theme="dark_cyan.xml")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
