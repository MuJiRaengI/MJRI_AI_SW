import os
import sys
import shutil
from datetime import datetime

sys.path.append(os.path.abspath("."))

from PySide6.QtWidgets import QMainWindow, QApplication
from PySide6.QtWidgets import QMessageBox, QFileDialog
from PySide6.QtWidgets import QDialog
from qt_material import apply_stylesheet

from ui.designer.ui_create_solution import Ui_CreateSolutionWindow
from source.solution.solution import Solution


class CreateSolutionWindow(QDialog, Ui_CreateSolutionWindow):
    def __init__(self, parent=None, solution: Solution = None):
        super().__init__(parent)
        self.setupUi(self)
        self.ledit_solution_root.setText(os.getcwd())

        self.solution = solution if solution else Solution()

        self.btn_ok.clicked.connect(self.slot_btn_ok)
        self.btn_cancel.clicked.connect(self.slot_btn_cancel)
        self.btn_browse.clicked.connect(self.slot_btn_browse)

    def slot_btn_browse(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Solution Root Folder", os.getcwd()
        )
        if folder_path:
            self.ledit_solution_root.setText(folder_path)

    def slot_btn_ok(self):
        solution_name = self.ledit_solution_name.text().strip()
        solution_root = self.ledit_solution_root.text().strip()

        # 경로 객체로 변환
        solution_dir = os.path.join(solution_root, solution_name)
        json_name = f"{solution_name}.json"

        # 이미 폴더가 존재하면 처리
        if os.path.exists(solution_dir):
            if solution_name == "New_Solution":
                shutil.rmtree(solution_dir)
            else:
                QMessageBox.warning(self, "Warning", "Solution already exists.")
                return

        # 폴더 생성
        os.makedirs(solution_dir, exist_ok=True)
        QMessageBox.information(
            self,
            "Success",
            f"Solution '{solution_name}' created at {solution_root}.",
        )

        # Solution 객체 정보 저장
        self.solution.root = solution_root
        self.solution.name = solution_name
        self.solution.json_name = json_name
        self.solution.save_json()
        self.accept()

    def slot_btn_cancel(self):
        self.reject()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme="dark_cyan.xml")
    window = CreateSolutionWindow()
    window.show()
    sys.exit(app.exec())
