import os
import sys
import shutil
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.abspath("."))

import ctypes
import win32gui
import win32process
import psutil

from PySide6.QtWidgets import QMainWindow, QApplication
from PySide6.QtWidgets import QMessageBox, QFileDialog
from PySide6.QtWidgets import QDialog
from PySide6.QtWidgets import QTabWidget
from qt_material import apply_stylesheet

from ui.designer.ui_base_tab import Ui_wdgt_base_tab
from ui.screen_window import Screen
from source.solution.solution import Solution


class WdgtBaseTab(QDialog, Ui_wdgt_base_tab):
    def __init__(self, parent=None, solution: Solution = None):
        super().__init__(parent)
        self.setupUi(self)
        self.solution = solution
        self.win_screen = Screen(self)

        # signal-slot connections
        self.btn_close.clicked.connect(self.slot_btn_close)
        self.btn_show_screen.clicked.connect(self.slot_btn_show_screen)
        self.btn_set_screen.clicked.connect(self.slot_btn_set_screen)

        self.update()

    def slot_btn_close(self):
        parent = self.parent()
        while parent is not None and not isinstance(parent, QTabWidget):
            parent = parent.parent()
        if parent is not None and isinstance(parent, QTabWidget):
            index = parent.indexOf(self)
            if index != -1:
                parent.removeTab(index)
        else:
            self.close()

    def slot_btn_show_screen(self):

        self.win_screen.resize(800, 60)
        self.win_screen.show()
        self.win_screen.update()

    def update(self):
        # Update the solution information displayed in the widget
        self.lbl_solution_root.setText(str(self.solution.root))
        self.lbl_solution_name.setText(str(self.solution.name))

        # cbox_target_window에 현재 열려있는 윈도우 전체(크롬, vscode, 메모장 등) 목록 추가

        def enum_windows():
            windows = []

            def callback(hwnd, _):
                if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
                    tid, pid = win32process.GetWindowThreadProcessId(hwnd)
                    try:
                        proc = psutil.Process(pid)
                        exe = proc.name()
                    except Exception:
                        exe = ""
                    title = win32gui.GetWindowText(hwnd)
                    windows.append(f"{title} ({exe})")

            win32gui.EnumWindows(callback, None)
            return windows

        self.cbox_target_window.clear()
        self.cbox_target_window.addItems(enum_windows())


if __name__ == "__main__":
    solution = Solution()

    solution.root = Path(os.getcwd())
    solution.name = Path("test")
    solution.json_name = Path("test.json")
    solution.task = "test_task"

    app = QApplication(sys.argv)
    apply_stylesheet(app, theme="dark_cyan.xml")
    window = WdgtBaseTab(None, solution)
    window.show()
    sys.exit(app.exec())
