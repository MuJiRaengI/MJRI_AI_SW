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
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QGraphicsScene
from qt_material import apply_stylesheet

from ui.designer.ui_base_tab import Ui_wdgt_base_tab
from ui.screen_window import Screen
from source.solution.solution import Solution


class WdgtBaseTab(QDialog, Ui_wdgt_base_tab):
    def __init__(self, parent=None, solution: Solution = None):
        super().__init__(parent)
        self.setupUi(self)
        self.solution = solution
        self.win_screen = Screen()
        self.win_screen.geometryChanged = self.sync_spinbox_with_screen

        self.fps_real_time_view = 30

        # signal-slot connections
        self.btn_close.clicked.connect(self.slot_btn_close)
        self.btn_show_screen.clicked.connect(self.slot_btn_show_screen)
        self.ckbx_real_time_view.stateChanged.connect(self.slot_ckbx_real_time_view)

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
        # 스핀박스에서 좌표와 크기 읽기
        x = self.spbx_screen_x.value()
        y = self.spbx_screen_y.value()
        w = self.spbx_screen_w.value()
        h = self.spbx_screen_h.value()
        # self.win_screen을 해당 좌표와 크기로 이동 및 리사이즈
        self.win_screen.setGeometry(x, y, w, h)
        self.win_screen.show()
        self.win_screen.setGeometry(x, y, w, h)
        self.win_screen.update()
        # win_screen의 위치/크기가 변경될 때 spinbox 값도 동기화
        self.win_screen.geometryChanged = self.sync_spinbox_with_screen

    def sync_spinbox_with_screen(self):
        geom = self.win_screen.geometry()
        self.spbx_screen_x.setValue(geom.x())
        self.spbx_screen_y.setValue(geom.y())
        self.spbx_screen_w.setValue(geom.width())
        self.spbx_screen_h.setValue(geom.height())

    def slot_ckbx_real_time_view(self):
        if self.ckbx_real_time_view.isChecked():
            # 실시간 미리보기 시작
            self._start_real_time_preview()
        else:
            # 실시간 미리보기 중지 및 마지막 한 장만 표시
            if hasattr(self, "_preview_timer"):
                self._preview_timer.stop()
            self._show_single_preview()

    def _start_real_time_preview(self):
        def update_preview():
            x = self.spbx_screen_x.value()
            y = self.spbx_screen_y.value()
            w = self.spbx_screen_w.value()
            h = self.spbx_screen_h.value()
            screen = QApplication.primaryScreen()
            if screen:
                pixmap = screen.grabWindow(0, x, y, w, h)
                view_rect = self.graphicsView.viewport().rect()
                view_size = view_rect.size()
                # PySide6에서는 transformMode 인자를 지원하지 않으므로 제거
                scaled_pixmap = pixmap.scaled(view_size, aspectMode=Qt.KeepAspectRatio)
                scene = QGraphicsScene()
                scene.addPixmap(scaled_pixmap)
                self.graphicsView.setScene(scene)

        if not hasattr(self, "_preview_timer"):
            self._preview_timer = QTimer(self)
            self._preview_timer.timeout.connect(update_preview)
        self._preview_timer.start(1000 / self.fps_real_time_view)
        update_preview()

    def _show_single_preview(self):
        x = self.spbx_screen_x.value()
        y = self.spbx_screen_y.value()
        w = self.spbx_screen_w.value()
        h = self.spbx_screen_h.value()
        screen = QApplication.primaryScreen()
        if screen:
            pixmap = screen.grabWindow(0, x, y, w, h)
            view_rect = self.graphicsView.viewport().rect()
            view_size = view_rect.size()
            # PySide6에서는 transformMode 인자를 지원하지 않으므로 제거
            scaled_pixmap = pixmap.scaled(view_size, aspectMode=Qt.KeepAspectRatio)
            scene = QGraphicsScene()
            scene.addPixmap(scaled_pixmap)
            self.graphicsView.setScene(scene)
        else:
            QMessageBox.warning(self, "Error", "화면 캡처를 할 수 없습니다.")

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

        # Qt 위젯의 화면 기준 좌표
        global_pos = self.win_screen.mapToGlobal(self.win_screen.rect().topLeft())
        x = global_pos.x()
        y = global_pos.y()

        print(f"Screen position: ({x}, {y})")


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
