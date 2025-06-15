import os
import sys

sys.path.append(os.path.abspath("."))
import shutil
from datetime import datetime
from pathlib import Path
import ctypes
import win32gui
import win32process
import psutil
import win32con
import importlib
import inspect
import os
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
        self._env_running = False
        self.setupUi(self)
        self.solution = solution
        self._screen = Screen()
        self._screen.geometryChanged = self.sync_spinbox_with_screen
        self.spbx_screen_x.setValue(getattr(self.solution, "screen_x", 0))
        self.spbx_screen_y.setValue(getattr(self.solution, "screen_y", 0))
        self.spbx_screen_w.setValue(getattr(self.solution, "screen_w", 0))
        self.spbx_screen_h.setValue(getattr(self.solution, "screen_h", 0))
        self.fps_real_time_view = 30
        self._last_window_list = []
        self.update_target_window_list()
        if self.solution.target_window:
            idx = self.cbox_target_window.findText(self.solution.target_window)
            if idx != -1:
                self.cbox_target_window.setCurrentIndex(idx)

        self.update_select_game_list()
        if self.solution.game:
            idx = self.cbox_select_game.findText(self.solution.game)
            if idx != -1:
                self.cbox_select_game.setCurrentIndex(idx)

        self.btn_close.clicked.connect(self.slot_btn_close)
        self.btn_show_screen.clicked.connect(self.slot_btn_show_screen)
        self.ckbx_real_time_view.stateChanged.connect(self.slot_ckbx_real_time_view)
        self.cbox_target_window.currentTextChanged.connect(
            self.slot_target_window_changed
        )
        self.btn_save.clicked.connect(self.slot_btn_save)
        self._window_list_timer = QTimer(self)
        self._window_list_timer.timeout.connect(self.update_target_window_list)
        self._window_list_timer.start(100)
        self._connect_spinbox_clamp()
        self.btn_self_play.clicked.connect(lambda: self.slot_btn_env_play("self_play"))
        self.btn_random_play.clicked.connect(
            lambda: self.slot_btn_env_play("random_play")
        )
        self.btn_train.clicked.connect(lambda: self.slot_btn_env_play("train"))
        self.btn_test.clicked.connect(lambda: self.slot_btn_env_play("test"))
        self.cbox_target_window.wheelEvent = lambda event: None
        self.cbox_select_game.wheelEvent = lambda event: None
        self.update()

    def update_select_game_list(self):
        """
        source/envs/__init__.py에서 from ... import ... 또는 __all__에 명시된 클래스만 cbox_select_game에 추가합니다.
        """

        # source.envs를 import하여, __all__ 또는 globals()에서 클래스만 추출
        try:
            envs_module = importlib.import_module("source.envs")
        except Exception:
            self.cbox_select_game.clear()
            return
        class_names = []
        # __all__이 있으면 그 안의 이름만, 없으면 globals()에서 직접 추출
        names = getattr(envs_module, "__all__", None)
        if names is None:
            names = [
                name for name, obj in vars(envs_module).items() if inspect.isclass(obj)
            ]
        for name in names:
            obj = getattr(envs_module, name, None)
            if inspect.isclass(obj):
                class_names.append(name)
        self.cbox_select_game.clear()
        self.cbox_select_game.addItems(class_names)

    def update_target_window_list(self):
        current_text = self.cbox_target_window.currentText()
        solution_target = self.solution.target_window

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

        items = enum_windows()
        # None 선택 가능하도록 맨 앞에 추가
        items.insert(0, "None")
        missing_target = bool(solution_target and solution_target not in items)
        if missing_target:
            items.append(f"(miss) {solution_target}")
        if set(items) != set(self._last_window_list):
            self.cbox_target_window.blockSignals(True)
            self.cbox_target_window.clear()
            for item in items:
                self.cbox_target_window.addItem(item)
            idx = self.cbox_target_window.findText(current_text)
            if idx != -1:
                self.cbox_target_window.setCurrentIndex(idx)
            elif missing_target:
                idx = self.cbox_target_window.findText(f"(miss) {solution_target}")
                if idx != -1:
                    self.cbox_target_window.setCurrentIndex(idx)
            else:
                # 아무것도 선택 안 했으면 None 선택
                self.cbox_target_window.setCurrentIndex(0)
            self.cbox_target_window.blockSignals(False)
            self._last_window_list = items

    def _get_env(self, env_text):
        """
        주어진 env_text에 해당하는 환경 클래스를 가져옵니다.
        """
        try:
            module = importlib.import_module(f"source.envs.{env_text.lower()}")
            env_class = getattr(module, env_text, None)
            if env_class is None:
                raise ImportError(f"{env_text} 클래스가 {module.__name__}에 없습니다.")
            return env_class
        except ImportError as e:
            QMessageBox.critical(
                self, "오류", f"게임을 불러오는 중 오류가 발생했습니다: {e}"
            )
            return None

    def slot_btn_env_play(self, mode):
        # 환경 실행 중인지 인스턴스 변수로 관리
        if self._env_running:
            QMessageBox.warning(
                self,
                "실행 중",
                "이 솔루션에서 가상 환경이 이미 실행 중입니다. 먼저 기존 환경을 종료하세요.",
            )
            return
        self._env_running = True
        try:
            env = self._get_env(self.cbox_select_game.currentText())
            if env is None:
                self._env_running = False
                return
            env = env()
            env.play(str(self.solution.root / self.solution.name), mode)
        finally:
            self._env_running = False

    def slot_btn_close(self):
        reply = QMessageBox.question(
            self,
            "확인",
            "정말 종료하시겠습니까?",
            QMessageBox.Ok | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if reply != QMessageBox.Ok:
            return
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
        if self._screen.isVisible():
            self._screen.close()
        else:
            base_x, base_y = self.get_target_window_position()
            x = base_x + self.spbx_screen_x.value()
            y = base_y + self.spbx_screen_y.value()
            w = self.spbx_screen_w.value()
            h = self.spbx_screen_h.value()
            self._screen.setGeometry(x, y, w, h)
            self._screen.show()

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
            scaled_pixmap = pixmap.scaled(view_size, aspectMode=Qt.KeepAspectRatio)
            scene = QGraphicsScene()
            scene.addPixmap(scaled_pixmap)
            self.graphicsView.setScene(scene)
        else:
            QMessageBox.warning(self, "Error", "화면 캡처를 할 수 없습니다.")

    def update(self):
        self.lbl_solution_root.setText(str(self.solution.root))
        self.lbl_solution_name.setText(str(self.solution.name))
        self.update_target_window_list()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            event.ignore()
        else:
            super().keyPressEvent(event)

    def update_solution_from_ui(self):
        """
        현재 UI의 값들로 self.solution 객체를 업데이트합니다.
        """
        self.solution.screen_x = self.spbx_screen_x.value()
        self.solution.screen_y = self.spbx_screen_y.value()
        self.solution.screen_w = self.spbx_screen_w.value()
        self.solution.screen_h = self.spbx_screen_h.value()
        target_text = self.cbox_target_window.currentText()
        if target_text == "None" or target_text.startswith("(miss) "):
            self.solution.target_window = None
        else:
            self.solution.target_window = target_text

        self.solution.game = self.cbox_select_game.currentText()

    def slot_btn_save(self):
        self.update_solution_from_ui()  # UI 값으로 solution 업데이트
        try:
            self.solution.save_json()
            QMessageBox.information(
                self, "저장 완료", "Solution 정보가 저장되었습니다."
            )
        except Exception as e:
            QMessageBox.critical(self, "저장 실패", f"저장 중 오류 발생: {e}")

    def slot_ckbx_real_time_view(self):
        if self.ckbx_real_time_view.isChecked():
            self._start_real_time_preview()
        else:
            if hasattr(self, "_preview_timer"):
                self._preview_timer.stop()
            self.graphicsView.setScene(None)

    def _get_window_client_offset(self, hwnd):
        win_left, win_top, _, _ = win32gui.GetWindowRect(hwnd)
        pt = win32gui.ClientToScreen(hwnd, (0, 0))
        client_left, client_top = pt
        offset_x = client_left - win_left
        offset_y = client_top - win_top
        return offset_x, offset_y

    def _start_real_time_preview(self):
        def update_preview():
            base_x, base_y = self.get_target_window_position()
            x = base_x + self.spbx_screen_x.value()
            y = base_y + self.spbx_screen_y.value()
            w = self.spbx_screen_w.value()
            h = self.spbx_screen_h.value()
            screen = QApplication.primaryScreen()
            if screen:
                pixmap = screen.grabWindow(0, x, y, w, h)
                view_rect = self.graphicsView.viewport().rect()
                view_size = view_rect.size()
                scaled_pixmap = pixmap.scaled(view_size, aspectMode=Qt.KeepAspectRatio)
                scene = QGraphicsScene()
                scene.addPixmap(scaled_pixmap)
                self.graphicsView.setScene(scene)
            else:
                self.graphicsView.setScene(None)

        if not hasattr(self, "_preview_timer"):
            self._preview_timer = QTimer(self)
            self._preview_timer.timeout.connect(update_preview)
        self._preview_timer.start(int(1000 / self.fps_real_time_view))
        update_preview()

    def sync_spinbox_to_solution(self):
        pass

    def slot_target_window_changed(self, text):
        self._clamp_spinbox_to_target_window()
        pass

    def sync_spinbox_with_screen(self):
        geom = self._screen.geometry()
        base_x, base_y = self.get_target_window_position()
        self.spbx_screen_x.setValue(geom.x() - base_x)
        self.spbx_screen_y.setValue(geom.y() - base_y)
        self.spbx_screen_w.setValue(geom.width())
        self.spbx_screen_h.setValue(geom.height())

    def clamp_spinbox_to_target_window(self):
        target_text = self.cbox_target_window.currentText()
        if not target_text or target_text.startswith("(miss) "):
            return
        hwnd = None

        def enum_hwnds():
            result = []

            def callback(hwnd, _):
                if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
                    tid, pid = win32process.GetWindowThreadProcessId(hwnd)
                    try:
                        proc = psutil.Process(pid)
                        exe = proc.name()
                    except Exception:
                        exe = ""
                    title = win32gui.GetWindowText(hwnd)
                    result.append((hwnd, f"{title} ({exe})"))

            win32gui.EnumWindows(callback, None)
            return result

        for hwnd_enum, item_text in enum_hwnds():
            if item_text == target_text:
                hwnd = hwnd_enum
                break
        if hwnd:
            left, top, right, bottom = win32gui.GetClientRect(hwnd)
            width = right - left
            height = bottom - top
            w = self.spbx_screen_w.value()
            h = self.spbx_screen_h.value()
            max_x = max(0, width - w)
            max_y = max(0, height - h)
            x = self.spbx_screen_x.value()
            y = self.spbx_screen_y.value()
            clamped_x = min(max(0, x), max_x)
            clamped_y = min(max(0, y), max_y)
            if clamped_x != x:
                self.spbx_screen_x.setValue(clamped_x)
            if clamped_y != y:
                self.spbx_screen_y.setValue(clamped_y)

    def _clamp_spinbox_to_target_window(self):
        target_text = self.cbox_target_window.currentText()
        if not target_text or target_text.startswith("(miss) "):
            return
        hwnd = None

        def enum_hwnds():
            result = []

            def callback(hwnd, _):
                if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
                    tid, pid = win32process.GetWindowThreadProcessId(hwnd)
                    try:
                        proc = psutil.Process(pid)
                        exe = proc.name()
                    except Exception:
                        exe = ""
                    title = win32gui.GetWindowText(hwnd)
                    result.append((hwnd, f"{title} ({exe})"))

            win32gui.EnumWindows(callback, None)
            return result

        for hwnd_enum, item_text in enum_hwnds():
            if item_text == target_text:
                hwnd = hwnd_enum
                break
        if hwnd:
            left, top, right, bottom = win32gui.GetClientRect(hwnd)
            client_w = right - left
            client_h = bottom - top
            # w, h가 target window의 크기를 넘지 않게 clamp
            w = self.spbx_screen_w.value()
            h = self.spbx_screen_h.value()
            clamped_w = min(w, client_w)
            clamped_h = min(h, client_h)
            if clamped_w != w:
                self.spbx_screen_w.setValue(clamped_w)
            if clamped_h != h:
                self.spbx_screen_h.setValue(clamped_h)
            # x, y도 기존대로 clamp
            max_x = max(0, client_w - self.spbx_screen_w.value())
            max_y = max(0, client_h - self.spbx_screen_h.value())
            x = self.spbx_screen_x.value()
            y = self.spbx_screen_y.value()
            clamped_x = min(max(0, x), max_x)
            clamped_y = min(max(0, y), max_y)
            if clamped_x != x:
                self.spbx_screen_x.setValue(clamped_x)
            if clamped_y != y:
                self.spbx_screen_y.setValue(clamped_y)

    def _connect_spinbox_clamp(self):
        # Connect valueChanged signals to clamping
        self.spbx_screen_x.valueChanged.connect(self._clamp_spinbox_to_target_window)
        self.spbx_screen_y.valueChanged.connect(self._clamp_spinbox_to_target_window)
        self.spbx_screen_w.valueChanged.connect(self._clamp_spinbox_to_target_window)
        self.spbx_screen_h.valueChanged.connect(self._clamp_spinbox_to_target_window)
        self.cbox_target_window.currentTextChanged.connect(
            self._clamp_spinbox_to_target_window
        )

    def get_target_window_position(self):
        """
        cbox_target_window에서 선택된 윈도우가 존재하면 해당 윈도우의 클라이언트 영역 (x, y) 좌표를 반환.
        최소화 등으로 좌표가 -32000 이하로 나오면 마지막 정상 좌표를 반환.
        존재하지 않으면 (0, 0) 반환.
        """
        if not hasattr(self, "_last_client_pos"):
            self._last_client_pos = (0, 0)
        target_text = self.cbox_target_window.currentText()
        if not target_text or target_text.startswith("(miss) "):
            return self._last_client_pos
        hwnd = None

        def enum_hwnds():
            result = []

            def callback(hwnd, _):
                if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
                    tid, pid = win32process.GetWindowThreadProcessId(hwnd)
                    try:
                        proc = psutil.Process(pid)
                        exe = proc.name()
                    except Exception:
                        exe = ""
                    title = win32gui.GetWindowText(hwnd)
                    result.append((hwnd, f"{title} ({exe})"))

            win32gui.EnumWindows(callback, None)
            return result

        for hwnd_enum, item_text in enum_hwnds():
            if item_text == target_text:
                hwnd = hwnd_enum
                break
        if hwnd:
            client_x, client_y = win32gui.ClientToScreen(hwnd, (0, 0))
            if client_x < 0 or client_y < 0:
                return self._last_client_pos
            self._last_client_pos = (client_x, client_y)
            return client_x, client_y
        else:
            return self._last_client_pos


if __name__ == "__main__":
    solution = Solution.load_json(
        os.path.join(os.getcwd(), "New_Solution", "New_Solution.json")
    )
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme="dark_cyan.xml")
    window = WdgtBaseTab(None, solution)
    window.show()
    sys.exit(app.exec())
