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
        self._last_window_list = []  # 마지막으로 표시한 윈도우 목록

        # Solution 값으로 초기화
        self.spbx_screen_x.setValue(self.solution.screen_x)
        self.spbx_screen_y.setValue(self.solution.screen_y)
        self.spbx_screen_w.setValue(self.solution.screen_w)
        self.spbx_screen_h.setValue(self.solution.screen_h)
        self.update_target_window_list()  # 콤보박스 목록 먼저 갱신
        if self.solution.target_window:
            idx = self.cbox_target_window.findText(self.solution.target_window)
            if idx != -1:
                self.cbox_target_window.setCurrentIndex(idx)

        # signal-slot connections
        self.btn_close.clicked.connect(self.slot_btn_close)
        self.btn_show_screen.clicked.connect(self.slot_btn_show_screen)
        self.ckbx_real_time_view.stateChanged.connect(self.slot_ckbx_real_time_view)
        self.cbox_target_window.currentTextChanged.connect(
            self.slot_target_window_changed
        )
        self.btn_save.clicked.connect(self.slot_btn_save)
        self.spbx_screen_x.valueChanged.connect(self.sync_spinbox_to_solution)
        self.spbx_screen_y.valueChanged.connect(self.sync_spinbox_to_solution)
        self.spbx_screen_w.valueChanged.connect(self.sync_spinbox_to_solution)
        self.spbx_screen_h.valueChanged.connect(self.sync_spinbox_to_solution)

        # 윈도우 목록 실시간 갱신 타이머
        self._window_list_timer = QTimer(self)
        self._window_list_timer.timeout.connect(self.update_target_window_list)
        self._window_list_timer.start(100)  # 0.1초마다 갱신

        self.real_x = 0
        self.real_y = 0

        self.update()

    def update_target_window_list(self):
        # 현재 선택된 텍스트를 기억
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
        # solution.target_window이 현재 목록에 없으면 추가
        missing_target = bool(solution_target and solution_target not in items)
        if missing_target:
            # (miss) prefix를 붙여서 추가
            items.append(f"(miss) {solution_target}")
        # 변경사항이 있을 때만 갱신
        if set(items) != set(self._last_window_list):
            self.cbox_target_window.blockSignals(True)
            self.cbox_target_window.clear()
            for item in items:
                self.cbox_target_window.addItem(item)
            # 기존 선택이 있으면 복원
            idx = self.cbox_target_window.findText(current_text)
            if idx != -1:
                self.cbox_target_window.setCurrentIndex(idx)
            elif missing_target:
                # (miss) prefix가 붙은 항목을 선택
                idx = self.cbox_target_window.findText(f"(miss) {solution_target}")
                if idx != -1:
                    self.cbox_target_window.setCurrentIndex(idx)
            self.cbox_target_window.blockSignals(False)
            self._last_window_list = items

    def slot_target_window_changed(self, text):
        # (miss) prefix가 있으면 제거해서 solution에 저장
        if text.startswith("(miss) "):
            self.solution.target_window = text[7:]
            # miss인 경우, real_x, real_y는 spinbox 값(절대좌표)
            self.real_x = self.spbx_screen_x.value()
            self.real_y = self.spbx_screen_y.value()
        else:
            self.solution.target_window = text
            # target_window가 miss가 아닌 경우, spinbox의 값을 target_window 기준 상대좌표로 변환하고,
            # real_x, real_y는 실제 절대좌표로 저장
            hwnd = None
            import win32gui, win32process, psutil

            # 현재 콤보박스 텍스트와 일치하는 윈도우 핸들 찾기
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
                if item_text == text:
                    hwnd = hwnd_enum
                    break
            if hwnd:
                win_left, win_top, win_right, win_bottom = win32gui.GetWindowRect(hwnd)
                rel_x = self.spbx_screen_x.value()
                rel_y = self.spbx_screen_y.value()
                offset_x, offset_y = self._get_window_client_offset(hwnd)
                self.real_x = win_left + offset_x + rel_x
                self.real_y = win_top + offset_y + rel_y
            else:
                self.real_x = self.spbx_screen_x.value()
                self.real_y = self.spbx_screen_y.value()

    def slot_btn_close(self):
        # win_screen이 열려 있으면 함께 닫기
        if self.win_screen.isVisible():
            self.win_screen.close()
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
        if self.win_screen.isVisible():
            self.win_screen.close()
            self.btn_show_screen.setText("Show Screen")
        else:
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
            self.btn_show_screen.setText("Hide Screen")
        # win_screen의 위치/크기가 변경될 때 spinbox 값도 동기화
        self.win_screen.geometryChanged = self.sync_spinbox_with_screen

    def sync_spinbox_with_screen(self):
        geom = self.win_screen.geometry()
        self.spbx_screen_x.setValue(geom.x())
        self.spbx_screen_y.setValue(geom.y())
        self.spbx_screen_w.setValue(geom.width())
        self.spbx_screen_h.setValue(geom.height())
        # spinbox가 바뀔 때 real_x, real_y도 갱신
        text = self.cbox_target_window.currentText()
        if text.startswith("(miss) "):
            self.real_x = geom.x()
            self.real_y = geom.y()
        else:
            hwnd = None
            import win32gui, win32process, psutil

            # 현재 콤보박스 텍스트와 일치하는 윈도우 핸들 찾기
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
                if item_text == text:
                    hwnd = hwnd_enum
                    break
            if hwnd:
                win_left, win_top, win_right, win_bottom = win32gui.GetWindowRect(hwnd)
                rel_x = geom.x()
                rel_y = geom.y()
                offset_x, offset_y = self._get_window_client_offset(hwnd)
                self.real_x = win_left + offset_x + rel_x
                self.real_y = win_top + offset_y + rel_y
            else:
                self.real_x = geom.x()
                self.real_y = geom.y()

    def sync_spinbox_to_solution(self):
        self.solution.screen_x = self.spbx_screen_x.value()
        self.solution.screen_y = self.spbx_screen_y.value()
        self.solution.screen_w = self.spbx_screen_w.value()
        self.solution.screen_h = self.spbx_screen_h.value()
        # spinbox가 바뀔 때 real_x, real_y도 갱신
        text = self.cbox_target_window.currentText()
        if text.startswith("(miss) "):
            self.real_x = self.spbx_screen_x.value()
            self.real_y = self.spbx_screen_y.value()
        else:
            hwnd = None
            import win32gui, win32process, psutil

            # 현재 콤보박스 텍스트와 일치하는 윈도우 핸들 찾기
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
                if item_text == text:
                    hwnd = hwnd_enum
                    break
            if hwnd:
                win_left, win_top, win_right, win_bottom = win32gui.GetWindowRect(hwnd)
                rel_x = self.spbx_screen_x.value()
                rel_y = self.spbx_screen_y.value()
                offset_x, offset_y = self._get_window_client_offset(hwnd)
                self.real_x = win_left + offset_x + rel_x
                self.real_y = win_top + offset_y + rel_y
            else:
                self.real_x = self.spbx_screen_x.value()
                self.real_y = self.spbx_screen_y.value()

    def _start_real_time_preview(self):
        def update_preview():
            text = self.cbox_target_window.currentText()
            if not text.startswith("(miss) ") and text:
                hwnd = None
                import win32gui, win32process, psutil

                def enum_hwnds():
                    result = []

                    def callback(hwnd, _):
                        if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(
                            hwnd
                        ):
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
                    if item_text == text:
                        hwnd = hwnd_enum
                        break
                if hwnd:
                    win_left, win_top, win_right, win_bottom = win32gui.GetWindowRect(
                        hwnd
                    )
                    rel_x = self.spbx_screen_x.value()
                    rel_y = self.spbx_screen_y.value()
                    offset_x, offset_y = self._get_window_client_offset(hwnd)
                    self.real_x = win_left + offset_x + rel_x
                    self.real_y = win_top + offset_y + rel_y
                else:
                    self.real_x = self.spbx_screen_x.value()
                    self.real_y = self.spbx_screen_y.value()
            else:
                self.real_x = self.spbx_screen_x.value()
                self.real_y = self.spbx_screen_y.value()
            x = self.real_x
            y = self.real_y
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
        self.update_target_window_list()

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

        # self.cbox_target_window.clear()
        # self.cbox_target_window.addItems(enum_windows())

        # Qt 위젯의 화면 기준 좌표
        global_pos = self.win_screen.mapToGlobal(self.win_screen.rect().topLeft())
        x = global_pos.x()
        y = global_pos.y()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            event.ignore()  # ESC 키 무시
        else:
            super().keyPressEvent(event)

    def slot_btn_save(self):
        try:
            self.solution.save_json()
            QMessageBox.information(
                self, "저장 완료", "Solution 정보가 저장되었습니다."
            )
        except Exception as e:
            QMessageBox.critical(self, "저장 실패", f"저장 중 오류 발생: {e}")

    def slot_ckbx_real_time_view(self):
        if self.ckbx_real_time_view.isChecked():
            # 실시간 미리보기 시작
            self._start_real_time_preview()
        else:
            # 실시간 미리보기 중지 및 마지막 한 장만 표시
            if hasattr(self, "_preview_timer"):
                self._preview_timer.stop()
            self._show_single_preview()

    def _get_window_client_offset(self, hwnd):
        # 윈도우의 클라이언트 영역 좌상단이 실제 스크린에서 어디에 있는지 반환
        import win32gui, win32con

        # 윈도우 전체 좌상단
        win_left, win_top, _, _ = win32gui.GetWindowRect(hwnd)
        # 클라이언트 좌상단 (윈도우 내부 기준)
        pt = win32gui.ClientToScreen(hwnd, (0, 0))
        client_left, client_top = pt
        # 오프셋: 클라이언트 좌상단 - 윈도우 좌상단
        offset_x = client_left - win_left
        offset_y = client_top - win_top
        return offset_x, offset_y

    def _start_real_time_preview(self):
        def update_preview():
            text = self.cbox_target_window.currentText()
            if not text.startswith("(miss) ") and text:
                hwnd = None
                import win32gui, win32process, psutil

                def enum_hwnds():
                    result = []

                    def callback(hwnd, _):
                        if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(
                            hwnd
                        ):
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
                    if item_text == text:
                        hwnd = hwnd_enum
                        break
                if hwnd:
                    win_left, win_top, win_right, win_bottom = win32gui.GetWindowRect(
                        hwnd
                    )
                    rel_x = self.spbx_screen_x.value()
                    rel_y = self.spbx_screen_y.value()
                    offset_x, offset_y = self._get_window_client_offset(hwnd)
                    self.real_x = win_left + offset_x + rel_x
                    self.real_y = win_top + offset_y + rel_y
                else:
                    self.real_x = self.spbx_screen_x.value()
                    self.real_y = self.spbx_screen_y.value()
            else:
                self.real_x = self.spbx_screen_x.value()
                self.real_y = self.spbx_screen_y.value()
            x = self.real_x
            y = self.real_y
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

        if not hasattr(self, "_preview_timer"):
            self._preview_timer = QTimer(self)
            self._preview_timer.timeout.connect(update_preview)
        self._preview_timer.start(1000 / self.fps_real_time_view)
        update_preview()

    # sync_spinbox_with_screen, sync_spinbox_to_solution, slot_target_window_changed에서도 동일하게 offset 적용
    def sync_spinbox_with_screen(self):
        geom = self.win_screen.geometry()
        self.spbx_screen_x.setValue(geom.x())
        self.spbx_screen_y.setValue(geom.y())
        self.spbx_screen_w.setValue(geom.width())
        self.spbx_screen_h.setValue(geom.height())
        text = self.cbox_target_window.currentText()
        if text.startswith("(miss) "):
            self.real_x = geom.x()
            self.real_y = geom.y()
        else:
            hwnd = None
            import win32gui, win32process, psutil

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
                if item_text == text:
                    hwnd = hwnd_enum
                    break
            if hwnd:
                win_left, win_top, win_right, win_bottom = win32gui.GetWindowRect(hwnd)
                rel_x = geom.x()
                rel_y = geom.y()
                offset_x, offset_y = self._get_window_client_offset(hwnd)
                self.real_x = win_left + offset_x + rel_x
                self.real_y = win_top + offset_y + rel_y
            else:
                self.real_x = geom.x()
                self.real_y = geom.y()

    def sync_spinbox_to_solution(self):
        self.solution.screen_x = self.spbx_screen_x.value()
        self.solution.screen_y = self.spbx_screen_y.value()
        self.solution.screen_w = self.spbx_screen_w.value()
        self.solution.screen_h = self.spbx_screen_h.value()
        text = self.cbox_target_window.currentText()
        if text.startswith("(miss) "):
            self.real_x = self.spbx_screen_x.value()
            self.real_y = self.spbx_screen_y.value()
        else:
            hwnd = None
            import win32gui, win32process, psutil

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
                if item_text == text:
                    hwnd = hwnd_enum
                    break
            if hwnd:
                win_left, win_top, win_right, win_bottom = win32gui.GetWindowRect(hwnd)
                rel_x = self.spbx_screen_x.value()
                rel_y = self.spbx_screen_y.value()
                offset_x, offset_y = self._get_window_client_offset(hwnd)
                self.real_x = win_left + offset_x + rel_x
                self.real_y = win_top + offset_y + rel_y
            else:
                self.real_x = self.spbx_screen_x.value()
                self.real_y = self.spbx_screen_y.value()

    def slot_target_window_changed(self, text):
        if text.startswith("(miss) "):
            self.solution.target_window = text[7:]
            self.real_x = self.spbx_screen_x.value()
            self.real_y = self.spbx_screen_y.value()
        else:
            self.solution.target_window = text
            hwnd = None
            import win32gui, win32process, psutil

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
                if item_text == text:
                    hwnd = hwnd_enum
                    break
            if hwnd:
                win_left, win_top, win_right, win_bottom = win32gui.GetWindowRect(hwnd)
                rel_x = self.spbx_screen_x.value()
                rel_y = self.spbx_screen_y.value()
                offset_x, offset_y = self._get_window_client_offset(hwnd)
                self.real_x = win_left + offset_x + rel_x
                self.real_y = win_top + offset_y + rel_y
            else:
                self.real_x = self.spbx_screen_x.value()
                self.real_y = self.spbx_screen_y.value()


if __name__ == "__main__":
    solution = Solution.load_json(
        os.path.join(os.getcwd(), "New_Solution", "New_Solution.json")
    )

    app = QApplication(sys.argv)
    apply_stylesheet(app, theme="dark_cyan.xml")
    window = WdgtBaseTab(None, solution)
    window.show()
    sys.exit(app.exec())
