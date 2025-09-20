import os
import sys

sys.path.append(os.path.abspath("."))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QDialog
from qt_material import apply_stylesheet

from ui.designer.ui_screen import Ui_screen


class Screen(QDialog, Ui_screen):
    RESIZE_MARGIN = 8  # 픽셀 단위, 조절 영역의 두께
    MIN_WIDTH = 50
    MIN_HEIGHT = 50

    def __init__(self, parent=None, h=0, w=0):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowFlags(Qt.FramelessWindowHint)
        # self.setAttribute(Qt.WA_TranslucentBackground, True)
        if h and w:
            self.resize(h, w)
        self._drag_active = False
        self._drag_position = None
        self._resize_active = False
        self._resize_direction = None
        self.setMouseTracking(True)
        for child in self.findChildren(QDialog):
            child.setMouseTracking(True)

    def _get_resize_direction(self, pos):
        margin = self.RESIZE_MARGIN
        rect = self.rect()
        left = pos.x() < margin
        right = pos.x() > rect.width() - margin
        top = pos.y() < margin
        bottom = pos.y() > rect.height() - margin
        # 대각선 우선
        if left and top:
            return "top_left"
        if right and top:
            return "top_right"
        if left and bottom:
            return "bottom_left"
        if right and bottom:
            return "bottom_right"
        if left:
            return "left"
        if right:
            return "right"
        if top:
            return "top"
        if bottom:
            return "bottom"
        return None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            direction = self._get_resize_direction(pos)
            if direction:
                self._resize_active = True
                self._resize_direction = direction
            else:
                self._drag_active = True
                self._drag_position = (
                    event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                )
            event.accept()

    def mouseMoveEvent(self, event):
        pos = event.pos()
        direction = self._get_resize_direction(pos)
        cursor_map = {
            "left": Qt.SizeHorCursor,
            "right": Qt.SizeHorCursor,
            "top": Qt.SizeVerCursor,
            "bottom": Qt.SizeVerCursor,
            "top_left": Qt.SizeFDiagCursor,
            "bottom_right": Qt.SizeFDiagCursor,
            "top_right": Qt.SizeBDiagCursor,
            "bottom_left": Qt.SizeBDiagCursor,
        }
        if self._resize_active:
            geo = self.geometry()
            min_w = self.MIN_WIDTH
            min_h = self.MIN_HEIGHT
            global_pos = event.globalPosition().toPoint()
            if self._resize_direction in cursor_map:
                self.setCursor(cursor_map[self._resize_direction])
            # --- 수정된 크기 조절 로직 ---
            if self._resize_direction == "left":
                diff = global_pos.x() - geo.x()
                new_x = geo.x() + diff
                new_w = geo.width() - diff
                if new_w < min_w:
                    new_x = geo.x() + (geo.width() - min_w)
                    new_w = min_w
                self.setGeometry(new_x, geo.y(), new_w, geo.height())
            elif self._resize_direction == "right":
                diff = global_pos.x() - (geo.x() + geo.width())
                new_w = geo.width() + diff
                if new_w < min_w:
                    new_w = min_w
                self.resize(new_w, geo.height())
            elif self._resize_direction == "top":
                diff = global_pos.y() - geo.y()
                new_y = geo.y() + diff
                new_h = geo.height() - diff
                if new_h < min_h:
                    new_y = geo.y() + (geo.height() - min_h)
                    new_h = min_h
                self.setGeometry(geo.x(), new_y, geo.width(), new_h)
            elif self._resize_direction == "bottom":
                diff = global_pos.y() - (geo.y() + geo.height())
                new_h = geo.height() + diff
                if new_h < min_h:
                    new_h = min_h
                self.resize(geo.width(), new_h)
            elif self._resize_direction == "top_left":
                diff_x = global_pos.x() - geo.x()
                diff_y = global_pos.y() - geo.y()
                new_x = geo.x() + diff_x
                new_y = geo.y() + diff_y
                new_w = geo.width() - diff_x
                new_h = geo.height() - diff_y
                if new_w < min_w:
                    new_x = geo.x() + (geo.width() - min_w)
                    new_w = min_w
                if new_h < min_h:
                    new_y = geo.y() + (geo.height() - min_h)
                    new_h = min_h
                self.setGeometry(new_x, new_y, new_w, new_h)
            elif self._resize_direction == "top_right":
                diff_x = global_pos.x() - (geo.x() + geo.width())
                diff_y = global_pos.y() - geo.y()
                new_y = geo.y() + diff_y
                new_w = geo.width() + diff_x
                new_h = geo.height() - diff_y
                if new_w < min_w:
                    new_w = min_w
                if new_h < min_h:
                    new_y = geo.y() + (geo.height() - min_h)
                    new_h = min_h
                self.setGeometry(geo.x(), new_y, new_w, new_h)
            elif self._resize_direction == "bottom_left":
                diff_x = global_pos.x() - geo.x()
                diff_y = global_pos.y() - (geo.y() + geo.height())
                new_x = geo.x() + diff_x
                new_w = geo.width() - diff_x
                new_h = geo.height() + diff_y
                if new_w < min_w:
                    new_x = geo.x() + (geo.width() - min_w)
                    new_w = min_w
                if new_h < min_h:
                    new_h = min_h
                self.setGeometry(new_x, geo.y(), new_w, new_h)
            elif self._resize_direction == "bottom_right":
                diff_x = global_pos.x() - (geo.x() + geo.width())
                diff_y = global_pos.y() - (geo.y() + geo.height())
                new_w = geo.width() + diff_x
                new_h = geo.height() + diff_y
                if new_w < min_w:
                    new_w = min_w
                if new_h < min_h:
                    new_h = min_h
                self.resize(new_w, new_h)
            event.accept()
        elif self._drag_active and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_position)
            event.accept()
        else:
            # 커서 변경 (항상 마우스 위치에 따라 미리 보여줌)
            if direction:
                self.setCursor(cursor_map[direction])
            else:
                self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):
        self._drag_active = False
        self._resize_active = False
        self._resize_direction = None
        self.setCursor(Qt.ArrowCursor)
        event.accept()

    def moveEvent(self, event):
        super().moveEvent(event)
        if hasattr(self, "geometryChanged") and callable(self.geometryChanged):
            self.geometryChanged()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "geometryChanged") and callable(self.geometryChanged):
            self.geometryChanged()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme="dark_cyan.xml")
    window = Screen()
    window.show()
    sys.exit(app.exec())
