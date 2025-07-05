import time
import ctypes
import win32gui


MOUSEEVENTF_LEFTDOWN = 0x0002  # 왼쪽 버튼 누름
MOUSEEVENTF_LEFTUP = 0x0004  # 왼쪽 버튼 뗌
MOUSEEVENTF_RIGHTDOWN = 0x0008  # 오른쪽 버튼 누름
MOUSEEVENTF_RIGHTUP = 0x0010  # 오른쪽 버튼 뗌


class MJRIMouse:
    def __init__(self):
        pass

    def drag(self, x1, y1, x2, y2, delay=0.5):
        """(x1, y1)에서 (x2, y2)로 왼쪽 버튼 드래그"""
        ctypes.windll.user32.SetCursorPos(x1, y1)
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(delay)
        ctypes.windll.user32.SetCursorPos(x2, y2)
        time.sleep(delay)
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        time.sleep(delay)

    def move(self, x, y, delay=1):
        """마우스를 (x, y)로 이동만 시킴"""
        ctypes.windll.user32.SetCursorPos(x, y)
        time.sleep(delay)

    def leftClick(self, x, y, delay=1):
        """(x, y)에서 왼쪽 클릭"""
        ctypes.windll.user32.SetCursorPos(x, y)
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        time.sleep(delay)

    def rightClick(self, x, y, delay=1):
        ctypes.windll.user32.SetCursorPos(x, y)
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
        time.sleep(delay)


if __name__ == "__main__":
    mouse = MJRIMouse()
    # 예시: 화면 중앙에서 오른쪽 클릭
    # mouse.rightClick(500, 500, delay=0.5)
    # mouse.move(500, 500, delay=0.5)
    # mouse.leftClick(500, 500, delay=0.5)

    mouse.drag(400, 400, 800, 800, delay=0.1)
