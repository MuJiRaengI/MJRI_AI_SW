import numpy as np
from mss import mss
from collections import deque


class MJRIScreen:
    def __init__(self, x=None, y=None, w=None, h=None, buffer_size=None):
        self.set_screen_pos(x, y, w, h)
        self.set_buffer(buffer_size)

    def set_screen_pos(self, x: int, y: int, w: int, h: int):
        """스크린 위치와 크기를 설정합니다."""
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def set_buffer(self, buffer_size: int):
        """스크린 캡처 버퍼를 설정합니다."""
        if buffer_size is not None and buffer_size > 0:
            self.screenshot_buffer = deque(maxlen=buffer_size)
        else:
            self.screenshot_buffer = None

    def capture(self):
        with mss() as sct:
            monitor = {"top": self.y, "left": self.x, "width": self.w, "height": self.h}
            screenshot = sct.grab(monitor)
        screenshot = np.array(screenshot)[..., :3]
        if self.is_fail_screenshot(screenshot):
            return None
        if self.screenshot_buffer is not None:
            self.screenshot_buffer.append(screenshot)
        return screenshot

    def is_fail_screenshot(self, screenshot: np.ndarray) -> bool:
        # 모든 픽셀이 0(검은색)인지 빠르게 확인
        return not np.any(screenshot)


if __name__ == "__main__":
    screen = MJRIScreen(0, 0, 800, 600)

    ss = screen.capture()
    print()
    # Example usage
    # This would typically be used in a game or application to define the screen area
