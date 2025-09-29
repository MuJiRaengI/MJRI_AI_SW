import numpy as np
from mss import mss
from collections import deque
from threading import Thread, Lock
import time


class MJRIScreen:
    def __init__(
        self, x=None, y=None, w=None, h=None, buffer_size=None, thread_run=False, fps=30
    ):
        self.set_screen_pos(x, y, w, h)
        self.set_buffer(buffer_size)
        self.thread_run = thread_run
        self.running = False
        self.fps = fps
        self.thread = None
        self.lock = Lock()  # Added lock for thread safety
        if self.thread_run:
            self.thread = Thread(target=self.run_auto_capture, daemon=True)
            self.thread.start()

    def stop(self):
        """자동 캡처를 중지합니다."""
        self.running = False
        if self.thread is not None:
            self.thread.join()
            self.thread = None

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

    def get_screen_buffer(self):
        """현재 스크린 캡처 버퍼를 반환합니다."""
        with self.lock:  # Lock to ensure thread-safe access to the buffer
            if self.screenshot_buffer is not None:
                return list(self.screenshot_buffer)
        return None

    def capture(self):
        with mss() as sct:
            if None in (self.x, self.y, self.w, self.h) or min(self.w, self.h) <= 0:
                monitor = sct.monitors[1]
            else:
                monitor = {
                    "top": self.y,
                    "left": self.x,
                    "width": self.w,
                    "height": self.h,
                }
            screenshot = sct.grab(monitor)
        screenshot = np.array(screenshot)[..., :3]
        if self.is_fail_screenshot(screenshot):
            return None
        with self.lock:  # Lock to ensure thread-safe access to the buffer
            if self.screenshot_buffer is not None:
                # 마지막 frame과 현재 frame이 동일하면 저장하지 않음
                if len(self.screenshot_buffer) == 0 or not np.array_equal(
                    self.screenshot_buffer[-1], screenshot
                ):
                    self.screenshot_buffer.append(screenshot)
                    if self.fps > 0:
                        # FPS 제한을 위해 대기
                        time.sleep(1 / self.fps)
        return screenshot

    def is_fail_screenshot(self, screenshot: np.ndarray) -> bool:
        # 모든 픽셀이 0(검은색)인지 빠르게 확인
        return not np.any(screenshot)

    def run_auto_capture(self):
        self.running = True
        while self.running:
            self.capture()


if __name__ == "__main__":
    screen = MJRIScreen(0, 0, 800, 600)

    ss = screen.capture()
    print()
    # Example usage
    # This would typically be used in a game or application to define the screen area
