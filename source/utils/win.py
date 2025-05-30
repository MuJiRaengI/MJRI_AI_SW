import win32gui
from mss import mss
import numpy as np

def get_all_window_titles():
    titles = []

    def enum_windows_callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                titles.append(title)

    win32gui.EnumWindows(enum_windows_callback, None)
    return titles

def capture_window(x, y, w, h):
    with mss() as sct:
        monitor = {"top": y, "left": x, "width": w, "height": h}
        screenshot = sct.grab(monitor)
    screenshot = np.array(screenshot)[..., :3]
    return screenshot
