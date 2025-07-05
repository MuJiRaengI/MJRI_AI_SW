import keyboard
import time


class MJRIKeyboard:
    def __init__(self):
        pass

    def press_and_release(self, key, delay=1):
        keyboard.press(key)
        time.sleep(delay)
        keyboard.release(key)

    def press(self, key, delay):
        keyboard.press(key)
        time.sleep(delay)

    def release(self, key, delay):
        keyboard.release(key)
        time.sleep(delay)

    def release_keys(self, keys):
        for key in keys:
            keyboard.release(key)

    def set_numbering(self, number):
        self.press("ctrl", delay=0.5)
        self.press_and_release(str(number), delay=0.5)
        self.release("ctrl", delay=0.5)


if __name__ == "__main__":
    key = MJRIKeyboard()
    # ctrl 누른 상태로 숫자 3키를 눌른다
    key.press("ctrl", delay=0.5)
    key.press_and_release("3", delay=0.5)
    key.release("ctrl", delay=0.5)
