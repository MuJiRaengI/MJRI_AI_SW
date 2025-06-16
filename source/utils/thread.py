from PySide6.QtCore import QThread, Signal


class EnvWorker(QThread):
    finished = Signal()
    progress = Signal(str)

    def __init__(self, env_class, solution_dir, mode):
        super().__init__()
        self.env_class = env_class
        self.solution_dir = solution_dir
        self.mode = mode

    def run(self):
        env = self.env_class()
        env.play(self.solution_dir, self.mode)
        self.finished.emit()
