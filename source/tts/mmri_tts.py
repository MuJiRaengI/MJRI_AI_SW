from melo.api import TTS
import sounddevice as sd
import soundfile as sf
import threading
import queue


class MMRITTS:
    def __init__(self, language="KR", device="cuda:0"):
        self.language = language
        self.device = device
        self.model = TTS(language=self.language, device=self.device)
        self.speaker_ids = self.model.hps.data.spk2id
        self._play_queue = queue.Queue(maxsize=3)
        self._player_thread = threading.Thread(target=self._play_worker, daemon=True)
        self._running = True
        self._player_thread.start()

    def _play_worker(self):
        while self._running:
            try:
                item = self._play_queue.get(timeout=1)
            except queue.Empty:
                # 큐가 비어있으면 1초 대기
                continue
            output_path = item[0]
            data, fs = sf.read(output_path, dtype="float32")
            sd.play(data, fs)
            sd.wait()
            self._play_queue.task_done()

    def speech(self, text, output_path=None, speed=1.1):
        if output_path is None:
            output_path = "output.wav"
        self.model.tts_to_file(
            text, self.speaker_ids[self.language], output_path, speed=speed
        )
        self._play_queue.put((output_path,))
        return output_path

    def stop(self):
        self._running = False

    def finish(self):
        self.stop()
