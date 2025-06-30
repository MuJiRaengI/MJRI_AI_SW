import os
import sys

sys.path.append(os.path.abspath("."))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "tts_key.json"

import re
import torch
import datetime
import time
import threading
import emoji
import io
import simpleaudio as sa
from collections import deque
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from google.cloud import texttospeech
from pydub import AudioSegment
from pydub.playback import play


class MJRIBot:
    def __init__(self, log_path=None):
        self.system_message = (
            "당신은 유튜버 무지랭이의 AI봇인 말랭이 입니다. 당신은 생방송을 혼자 진행하고 있으며 시청자의 질문에 친절히 답변해야 합니다. "
            "무지랭이는 주로 월요일 오후 8시에 생방송을 진행하며, 인공지능을 사용하여 게임을 플레이 하는 콘텐츠를 다루는 유튜버입니다. "
            "당신은 생방송에서 무지랭이의 방송에 해가되는 내용(마약, 정치, 폭력, 성적인 내용 등등)에 대한 답변을 할 수 없습니다. "
            "단순 감탄사나 의미없는 질문에 대해서는 구독과 좋아요를 눌러달라는 답변을 해도 좋습니다. "
            "당신은 반드시 구어체를 사용해야 하며, 대답은 되도록 간단하고 명확해야 하며 중요한 내용으로 짧게 대답하세요. "
            "현재 시간은 %s 입니다. "
        )
        self.log_path = log_path
        self.model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
        self.voice_file_name = os.path.abspath(
            r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\answer.wav"
        )

        self.model = None
        self.tokenizer = None
        self.processor = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tts = texttospeech.TextToSpeechClient()
        self.voice = texttospeech.VoiceSelectionParams(
            language_code="ko-KR",  # 한국어
            name="ko-KR-Chirp3-HD-Leda",  # ← 고급 음성 지정
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
        )
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        )

        self.text_queue = deque(maxlen=1)
        self.llm_thread = threading.Thread(target=self._llm_worker, daemon=True)
        self.tts_queue = deque(maxlen=1)
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)

    def start(self):
        self.running = True
        self.llm_thread.start()
        self.tts_thread.start()

    def stop(self):
        self.running = False

    def _tts_worker(self):
        self.running = True
        while self.running:
            print("running tts worker")
            if len(self.tts_queue) > 0:
                answer = self.tts_queue.popleft()
                if answer is None:
                    break
                self.speech(answer)
            else:
                time.sleep(1)

    def _llm_worker(self):
        self.running = True
        while self.running:
            print("running llm worker")
            if len(self.text_queue) > 0:
                batch = self.text_queue.popleft()
                if batch is None:
                    break
                nickname, text = batch
                answer = self.chat(text)
                tts_answer = f"{nickname}님, {answer}"
                self.tts_queue.append(tts_answer)
            else:
                time.sleep(1)

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True, load_in_8bit=True
        )
        self.preprocessor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def append_text(self, nickname, text):
        print(f"Appending text into queue: {nickname}: {text}")
        self.text_queue.append((nickname, text))

    def chat(self, text):
        print(f"Chatting with text: {text}")
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        system_message = self.system_message % now_str
        chat = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"{text}"},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            chat, return_tensors="pt", tokenize=True
        )
        input_ids = input_ids.to(device="cuda")

        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=2048,
            do_sample=True,
            top_p=0.6,
            temperature=0.5,
            repetition_penalty=1.0,
        )

        decoded = self.tokenizer.batch_decode(output_ids)[0]
        # assistant 답변만 추출
        match = re.findall(
            r"<\|im_start\|>assistant(.*?)<\|im_end\|><\|endofturn\|>",
            decoded,
            re.DOTALL,
        )
        answer = match[-1].strip() if match else decoded.strip()
        answer = emoji.replace_emoji(answer, replace="")
        return answer

    def speech(self, text):
        synthesis_input = texttospeech.SynthesisInput(text=text)
        response = self.tts.synthesize_speech(
            input=synthesis_input, voice=self.voice, audio_config=self.audio_config
        )

        audio_stream = io.BytesIO(response.audio_content)
        sound = AudioSegment.from_file(audio_stream, format="wav")

        # pydub에서 raw data 추출
        raw_data = sound.raw_data
        playback = sa.play_buffer(
            raw_data,
            num_channels=sound.channels,
            bytes_per_sample=sound.sample_width,
            sample_rate=sound.frame_rate,
        )
        playback.wait_done()


if __name__ == "__main__":
    mjri_bot = MJRIBot()
    mjri_bot.load_model()
    mjri_bot.start()
    # image_path = "테스트.png"
    nickname = "무지랭이"
    text = "내가 워든 사냥을 갔는데 3번쨰 갔거든? 근데 또 죽어버렸어. 워든이 타겟팅을 바꿔서 철 골램이랑 싸우다 말고 나를 떄라더라고.. 너무 화가나 응원을 해줘. 그리고 마지막으로 시청자님들꼐 인사를 해줘."
    # text = "AI가 어떻게 대답함?"

    print("3초 대기")
    time.sleep(3)  # 모델 로딩 대기

    mjri_bot.append_text(nickname, text)

    # print(f"{nickname}님이 질문: {text}")
    time.sleep(5)  # 질문 대기
    # for i in range(10):
    #     mjri_bot.append_text(f"무지랭이{i + 1}", f"테스트 질문 {i + 1}번쨰  질문이야")
    # mjri_bot.stop()
    time.sleep(140)  # 대기 후 답변 출력
    print("끝")

    # # answer = mjri_bot.chat(nickname, text)
    # # text = f"{nickname}님, {answer}"
    # # print(text)
    # # mjri_bot.speech(text)

    # path = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\answer.mp3"

    # # sound = AudioSegment.from_file(path, format="mp3")
    # # play(sound)
    # from playsound import playsound

    # playsound(path)
    # import pygame

    # pygame.mixer.init()
    # pygame.mixer.music.load(path)
    # pygame.mixer.music.play()
    # while pygame.mixer.music.get_busy():
    #     time.sleep(0.1)
