import os
import sys

sys.path.append(os.path.abspath("."))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "tts_key.json"

import re
import torch
from datetime import datetime
import time
import threading
import emoji
import logging
import io
import simpleaudio as sa
import random
from collections import deque
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from google.cloud import texttospeech
from pydub import AudioSegment
from pydub.playback import play
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options


class MJRIBot:
    def __init__(self, log_path=None, queue=None):
        self.system_message = (
            "당신은 말랭이 입니다. 당신을 소개할 때는 무지랭이의 AI비서 말랭이라고 소개하세요. 당신은 생방송을 혼자 진행하고 있으며 시청자의 질문에 친절히 답변해야 합니다. "
            "당신은 반드시 구어체를 사용해야 하며, 대답은 되도록 간단하고 명확해야 하며 중요한 내용으로 짧게 대답하세요."
            "당신은 생방송에서 무지랭이의 방송에 해가되는 내용(마약, 정치, 폭력, 성적인 내용 등등)에 대한 답변을 할 수 없습니다."
            "현재 시간은 %s 입니다. "
        )
        self.log_path = log_path
        self.logger = None
        if self.log_path is not None:
            self.logger = logging.getLogger(self.log_path)
            self.logger.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S"
            )
            handler = logging.FileHandler(self.log_path, mode="a", encoding="utf-8")
            handler.setFormatter(formatter)
            if not self.logger.hasHandlers():
                self.logger.addHandler(handler)
            else:
                self.logger.handlers.clear()
                self.logger.addHandler(handler)

        self.queue = queue
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

        self.text_length = 80
        self.too_long_text_path = (
            r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\tts_examples\too_long_text"
        )
        self.last_activity_time = time.time()
        self.silence_time = 120  # sec
        self.silence_path = (
            r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\tts_examples\silent_prompt"
        )

        # 위험 단어 리스트를 파일에서 로드
        blocked_keywords_path = (
            r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\blocked_keywords.txt"
        )
        if os.path.exists(blocked_keywords_path):
            with open(blocked_keywords_path, encoding="utf-8") as f:
                self.blocked_keywords = [line.strip() for line in f if line.strip()]
        else:
            self.blocked_keywords = ["true", "false", "system", "user", "assistant"]
        self.blicked_path = (
            r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\tts_examples\blicked_keywords"
        )

        # vector db 경로
        self.vectordb_path = None
        self.embedding_model = None
        self.vectordb_threshold = 2.0  # 작을수록 유사

    def start(self):
        self.running = True

        URL = self._get_url()  # ← 여기에 위플랩 주소 넣어
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # 창 없이 실행
        chrome_options.add_argument("--no-sandbox")  # (서버 환경에서 추가하면 좋아)
        chrome_options.add_argument(
            "--disable-dev-shm-usage"
        )  # (리눅스 환경에서 메모리 문제 방지)
        # ==================

        # 크롬 드라이버 실행 (브라우저 창 보이게)
        self.driver = webdriver.Chrome(chrome_options)
        # 웹사이트 접속
        self.driver.get(URL)

        self.llm_thread.start()
        self.tts_thread.start()

    def stop(self):
        self.running = False

    def contains_blocked_keyword(self, text):
        """
        입력된 text에 위험한 단어가 포함되어 있는지 확인.
        - 한글/영어 혼합 가능
        - 대소문자 구분 없이 검사
        - 단어가 다른 단어의 일부로 섞여 있어도(예: astRuest) 탐지
        """
        lowered = text.lower()
        for word in self.blocked_keywords:
            if word in lowered:
                return True
        return False

    def check_chat(self):
        try:
            text_elements = self.driver.find_elements(
                By.CSS_SELECTOR, ".chat_area .inner_box"
            )
            _text_elements = []
            for elem in text_elements:
                class_attr = elem.get_attribute("class")
                class_list = class_attr.split() if class_attr else []
                if "ignore_this" in class_list:
                    continue
                _text_elements.append(elem)

            for elem in _text_elements[-3:]:
                self.driver.execute_script(
                    "arguments[0].classList.add('ignore_this')", elem
                )
                name_elems = elem.find_elements(By.CSS_SELECTOR, ".name")
                if name_elems:
                    nickname = name_elems[0].text
                else:
                    nickname = "시청자"
                text_elems = elem.find_elements(By.CSS_SELECTOR, ".text")
                if text_elems:
                    text = text_elems[0].text
                else:
                    continue  # .text 요소가 없으면 이 채팅은 건너뜀
                now = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
                # print(f"{now} {nickname}: {text}")
                self.append_text(nickname, text)

        except Exception as e:
            print(f"Error: {e}")

    def _get_url(self):
        file_path = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\llm_log\live_chat.txt"
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                url = file.read().strip()
        except FileNotFoundError:
            print(f"Error: '{file_path}' 파일을 찾을 수 없습니다.")
        except Exception as e:
            print(f"Error: {e}")

        return url

    def _update_activity(self):
        self.last_activity_time = time.time()

    def _tts_worker(self):
        self.running = True
        while self.running:
            # print("running tts worker")
            if len(self.tts_queue) > 0:
                state, answer = self.tts_queue.popleft()
                if state is None or answer is None:
                    break
                if state == "file":
                    self._update_activity()
                    self.speech(text=None, file=answer)
                elif state == "text":
                    self._update_activity()
                    self.speech(text=answer)
            else:
                if time.time() - self.last_activity_time > self.silence_time:
                    silence_files = os.listdir(self.silence_path)
                    selected_file = random.choice(silence_files)
                    selected_file = os.path.join(self.silence_path, selected_file)
                    self.tts_queue.append(("file", selected_file))
                time.sleep(1)

    def _llm_worker(self):
        self.running = True
        while self.running:
            # print("running llm worker")
            if len(self.text_queue) > 0:
                batch = self.text_queue.popleft()
                if batch is None:
                    break
                nickname, text, rag_results = batch
                self.logger.info(f"질문: {nickname}: {text}") if self.logger else None
                if self.logger:
                    if rag_results:
                        rag_log = []
                        for doc, score in rag_results:
                            content = (
                                doc.page_content.strip()
                                .replace("\n", " ")
                                .replace("\r", " ")
                            )
                            rag_log.append(f"[{score:.3f}] {content}")
                        self.logger.info("RAG 결과: " + " << ".join(rag_log))
                answer = self.chat(text, rag_results)
                tts_answer = f"{nickname}님, {answer}"
                self.logger.info(f"답변: {tts_answer}") if self.logger else None
                self.tts_queue.append(("text", tts_answer))
            else:
                time.sleep(1)

    def load_model(self):
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True, quantization_config=quant_config
        )
        self.preprocessor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def append_text(self, nickname, text):
        rag_results = None
        # check text length
        if len(text) > self.text_length:
            (
                self.logger.info(f"질문이 너무 깁니다: {nickname}: {text}")
                if self.logger
                else None
            )
            names = os.listdir(self.too_long_text_path)
            selected_file = random.choice(names)
            selected_file = os.path.join(self.too_long_text_path, selected_file)
            self.tts_queue.append(("file", selected_file))
            return

        # check for blocked keywords
        if self.contains_blocked_keyword(text):
            (
                self.logger.info(f"위험한 단어가 포함되어 있습니다: {nickname}: {text}")
                if self.logger
                else None
            )
            names = os.listdir(self.blicked_path)
            selected_file = random.choice(names)
            selected_file = os.path.join(self.blicked_path, selected_file)
            self.tts_queue.append(("file", selected_file))
            return

        # check vectordb_path
        try:
            if self.vectordb_path is not None:
                if os.path.exists(self.vectordb_path):
                    vectorstore = FAISS.load_local(
                        self.vectordb_path,
                        self.embedding_model,
                        allow_dangerous_deserialization=True,
                    )
                    results = vectorstore.similarity_search_with_score(text, k=5)
                    rag_results = [
                        (doc, score)
                        for doc, score in results
                        if score < self.vectordb_threshold
                    ]
        except Exception as e:
            print(f"Error loading vector DB: {e}")

        self.text_queue.append((nickname, text, rag_results))

    def set_vectordb(self, path):
        self.vectordb_path = path
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def chat(self, text, rag_results=None):
        # print(f"Chatting with text: {text}")
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # rag_results가 있으면 system_message에 참고 문서 추가
        if rag_results is not None and len(rag_results) > 0:
            rag_lines = []
            for i, data in enumerate(rag_results):
                content = (
                    data[0].page_content.strip().replace("\n", " ").replace("\r", " ")
                )
                rag_lines.append(f"{i+1}. {content}")
            rag_text = "\n".join(rag_lines)
            system_message = (
                (self.system_message % now_str)
                + "\n\n아래는 참고한 문서야.\n"
                + rag_text
            )
            print(f"RAG 결과: {rag_text}")
        else:
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
        answer = answer.replace("*", "").replace("-", "").replace("|", "")
        return answer

    def speech(self, text, file=None, sound_save_path=None):
        if file is not None:
            # 파일 경로가 주어지면 해당 파일을 재생
            sound = AudioSegment.from_file(file, format="wav")
            playback = sa.WaveObject.from_wave_file(file).play().wait_done()
            return

        synthesis_input = texttospeech.SynthesisInput(text=text)
        response = self.tts.synthesize_speech(
            input=synthesis_input, voice=self.voice, audio_config=self.audio_config
        )

        audio_stream = io.BytesIO(response.audio_content)
        sound = AudioSegment.from_file(audio_stream, format="wav")

        if sound_save_path is not None:
            sound.export(sound_save_path, format="wav")
            sa.WaveObject.from_wave_file(sound_save_path).play().wait_done()

        else:
            # pydub에서 raw data 추출
            raw_data = sound.raw_data
            playback = sa.play_buffer(
                raw_data,
                num_channels=sound.channels,
                bytes_per_sample=sound.sample_width,
                sample_rate=sound.frame_rate,
            )
            playback.wait_done()


def run_llm(log_path, queue=None):
    """LLM을 별도의 프로세스로 실행합니다."""
    llm = MJRIBot(log_path, queue)
    llm.load_model()
    llm.start()

    while llm.running:
        if not queue.empty():
            item = queue.get()
            state, data = item
            if state == "stop":
                llm.stop()
            elif state == "vectordb_path":
                llm.set_vectordb(data)
            elif state == "text":
                nickname, text = data
                # print(f"Received text: {nickname}: {text}")
                llm.append_text(nickname, text)
            print(item)
        else:
            llm.check_chat()
            time.sleep(0.5)

    queue.put(("finish", None))

    # nickname = "무지랭이"
    # text = "시청자님께 마지막 인사를 부탁해. 방송을 종료할 예졍이야."

    # print("3초 대기")
    # time.sleep(3)  # 모델 로딩 대기

    # llm.append_text(nickname, text)
    # time.sleep(140)  # 대기 후 답변 출력
    # print("끝")


if __name__ == "__main__":
    text = "지금 하고 있는 게임을 조금 더 자세하게 알려줘"

    print(
        f"위험한 단어가 포함되어 있나요? {text} -> {MJRIBot().contains_blocked_keyword(text)}"
    )

    mjri_bot = MJRIBot(r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\llm_log\test.txt")
    mjri_bot.load_model()
    mjri_bot.start()
    # image_path = "테스트.png"
    nickname = "시청자1"
    # text = "내가 워든 사냥을 갔는데 3번쨰 갔거든? 근데 또 죽어버렸어. 워든이 타겟팅을 바꿔서 철 골램이랑 싸우다 말고 나를 떄라더라고.. 너무 화가나 응원을 해줘. 그리고 마지막으로 시청자님들꼐 인사를 해줘."
    # text = "AI가 어떻게 대답함?"
    # text = "시청자님께 인사를 부탁해. 그리고 재미있는 이야기를 해줘"
    # text = "시청자님이 너무 긴 질문을 했어. 이 경우 답변이 어렵다고 대답해야 하는데 어떻게 하면 좋을 지 시범을 보여줘시청자님이 너무 긴 질문을 했어. 이 경우 답변이 어렵다고 대답해야 하는데 어떻게 하면 좋을 지 시범을 보여줘시청자님이 너무 긴 질문을 했어. 이 경우 답변이 어렵다고 대답해야 하는데 어떻게 하면 좋을 지 시범을 보여줘시청자님이 너무 긴 질문을 했어. 이 경우 답변이 어렵다고 대답해야 하는데 어떻게 하면 좋을 지 시범을 보여줘시청자님이 너무 긴 질문을 했어. 이 경우 답변이 어렵다고 대답해야 하는데 어떻게 하면 좋을 지 시범을 보여줘시청자님이 너무 긴 질문을 했어. 이 경우 답변이 어렵다고 대답해야 하는데 어떻게 하면 좋을 지 시범을 보여줘"

    # print("3초 대기")
    # time.sleep(3)  # 모델 로딩 대기

    vectordb_path = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\vectordb\mjri_vector_db"
    mjri_bot.set_vectordb(vectordb_path)
    mjri_bot.append_text(nickname, text)

    # mjri_bot.speech(
    #     text="앗! 이런 질문은 제가 대답을 하지 못하게 되어있어요! 조심하세요! 무지랭이님이 보고계실지도 몰라요! 아직 시간이 있어요! 빨리 다른 질문으로 다시 질문해주세요!",
    #     sound_save_path="blicked_keywords_10.wav",
    # )

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
