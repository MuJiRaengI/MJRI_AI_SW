import os
import sys

sys.path.append(os.path.abspath("."))

import re
import torch
import datetime
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from source.tts.mmri_tts import MMRITTS


class MJRIBot:
    def __init__(self, log_path=None):
        self.system_message = (
            "당신은 네이버의 하이퍼스케일 모델인 하이퍼클로바 엑스 기술을 기반으로 만들어진 인공지능 언어모델입니다."
            "당신은 유튜버 무지랭이의 AI봇인 말랭이 입니다. 당신은 생방송을 혼자 진행하고 있으며 시청자의 질문에 친절히 답변해야 합니다. "
            "무지랭이는 주로 월요일 오후 8시에 생방송을 진행하며, 인공지능을 사용하여 게임을 플레이 하는 콘텐츠를 다루는 유튜버입니다. "
            "당신은 생방송에서 무지랭이의 방송에 해가되는 내용(마약, 정치, 폭력, 성적인 내용 등등)에 대한 답변을 할 수 없습니다. "
            "텍스트에는 []사이에 시청자의 이름이 함께 제공됩니다. "
            "답변은 반드시 한글로 답변하고, 짧고 간단하게 해야하며 특수문자는 사용하지 마세요. "
            "현재 시간은 %s 입니다. "
        )
        self.log_path = log_path
        self.model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"

        self.model = None
        self.tokenizer = None
        self.processor = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.tts = MMRITTS(language="KR", device=self.device)

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True, load_in_8bit=True
        )
        self.preprocessor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def chat(self, nickname, text, image_path=None):
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        system_message = self.system_message % now_str
        chat = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"[{nickname}]: {text}"},
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
        return answer

    def speech(self, answer):
        self.tts.speech(answer, output_path="answer.wav")


if __name__ == "__main__":
    mjri_bot = MJRIBot()
    mjri_bot.load_model()
    # image_path = "테스트.png"
    nickname = "가나다라"
    text = "너는 뭐로 만들어졌어?"
    answer = mjri_bot.chat(nickname, text)
    print(f"Answer: {answer}")
    mjri_bot.speech(answer)

    # text = "지금 뭐하고 있는거야?"
    # answer = mjri_bot.chat(nickname, text)
    # print(f"Answer: {answer}")
    # mjri_bot.speech(answer)

    import time

    time.sleep(120)  # 음성 재생 대기

    mjri_bot.tts.stop()
