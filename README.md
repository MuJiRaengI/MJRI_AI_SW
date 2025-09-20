# MJRI_AI_SW
무지랭이의 AI Software

`pip install -r requirements.txt`

아래 코드를 실행해야 모든 gym 게임을 실행할 수 있음
`AutoROM --accept-license`

만약 `gymnasium`에서 특정 게임을 찾을 수 없다고 한 경우, 
```python
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

```
위와 같이 ale_py를 등록해줘야 함

pytorch 설치
link : https://pytorch.org/get-started/locally/

`pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126`


TTS 설치 방법
```
git clone https://github.com/myshell-ai/MeloTTS.git
cd MeloTTS
pip install -e .
python -m unidic download

```
