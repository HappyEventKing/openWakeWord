# test_params.py
import requests
import edge_tts
import asyncio
import os
import dashscope
from dashscope.audio.tts import SpeechSynthesizer

KOKORO_URL = "http://192.168.12.110:8880/v1/audio/speech"
ALIYUN_API_KEY = "sk-e9f779100b7e4acea5cb23ae9b741995"

# ===== 测试 Kokoro 参数 =====
def test_kokoro_params():
    print("🎙️ 测试 Kokoro 参数...")
    for speed in [0.5, 0.75, 1.0, 1.25, 1.5]:
        r = requests.post(KOKORO_URL, json={
            "model": "kokoro",
            "input": "Hey Eventi",
            "voice": "af_heart",
            "speed": speed,
        }, timeout=30)
        print(f"  speed={speed}: {'✅' if r.status_code == 200 else '❌'} {r.status_code}")

# ===== 测试 阿里云 参数 =====
def test_aliyun_params():
    print("\n🎙️ 测试 阿里云 参数...")
    dashscope.api_key = ALIYUN_API_KEY

    for rate in [-500, -200, 0, 200, 500]:
        result = SpeechSynthesizer.call(
            model="sambert-eva-v1",
            text="Hey Eventi",
            sample_rate=16000,
            format="wav",
            speech_rate=rate,
        )
        ok = result.get_audio_data() is not None
        print(f"  speech_rate={rate}: {'✅' if ok else '❌'}")

    for pitch in [-500, -200, 0, 200, 500]:
        result = SpeechSynthesizer.call(
            model="sambert-eva-v1",
            text="Hey Eventi",
            sample_rate=16000,
            format="wav",
            pitch_rate=pitch,
        )
        ok = result.get_audio_data() is not None
        print(f"  pitch_rate={pitch}: {'✅' if ok else '❌'}")

# ===== 测试 Edge TTS 参数 =====
async def test_edge_params():
    print("\n🎙️ 测试 Edge TTS 参数...")

    for rate in ["-30%", "-15%", "+0%", "+15%", "+30%"]:
        try:
            communicate = edge_tts.Communicate("Hey Eventi", "en-US-GuyNeural", rate=rate)
            await communicate.save(f"temp_rate_{rate}.mp3")
            os.remove(f"temp_rate_{rate}.mp3")
            print(f"  rate={rate}: ✅")
        except Exception as e:
            print(f"  rate={rate}: ❌ {e}")

    for pitch in ["-20Hz", "-10Hz", "+0Hz", "+10Hz", "+20Hz"]:
        try:
            communicate = edge_tts.Communicate("Hey Eventi", "en-US-GuyNeural", pitch=pitch)
            await communicate.save(f"temp_pitch_{pitch}.mp3")
            os.remove(f"temp_pitch_{pitch}.mp3")
            print(f"  pitch={pitch}: ✅")
        except Exception as e:
            print(f"  pitch={pitch}: ❌ {e}")

async def main():
    test_kokoro_params()
    test_aliyun_params()
    await test_edge_params()

asyncio.run(main())