"""
统一音色测试工具。
测试各平台音色是否正常可用，并保存样本供试听参考。
"""
import argparse
import asyncio
import os
import sys
import requests
import dashscope
from dashscope.audio.tts import SpeechSynthesizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from custom.tts.voices import get_voices
from dotenv import load_dotenv

load_dotenv()

ALIYUN_API_KEY = os.getenv("ALIYUN_API_KEY", "")
KOKORO_URL = os.getenv("KOKORO_URL", "http://localhost:8880/v1/audio/speech")


def test_aliyun(models, text, output_dir):
    """测试阿里云音色。"""
    if not ALIYUN_API_KEY:
        print("跳过阿里云：ALIYUN_API_KEY 未设置")
        return
    dashscope.api_key = ALIYUN_API_KEY
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n测试阿里云 ({len(models)} 个模型)...")
    for model in models:
        out = os.path.join(output_dir, f"aliyun_{model}.wav")
        try:
            result = SpeechSynthesizer.call(
                model=model, text=text, sample_rate=16000, format="wav"
            )
            if result.get_audio_data():
                with open(out, "wb") as f:
                    f.write(result.get_audio_data())
                print(f"  ✅ {model}")
            else:
                print(f"  ❌ {model}: 无数据")
        except Exception as e:
            print(f"  ❌ {model}: {e}")


def test_kokoro(voices, text, output_dir):
    """测试 Kokoro 音色。"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n测试 Kokoro ({len(voices)} 个音色)...")
    for voice in voices:
        temp = os.path.join(output_dir, f"kokoro_{voice}.mp3")
        out = os.path.join(output_dir, f"kokoro_{voice}.wav")
        try:
            r = requests.post(
                KOKORO_URL,
                json={"model": "kokoro", "input": text, "voice": voice, "speed": 1.0},
                timeout=30
            )
            if r.status_code == 200:
                with open(temp, "wb") as f:
                    f.write(r.content)
                # 简单提示用户手动用 ffmpeg 转换（或后续统一处理）
                print(f"  ✅ {voice} (mp3 saved)")
            else:
                print(f"  ❌ {voice}: {r.status_code}")
        except Exception as e:
            print(f"  ❌ {voice}: {e}")


async def test_edge(voices, text, output_dir):
    """测试 Edge TTS 音色。"""
    import edge_tts
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n测试 Edge TTS ({len(voices)} 个音色)...")
    for voice in voices:
        out = os.path.join(output_dir, f"edge_{voice}.mp3")
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(out)
            print(f"  ✅ {voice}")
        except Exception as e:
            print(f"  ❌ {voice}: {e}")


def main():
    parser = argparse.ArgumentParser(description="测试 TTS 平台音色")
    parser.add_argument("--config", default="hey_eventi.yml", help="训练配置文件")
    parser.add_argument("--provider", default="all", help="平台: aliyun, kokoro, edge, all")
    parser.add_argument("--wake-word", default="Hey Eventi", help="测试文本")
    parser.add_argument("--output-dir", default="voice_test", help="输出目录")
    args = parser.parse_args()

    import yaml
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}

    providers = [args.provider] if args.provider != "all" else config.get("tts_providers", ["aliyun", "kokoro", "edge"])

    if "aliyun" in providers:
        models = get_voices("aliyun", config)
        if models:
            test_aliyun(models, args.wake_word, args.output_dir)
    if "kokoro" in providers:
        voices = get_voices("kokoro", config)
        if voices:
            test_kokoro(voices, args.wake_word, args.output_dir)
    if "edge" in providers:
        voices = get_voices("edge", config)
        if voices:
            asyncio.run(test_edge(voices, args.wake_word, args.output_dir))

    print(f"\n完成！样本保存在 {args.output_dir}/")


if __name__ == "__main__":
    main()
