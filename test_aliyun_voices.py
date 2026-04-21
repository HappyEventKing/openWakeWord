import dashscope
import os
from dashscope.audio.tts import SpeechSynthesizer

dashscope.api_key = "sk-e9f779100b7e4acea5cb23ae9b741995"  # 填入你的 key

FFMPEG_EXE = os.path.abspath(r"tools\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe")
os.makedirs("voice_test", exist_ok=True)

# 测试候选模型
models = [
    "sambert-eva-v1",
    "sambert-beth-v1",
    "sambert-cindy-v1",
    "sambert-perla-v1",
    "sambert-clara-v1",
    "sambert-camila-v1",
    "sambert-cally-v1",
    "sambert-hanna-v1",
    "sambert-donna-v1",
    "sambert-betty-v1",
    "sambert-brian-v1",
    "sambert-indah-v1",
    "sambert-zhichu-v1",
    "sambert-zhide-v1",
    "sambert-zhida-v1",
    "sambert-zhishu-v1",
    "sambert-zhiyue-v1",
    "sambert-zhiye-v1",
    "sambert-zhiya-v1",
    "sambert-zhiying-v1",
    "sambert-zhistella-v1",
    "sambert-zhihao-v1",
    "sambert-zhilun-v1",
    "sambert-zhimao-v1",
    "sambert-zhigui-v1",
    "sambert-zhinan-v1",
    "sambert-zhixiao-v1",
    "sambert-zhimo-v1",
    "sambert-zhiming-v1",
    "sambert-zhiru-v1",
    "sambert-zhiqi-v1",
    "sambert-zhiting-v1",
    "sambert-zhiyuan-v1",
    "sambert-zhixiang-v1",
    "sambert-zhijia-v1",
    "sambert-zhifei-v1",
    "sambert-zhiwei-v1",
    "sambert-zhijing-v1",
    "sambert-zhimiao-emo-v1",
    "sambert-zhishuo-v1",
    "sambert-zhiqian-v1",
    "sambert-zhina-v1",
]

for model in models:
    out = f"voice_test/{model}.wav"
    try:
        result = SpeechSynthesizer.call(
            model=model,
            text="Hey Eventi",
            sample_rate=16000,
            format="wav"
        )
        if result.get_audio_data():
            with open(out, "wb") as f:
                f.write(result.get_audio_data())
            print(f"✅ {model}")
        else:
            print(f"❌ {model} 无数据")
    except Exception as e:
        print(f"❌ {model}: {e}")

print("\n完成！请听 voice_test/ 目录下的文件")
print("把发音好的模型名字告诉我")