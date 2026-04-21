import requests, os, subprocess

KOKORO_URL = "http://192.168.12.110:8880/v1/audio/speech"
FFMPEG_EXE = os.path.abspath(r"tools\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe")
os.makedirs("voice_test", exist_ok=True)

# 测试非英文音色
test_voices = [
    "ef_dora", "em_alex", "em_santa",
    "ff_siwis",
    "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
    "if_sara", "im_nicola",
    "pf_dora", "pm_alex", "pm_santa",
    "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
    "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
    "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro",
    "jm_kumo",
]

for voice in test_voices:
    temp = f"voice_test/kokoro_{voice}.mp3"
    out  = f"voice_test/kokoro_{voice}.wav"
    try:
        r = requests.post(
            KOKORO_URL,
            json={"model": "kokoro", "input": "Hey Eventi", "voice": voice, "speed": 1.0},
            timeout=30
        )
        if r.status_code == 200:
            with open(temp, "wb") as f:
                f.write(r.content)
            cmd = [FFMPEG_EXE, "-y", "-i", temp,
                   "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", out]
            subprocess.run(cmd, capture_output=True)
            os.remove(temp)
            print(f"✅ {voice}")
        else:
            print(f"❌ {voice}: {r.status_code}")
    except Exception as e:
        print(f"❌ {voice}: {e}")