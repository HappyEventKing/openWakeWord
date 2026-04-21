import asyncio
import edge_tts
import os
import requests
import uuid
import dashscope
import subprocess
import random
from itertools import product
from tqdm import tqdm
from dashscope.audio.tts import SpeechSynthesizer
from dotenv import load_dotenv

load_dotenv()

# ===== 配置 =====
POSITIVE_TRAIN_DIR = r"hey_eventi_model\hey_eventi\positive_train"
POSITIVE_TEST_DIR  = r"hey_eventi_model\hey_eventi\positive_test"
NEGATIVE_TRAIN_DIR = r"hey_eventi_model\hey_eventi\negative_train"
NEGATIVE_TEST_DIR  = r"hey_eventi_model\hey_eventi\negative_test"
TEMP_DIR           = r"temp_audio"
ALIYUN_API_KEY     = os.getenv("ALIYUN_API_KEY", "")
KOKORO_URL         = os.getenv("KOKORO_URL", "http://localhost:8880/v1/audio/speech")
FFMPEG_EXE         = os.getenv("FFMPEG_PATH", os.path.abspath(r"tools\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"))

REPEATS            = 5
TRAIN_RATIO        = 0.8

ALIYUN_CONCURRENCY = 5
KOKORO_CONCURRENCY = 1
EDGE_CONCURRENCY   = 20

os.makedirs(POSITIVE_TRAIN_DIR, exist_ok=True)
os.makedirs(POSITIVE_TEST_DIR,  exist_ok=True)
os.makedirs(NEGATIVE_TRAIN_DIR, exist_ok=True)
os.makedirs(NEGATIVE_TEST_DIR,  exist_ok=True)
os.makedirs(TEMP_DIR,           exist_ok=True)

# ===== 音色列表 =====
ALIYUN_MODELS = [
    "sambert-eva-v1", "sambert-beth-v1", "sambert-cindy-v1",
    "sambert-perla-v1", "sambert-clara-v1", "sambert-camila-v1",
    "sambert-cally-v1", "sambert-hanna-v1", "sambert-donna-v1",
    "sambert-betty-v1", "sambert-brian-v1", "sambert-indah-v1",
    "sambert-zhichu-v1", "sambert-zhide-v1", "sambert-zhida-v1",
    "sambert-zhishu-v1", "sambert-zhiyue-v1", "sambert-zhiye-v1",
    "sambert-zhiya-v1", "sambert-zhiying-v1", "sambert-zhistella-v1",
    "sambert-zhihao-v1", "sambert-zhilun-v1", "sambert-zhimao-v1",
    "sambert-zhigui-v1", "sambert-zhinan-v1", "sambert-zhixiao-v1",
    "sambert-zhimo-v1", "sambert-zhiming-v1", "sambert-zhiru-v1",
    "sambert-zhiqi-v1", "sambert-zhiting-v1", "sambert-zhiyuan-v1",
    "sambert-zhixiang-v1", "sambert-zhijia-v1", "sambert-zhifei-v1",
    "sambert-zhiwei-v1", "sambert-zhijing-v1", "sambert-zhimiao-emo-v1",
    "sambert-zhishuo-v1", "sambert-zhiqian-v1", "sambert-zhina-v1",
]

KOKORO_VOICES = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jadzia",
    "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river",
    "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric",
    "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck",
    "am_santa", "bf_alice", "bf_emma", "bf_lily", "bm_daniel",
    "bm_fable", "bm_george", "bm_lewis",
]

EDGE_VOICES = [
    "en-AU-WilliamMultilingualNeural", "en-AU-NatashaNeural",
    "en-CA-ClaraNeural", "en-CA-LiamNeural",
    "en-HK-YanNeural", "en-HK-SamNeural",
    "en-IN-NeerjaExpressiveNeural", "en-IN-NeerjaNeural", "en-IN-PrabhatNeural",
    "en-IE-ConnorNeural", "en-IE-EmilyNeural",
    "en-KE-AsiliaNeural", "en-KE-ChilembaNeural",
    "en-NZ-MitchellNeural", "en-NZ-MollyNeural",
    "en-NG-AbeoNeural", "en-NG-EzinneNeural",
    "en-PH-JamesNeural", "en-PH-RosaNeural",
    "en-US-AvaNeural", "en-US-AndrewNeural", "en-US-EmmaNeural", "en-US-BrianNeural",
    "en-SG-LunaNeural", "en-SG-WayneNeural",
    "en-ZA-LeahNeural", "en-ZA-LukeNeural",
    "en-TZ-ElimuNeural", "en-TZ-ImaniNeural",
    "en-GB-LibbyNeural", "en-GB-MaisieNeural", "en-GB-RyanNeural",
    "en-GB-SoniaNeural", "en-GB-ThomasNeural",
    "en-US-AnaNeural", "en-US-AndrewMultilingualNeural", "en-US-AriaNeural",
    "en-US-AvaMultilingualNeural", "en-US-BrianMultilingualNeural",
    "en-US-ChristopherNeural", "en-US-EmmaMultilingualNeural",
    "en-US-EricNeural", "en-US-GuyNeural", "en-US-JennyNeural",
    "en-US-MichelleNeural", "en-US-RogerNeural", "en-US-SteffanNeural",
    "zh-HK-HiuGaaiNeural", "zh-HK-HiuMaanNeural", "zh-HK-WanLungNeural",
    "zh-CN-XiaoxiaoNeural", "zh-CN-XiaoyiNeural", "zh-CN-YunjianNeural",
    "zh-CN-YunxiNeural", "zh-CN-YunxiaNeural", "zh-CN-YunyangNeural",
    "zh-CN-liaoning-XiaobeiNeural", "zh-TW-HsiaoChenNeural",
    "zh-TW-YunJheNeural", "zh-TW-HsiaoYuNeural", "zh-CN-shaanxi-XiaoniNeural",
]

# ===== 正样本文本 =====
POSITIVE_TEXT = "Hey Eventi"

# ===== 负样本短语 =====
NEGATIVE_PHRASES = [
    "Hey Event", "Hey Evenly", "Hey Avanti", "Hey Aventure",
    "Hey Eventing", "Hey Invention", "Hey Prevention", "A Venti",
    "Hey Advent", "Hey Eventer", "Hey Avenger", "Hey Eventual",
    "Hey Eventful", "Hey Evermore",
    "Hey Siri", "Hey Google", "Hey Alexa", "OK Google",
    "Hey Cortana", "Hey Bixby",
    "Hey everyone", "Hey everybody", "Hey there", "Hey you",
    "Hey listen", "Hey wait",
    "Hello world", "Good morning", "What time is it",
    "Turn on the lights", "Play some music", "Set a timer",
]

# ===== 参数组合 =====
ALIYUN_SPEECH_RATES = [-200, -100, 0, 100, 200]
ALIYUN_PITCH_RATES  = [-200, 0, 200]
KOKORO_SPEEDS       = [0.75, 0.9, 1.0, 1.1, 1.25]
EDGE_RATES          = ["-20%", "-10%", "+0%", "+10%", "+20%"]
EDGE_PITCHES        = ["-10Hz", "+0Hz", "+10Hz"]

pbar = None

# ===== 工具函数 =====
def convert_to_wav(input_path, output_path):
    cmd = [
        FFMPEG_EXE, "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-sample_fmt", "s16",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode == 0:
        if os.path.exists(input_path):
            os.remove(input_path)
        return True
    return False

def get_output_path(task):
    idx    = task["idx"]
    source = task["source"]
    kind   = task["kind"]
    split  = task["split"]

    if kind == "positive":
        base = POSITIVE_TRAIN_DIR if split == "train" else POSITIVE_TEST_DIR
        return os.path.join(base, f"{source}_{idx:06d}.wav")
    else:
        base = NEGATIVE_TRAIN_DIR if split == "train" else NEGATIVE_TEST_DIR
        phrase_tag = task["phrase_tag"]
        return os.path.join(base, f"{source}_{phrase_tag}_{idx:06d}.wav")

# ===== 构建正样本任务 =====
def build_positive_tasks():
    aliyun_tasks = []
    kokoro_tasks = []
    edge_tasks   = []

    # 阿里云: 42 × 15 × 5 = 3,150
    for repeat in range(REPEATS):
        for model in ALIYUN_MODELS:
            for speech_rate, pitch_rate in product(ALIYUN_SPEECH_RATES, ALIYUN_PITCH_RATES):
                aliyun_tasks.append({
                    "kind":        "positive",
                    "source":      "aliyun",
                    "text":        POSITIVE_TEXT,
                    "model":       model,
                    "speech_rate": speech_rate,
                    "pitch_rate":  pitch_rate,
                    "repeat":      repeat,
                })

    # Kokoro: 28 × 5 × 5 = 700
    for repeat in range(REPEATS):
        for voice in KOKORO_VOICES:
            for speed in KOKORO_SPEEDS:
                kokoro_tasks.append({
                    "kind":   "positive",
                    "source": "kokoro",
                    "text":   POSITIVE_TEXT,
                    "voice":  voice,
                    "speed":  speed,
                    "repeat": repeat,
                })

    # Edge: 61 × 15 × 5 = 4,575
    for repeat in range(REPEATS):
        for voice in EDGE_VOICES:
            for rate, pitch in product(EDGE_RATES, EDGE_PITCHES):
                edge_tasks.append({
                    "kind":   "positive",
                    "source": "edge",
                    "text":   POSITIVE_TEXT,
                    "voice":  voice,
                    "rate":   rate,
                    "pitch":  pitch,
                    "repeat": repeat,
                })

    all_tasks = aliyun_tasks + kokoro_tasks + edge_tasks
    random.shuffle(all_tasks)

    n_train = int(len(all_tasks) * TRAIN_RATIO)
    for idx, t in enumerate(all_tasks):
        t["idx"]   = idx
        t["split"] = "train" if idx < n_train else "test"

    return all_tasks, len(aliyun_tasks), len(kokoro_tasks), len(edge_tasks), n_train

# ===== 构建负样本任务 =====
def build_negative_tasks():
    aliyun_tasks = []
    kokoro_tasks = []
    edge_tasks   = []

    # 阿里云: 32短语 × 42音色 × 1(默认参数) = 1,344
    for phrase in NEGATIVE_PHRASES:
        phrase_tag = phrase.replace(" ", "_").replace("'", "")[:20]
        for model in ALIYUN_MODELS:
            aliyun_tasks.append({
                "kind":        "negative",
                "source":      "aliyun",
                "text":        phrase,
                "phrase_tag":  phrase_tag,
                "model":       model,
                "speech_rate": 0,
                "pitch_rate":  0,
                "repeat":      0,
            })

    # Kokoro: 32短语 × 28音色 × 1(默认速度) = 896
    for phrase in NEGATIVE_PHRASES:
        phrase_tag = phrase.replace(" ", "_").replace("'", "")[:20]
        for voice in KOKORO_VOICES:
            kokoro_tasks.append({
                "kind":       "negative",
                "source":     "kokoro",
                "text":       phrase,
                "phrase_tag": phrase_tag,
                "voice":      voice,
                "speed":      1.0,
                "repeat":     0,
            })

    # Edge: 32短语 × 61音色 × 1(默认参数) = 1,952
    for phrase in NEGATIVE_PHRASES:
        phrase_tag = phrase.replace(" ", "_").replace("'", "")[:20]
        for voice in EDGE_VOICES:
            edge_tasks.append({
                "kind":       "negative",
                "source":     "edge",
                "text":       phrase,
                "phrase_tag": phrase_tag,
                "voice":      voice,
                "rate":       "+0%",
                "pitch":      "+0Hz",
                "repeat":     0,
            })

    all_tasks = aliyun_tasks + kokoro_tasks + edge_tasks
    random.shuffle(all_tasks)

    n_train = int(len(all_tasks) * TRAIN_RATIO)
    for idx, t in enumerate(all_tasks):
        t["idx"]   = idx
        t["split"] = "train" if idx < n_train else "test"

    return all_tasks, len(aliyun_tasks), len(kokoro_tasks), len(edge_tasks), n_train

# ===== 生成函数 =====
async def gen_aliyun(task, sem, stats):
    async with sem:
        out_path = get_output_path(task)
        if os.path.exists(out_path):
            stats["skip"] += 1
            pbar.update(1)
            return
        temp = os.path.join(TEMP_DIR, f"aliyun_{uuid.uuid4().hex}.wav")
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: SpeechSynthesizer.call(
                model=task["model"],
                text=task["text"],
                sample_rate=16000,
                format="wav",
                speech_rate=task["speech_rate"],
                pitch_rate=task["pitch_rate"],
            ))
            if result.get_audio_data():
                with open(temp, "wb") as f:
                    f.write(result.get_audio_data())
                ok = await loop.run_in_executor(None, convert_to_wav, temp, out_path)
                stats["success" if ok else "fail"] += 1
            else:
                stats["fail"] += 1
        except Exception as e:
            stats["fail"] += 1
            tqdm.write(f"❌ 阿里云 [{task['kind']}] {task['model']} \"{task['text']}\": {e}")
        pbar.update(1)

async def gen_kokoro(task, sem, stats):
    async with sem:
        out_path = get_output_path(task)
        if os.path.exists(out_path):
            stats["skip"] += 1
            pbar.update(1)
            return
        temp = os.path.join(TEMP_DIR, f"kokoro_{uuid.uuid4().hex}.mp3")
        try:
            loop = asyncio.get_event_loop()
            r = await loop.run_in_executor(None, lambda: requests.post(
                KOKORO_URL,
                json={
                    "model": "kokoro",
                    "input": task["text"],
                    "voice": task["voice"],
                    "speed": task["speed"],
                },
                timeout=30
            ))
            if r.status_code == 200:
                with open(temp, "wb") as f:
                    f.write(r.content)
                ok = await loop.run_in_executor(None, convert_to_wav, temp, out_path)
                stats["success" if ok else "fail"] += 1
            else:
                stats["fail"] += 1
                tqdm.write(f"❌ Kokoro [{task['kind']}] {task['voice']} \"{task['text']}\": {r.status_code}")
        except Exception as e:
            stats["fail"] += 1
            tqdm.write(f"❌ Kokoro [{task['kind']}] {task['voice']} \"{task['text']}\": {e}")
        pbar.update(1)

async def gen_edge(task, sem, stats):
    async with sem:
        out_path = get_output_path(task)
        if os.path.exists(out_path):
            stats["skip"] += 1
            pbar.update(1)
            return
        temp = os.path.join(TEMP_DIR, f"edge_{uuid.uuid4().hex}.mp3")
        try:
            communicate = edge_tts.Communicate(
                task["text"],
                task["voice"],
                rate=task["rate"],
                pitch=task["pitch"],
            )
            await communicate.save(temp)
            loop = asyncio.get_event_loop()
            ok = await loop.run_in_executor(None, convert_to_wav, temp, out_path)
            stats["success" if ok else "fail"] += 1
        except Exception as e:
            stats["fail"] += 1
            tqdm.write(f"❌ Edge [{task['kind']}] {task['voice']} \"{task['text']}\": {e}")
        pbar.update(1)

# ===== 兼容 openwakeword/train.py 的 stub 函数 =====
def generate_samples(text, max_samples, batch_size, noise_scales, noise_scale_ws,
                     length_scales, output_dir, auto_reduce_batch_size=True, file_names=None):
    """
    Stub function for compatibility with openwakeword/train.py.
    Actual data generation is done by running this script directly.
    """
    print(f"[Stub] generate_samples called for {output_dir}, skipping (data already generated)")
    return


# ===== 主程序 =====
async def main():
    global pbar
    dashscope.api_key = ALIYUN_API_KEY

    random.seed(42)

    pos_tasks, pos_aliyun, pos_kokoro, pos_edge, pos_n_train = build_positive_tasks()
    neg_tasks, neg_aliyun, neg_kokoro, neg_edge, neg_n_train = build_negative_tasks()

    all_tasks = pos_tasks + neg_tasks

    print("=" * 55)
    print("  生成 Hey Eventi 训练样本（正样本 + 负样本）")
    print(f"  并发: 阿里云={ALIYUN_CONCURRENCY}  Kokoro={KOKORO_CONCURRENCY}  Edge={EDGE_CONCURRENCY}")
    print("=" * 55)
    print(f"\n  【正样本】共 {len(pos_tasks)} 个")
    print(f"    阿里云: {pos_aliyun}  Kokoro: {pos_kokoro}  Edge: {pos_edge}")
    print(f"    train: {pos_n_train}  test: {len(pos_tasks) - pos_n_train}")
    print(f"\n  【负样本】共 {len(neg_tasks)} 个")
    print(f"    阿里云: {neg_aliyun}  Kokoro: {neg_kokoro}  Edge: {neg_edge}")
    print(f"    train: {neg_n_train}  test: {len(neg_tasks) - neg_n_train}")
    print(f"\n  总计: {len(all_tasks)} 个")
    print("=" * 55)

    aliyun_sem = asyncio.Semaphore(ALIYUN_CONCURRENCY)
    kokoro_sem = asyncio.Semaphore(KOKORO_CONCURRENCY)
    edge_sem   = asyncio.Semaphore(EDGE_CONCURRENCY)

    stats = {"success": 0, "fail": 0, "skip": 0}
    pbar  = tqdm(total=len(all_tasks), desc="生成进度")

    await asyncio.gather(
        *[gen_aliyun(t, aliyun_sem, stats) for t in all_tasks if t["source"] == "aliyun"],
        *[gen_kokoro(t, kokoro_sem, stats) for t in all_tasks if t["source"] == "kokoro"],
        *[gen_edge(t, edge_sem, stats)     for t in all_tasks if t["source"] == "edge"],
    )

    pbar.close()

    pos_train = len([f for f in os.listdir(POSITIVE_TRAIN_DIR) if f.endswith(".wav")])
    pos_test  = len([f for f in os.listdir(POSITIVE_TEST_DIR)  if f.endswith(".wav")])
    neg_train = len([f for f in os.listdir(NEGATIVE_TRAIN_DIR) if f.endswith(".wav")])
    neg_test  = len([f for f in os.listdir(NEGATIVE_TEST_DIR)  if f.endswith(".wav")])

    print(f"\n{'=' * 55}")
    print(f"  ✅ 完成！")
    print(f"     成功: {stats['success']}  跳过: {stats['skip']}  失败: {stats['fail']}")
    print(f"     positive_train: {pos_train}  positive_test: {pos_test}")
    print(f"     negative_train: {neg_train}  negative_test: {neg_test}")
    print(f"{'=' * 55}")

asyncio.run(main())