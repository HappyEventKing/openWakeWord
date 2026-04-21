"""
多平台 TTS 样本生成器。
兼容 openwakeword/train.py 的 generate_samples 函数签名。
支持阿里云、Kokoro、Edge TTS 三个平台。
"""
import asyncio
import edge_tts
import os
import requests
import uuid
import dashscope
import subprocess
import random
import sys
import argparse
from itertools import product
from tqdm import tqdm
from dashscope.audio.tts import SpeechSynthesizer
from dotenv import load_dotenv
import yaml

# 加载环境变量
load_dotenv()

# 从 voices.py 导入音色管理
from .voices import get_voices, get_params

# ===== 环境变量 =====
ALIYUN_API_KEY = os.getenv("ALIYUN_API_KEY", "")
KOKORO_URL = os.getenv("KOKORO_URL", "http://localhost:8880/v1/audio/speech")
FFMPEG_EXE = os.getenv("FFMPEG_PATH", "")
if not FFMPEG_EXE:
    # 尝试自动查找 ffmpeg
    for candidate in ["ffmpeg", r"tools\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"]:
        if sys.platform == "win32" and not candidate.endswith(".exe"):
            candidate += ".exe"
        if os.path.exists(candidate) or subprocess.run([candidate, "-version"], capture_output=True).returncode == 0:
            FFMPEG_EXE = candidate
            break
    if not FFMPEG_EXE:
        FFMPEG_EXE = "ffmpeg"

# ===== 负样本短语默认值 =====
DEFAULT_NEGATIVE_PHRASES = [
    "Hey Siri", "Hey Google", "Hey Alexa", "OK Google",
    "Hey Cortana", "Hey Bixby",
    "Hey everyone", "Hey everybody", "Hey there", "Hey you",
    "Hey listen", "Hey wait",
    "Hello world", "Good morning", "What time is it",
    "Turn on the lights", "Play some music", "Set a timer",
]

pbar = None


def load_config(config_path="hey_eventi.yml"):
    """加载训练配置文件。"""
    if not os.path.exists(config_path):
        # 尝试在项目根目录查找
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        alt = os.path.join(root, config_path)
        if os.path.exists(alt):
            config_path = alt
        else:
            return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def convert_to_wav(input_path, output_path):
    """用 ffmpeg 转换为 16kHz 单声道 16-bit WAV。"""
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


def build_tasks(text, providers, voices_cfg, params_cfg, repeats=5, train_ratio=0.8,
                positive_train_dir="positive_train", positive_test_dir="positive_test",
                negative_train_dir="negative_train", negative_test_dir="negative_test",
                is_negative=False, phrase_tag=None):
    """
    构建 TTS 生成任务列表。
    text: 字符串或字符串列表
    providers: 平台列表，如 ["aliyun", "kokoro", "edge"]
    """
    tasks = []

    if isinstance(text, str):
        texts = [text]
    else:
        texts = text

    # 阿里云任务
    if "aliyun" in providers:
        models = get_voices("aliyun", voices_cfg)
        aliyun_params = get_params("aliyun", params_cfg)
        speech_rates = aliyun_params.get("speech_rates", [-200, -100, 0, 100, 200])
        pitch_rates = aliyun_params.get("pitch_rates", [-200, 0, 200])
        for repeat in range(repeats):
            for model in models:
                for sr, pr in product(speech_rates, pitch_rates):
                    tasks.append({
                        "kind": "negative" if is_negative else "positive",
                        "source": "aliyun",
                        "text": random.choice(texts),
                        "model": model,
                        "speech_rate": sr,
                        "pitch_rate": pr,
                        "repeat": repeat,
                        "phrase_tag": phrase_tag,
                    })

    # Kokoro 任务
    if "kokoro" in providers:
        kokoro_voices = get_voices("kokoro", voices_cfg)
        kokoro_params = get_params("kokoro", params_cfg)
        speeds = kokoro_params.get("speeds", [0.75, 0.9, 1.0, 1.1, 1.25])
        for repeat in range(repeats):
            for voice in kokoro_voices:
                for speed in speeds:
                    tasks.append({
                        "kind": "negative" if is_negative else "positive",
                        "source": "kokoro",
                        "text": random.choice(texts),
                        "voice": voice,
                        "speed": speed,
                        "repeat": repeat,
                        "phrase_tag": phrase_tag,
                    })

    # Edge 任务
    if "edge" in providers:
        edge_voices = get_voices("edge", voices_cfg)
        edge_params = get_params("edge", params_cfg)
        rates = edge_params.get("rates", ["-20%", "-10%", "+0%", "+10%", "+20%"])
        pitches = edge_params.get("pitches", ["-10Hz", "+0Hz", "+10Hz"])
        for repeat in range(repeats):
            for voice in edge_voices:
                for rate, pitch in product(rates, pitches):
                    tasks.append({
                        "kind": "negative" if is_negative else "positive",
                        "source": "edge",
                        "text": random.choice(texts),
                        "voice": voice,
                        "rate": rate,
                        "pitch": pitch,
                        "repeat": repeat,
                        "phrase_tag": phrase_tag,
                    })

    random.shuffle(tasks)
    n_train = int(len(tasks) * train_ratio)
    for idx, t in enumerate(tasks):
        t["idx"] = idx
        if is_negative:
            t["split"] = "train" if idx < n_train else "test"
        else:
            t["split"] = "train" if idx < n_train else "test"
    return tasks


def _is_explicit_split_dir(path):
    """判断路径是否已经是明确的 train/test 目录（兼容 train.py 调用方式）。"""
    p = path.lower().replace("\\", "/")
    return any(p.endswith(s) for s in ["/positive_train", "/positive_test",
                                       "/negative_train", "/negative_test"])


def get_output_path(task, base_dir):
    """根据任务类型和 split 决定输出路径。
    当 base_dir 已经是明确的 split 目录时，直接输出到该目录（兼容 train.py）。
    """
    kind = task["kind"]
    split = task["split"]
    source = task["source"]
    idx = task["idx"]

    if _is_explicit_split_dir(base_dir):
        # train.py 调用方式：output_dir 已经是目标目录
        sub = base_dir
    else:
        if kind == "positive":
            sub = os.path.join(base_dir, "positive_train" if split == "train" else "positive_test")
        else:
            sub = os.path.join(base_dir, "negative_train" if split == "train" else "negative_test")

    os.makedirs(sub, exist_ok=True)

    if kind == "positive":
        return os.path.join(sub, f"{source}_{idx:06d}.wav")
    else:
        tag = task.get("phrase_tag", "neg")
        return os.path.join(sub, f"{source}_{tag}_{idx:06d}.wav")


async def gen_aliyun(task, sem, stats, base_dir):
    """阿里云 TTS 异步生成。"""
    async with sem:
        out_path = get_output_path(task, base_dir)
        if os.path.exists(out_path):
            stats["skip"] += 1
            pbar.update(1)
            return
        temp = os.path.join(base_dir, "temp_audio", f"aliyun_{uuid.uuid4().hex}.wav")
        os.makedirs(os.path.dirname(temp), exist_ok=True)
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
            tqdm.write(f"阿里云 [{task['kind']}] {task['model']} \"{task['text']}\": {e}")
        pbar.update(1)


async def gen_kokoro(task, sem, stats, base_dir):
    """Kokoro TTS 异步生成。"""
    async with sem:
        out_path = get_output_path(task, base_dir)
        if os.path.exists(out_path):
            stats["skip"] += 1
            pbar.update(1)
            return
        temp = os.path.join(base_dir, "temp_audio", f"kokoro_{uuid.uuid4().hex}.mp3")
        os.makedirs(os.path.dirname(temp), exist_ok=True)
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
                tqdm.write(f"Kokoro [{task['kind']}] {task['voice']} \"{task['text']}\": {r.status_code}")
        except Exception as e:
            stats["fail"] += 1
            tqdm.write(f"Kokoro [{task['kind']}] {task['voice']} \"{task['text']}\": {e}")
        pbar.update(1)


async def gen_edge(task, sem, stats, base_dir):
    """Edge TTS 异步生成。"""
    async with sem:
        out_path = get_output_path(task, base_dir)
        if os.path.exists(out_path):
            stats["skip"] += 1
            pbar.update(1)
            return
        temp = os.path.join(base_dir, "temp_audio", f"edge_{uuid.uuid4().hex}.mp3")
        os.makedirs(os.path.dirname(temp), exist_ok=True)
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
            tqdm.write(f"Edge [{task['kind']}] {task['voice']} \"{task['text']}\": {e}")
        pbar.update(1)


def generate_samples(text, max_samples, batch_size, noise_scales, noise_scale_ws,
                     length_scales, output_dir, auto_reduce_batch_size=True, file_names=None):
    """
    兼容 openwakeword/train.py 的 generate_samples 函数。
    从配置文件读取 TTS 平台和音色设置。
    """
    global pbar

    # 尝试加载训练配置
    config = load_config()
    providers = config.get("tts_providers", ["aliyun", "kokoro", "edge"])
    repeats = config.get("tts_repeats", 5)
    train_ratio = config.get("tts_train_ratio", 0.8)

    voices_cfg = config
    params_cfg = config

    # 设置阿里云 API key
    if ALIYUN_API_KEY:
        dashscope.api_key = ALIYUN_API_KEY
    elif "aliyun" in providers:
        print("警告: ALIYUN_API_KEY 未设置，跳过阿里云 TTS")
        providers = [p for p in providers if p != "aliyun"]

    # 检查是否需要生成（兼容 train.py 的跳过逻辑）
    n_current = len([f for f in os.listdir(output_dir) if f.endswith(".wav")]) if os.path.exists(output_dir) else 0
    if n_current >= max_samples * 0.95:
        print(f"[跳过] {output_dir} 已有 ~{n_current} 个样本，接近目标 {max_samples}")
        return

    # 判断是正样本还是负样本目录
    is_negative = "negative" in output_dir.lower()

    # 构建任务
    if is_negative and isinstance(text, list):
        # 负样本：为每个短语单独构建任务
        all_tasks = []
        for phrase in text:
            tag = phrase.replace(" ", "_").replace("'", "")[:20]
            tasks = build_tasks(
                phrase, providers, voices_cfg, params_cfg,
                repeats=1, train_ratio=train_ratio,
                is_negative=True, phrase_tag=tag
            )
            all_tasks.extend(tasks)
    else:
        all_tasks = build_tasks(
            text, providers, voices_cfg, params_cfg,
            repeats=repeats, train_ratio=train_ratio,
            is_negative=is_negative
        )

    # 限制样本数量
    if len(all_tasks) > max_samples:
        all_tasks = all_tasks[:max_samples]

    # 并发控制
    aliyun_sem = asyncio.Semaphore(config.get("aliyun_concurrency", 5))
    kokoro_sem = asyncio.Semaphore(config.get("kokoro_concurrency", 1))
    edge_sem = asyncio.Semaphore(config.get("edge_concurrency", 20))

    stats = {"success": 0, "fail": 0, "skip": 0}
    pbar = tqdm(total=len(all_tasks), desc=f"生成 {'负' if is_negative else '正'}样本")

    # 兼容 train.py：如果 output_dir 已经是明确的 split 目录，直接用它作为 base_dir
    base_dir = output_dir if _is_explicit_split_dir(output_dir) else os.path.dirname(output_dir)

    async def run_all():
        await asyncio.gather(
            *[gen_aliyun(t, aliyun_sem, stats, base_dir) for t in all_tasks if t["source"] == "aliyun"],
            *[gen_kokoro(t, kokoro_sem, stats, base_dir) for t in all_tasks if t["source"] == "kokoro"],
            *[gen_edge(t, edge_sem, stats, base_dir) for t in all_tasks if t["source"] == "edge"],
        )

    asyncio.run(run_all())
    pbar.close()

    print(f"  成功: {stats['success']}  跳过: {stats['skip']}  失败: {stats['fail']}")


def main():
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="多平台 TTS 样本生成")
    parser.add_argument("--config", default="hey_eventi.yml", help="训练配置文件路径")
    parser.add_argument("--wake-word", default="", help="唤醒词文本（覆盖配置）")
    parser.add_argument("--providers", default="", help="TTS 平台，逗号分隔: aliyun,kokoro,edge")
    parser.add_argument("--output-dir", default="", help="输出目录（覆盖配置）")
    parser.add_argument("--positive-only", action="store_true", help="只生成正样本")
    parser.add_argument("--negative-only", action="store_true", help="只生成负样本")
    args = parser.parse_args()

    config = load_config(args.config)
    wake_word = args.wake_word or config.get("target_phrase", ["Hey Eventi"])[0]
    output_dir = args.output_dir or config.get("output_dir", "./hey_eventi_model")
    providers = args.providers.split(",") if args.providers else config.get("tts_providers", ["aliyun", "kokoro", "edge"])

    model_name = config.get("model_name", "model")
    base_dir = os.path.join(output_dir, model_name)
    os.makedirs(base_dir, exist_ok=True)

    if ALIYUN_API_KEY:
        dashscope.api_key = ALIYUN_API_KEY
    elif "aliyun" in providers:
        print("警告: ALIYUN_API_KEY 未设置，跳过阿里云 TTS")
        providers = [p for p in providers if p != "aliyun"]

    voices_cfg = config
    params_cfg = config
    repeats = config.get("tts_repeats", 5)
    train_ratio = config.get("tts_train_ratio", 0.8)

    if not args.negative_only:
        print(f"\n生成正样本: \"{wake_word}\"")
        tasks = build_tasks(wake_word, providers, voices_cfg, params_cfg, repeats=repeats, train_ratio=train_ratio)
        _run_generation(tasks, base_dir, "正样本")

    if not args.positive_only:
        print(f"\n生成负样本...")
        negative_phrases = config.get("custom_negative_phrases", DEFAULT_NEGATIVE_PHRASES)
        all_neg_tasks = []
        for phrase in negative_phrases:
            tag = phrase.replace(" ", "_").replace("'", "")[:20]
            tasks = build_tasks(phrase, providers, voices_cfg, params_cfg,
                                repeats=1, train_ratio=train_ratio, is_negative=True, phrase_tag=tag)
            all_neg_tasks.extend(tasks)
        _run_generation(all_neg_tasks, base_dir, "负样本")

    print("\n完成！")


def _run_generation(tasks, base_dir, label):
    """执行一批生成任务。"""
    global pbar
    aliyun_sem = asyncio.Semaphore(5)
    kokoro_sem = asyncio.Semaphore(1)
    edge_sem = asyncio.Semaphore(20)
    stats = {"success": 0, "fail": 0, "skip": 0}
    pbar = tqdm(total=len(tasks), desc=f"生成{label}")

    async def run():
        await asyncio.gather(
            *[gen_aliyun(t, aliyun_sem, stats, base_dir) for t in tasks if t["source"] == "aliyun"],
            *[gen_kokoro(t, kokoro_sem, stats, base_dir) for t in tasks if t["source"] == "kokoro"],
            *[gen_edge(t, edge_sem, stats, base_dir) for t in tasks if t["source"] == "edge"],
        )

    asyncio.run(run())
    pbar.close()
    print(f"  {label}: 成功 {stats['success']}  跳过 {stats['skip']}  失败 {stats['fail']}")
