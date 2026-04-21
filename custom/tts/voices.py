"""
TTS 音色列表与参数配置管理。
支持从训练配置文件覆盖默认值。
"""

# ===== 阿里云音色 =====
ALIYUN_MODELS_DEFAULT = [
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

# ===== Kokoro 音色 =====
KOKORO_VOICES_DEFAULT = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jadzia",
    "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river",
    "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric",
    "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck",
    "am_santa", "bf_alice", "bf_emma", "bf_lily", "bm_daniel",
    "bm_fable", "bm_george", "bm_lewis",
]

# ===== Edge TTS 音色 =====
EDGE_VOICES_DEFAULT = [
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

# ===== 参数组合默认值 =====
TTS_PARAMS_DEFAULT = {
    "aliyun": {
        "speech_rates": [-200, -100, 0, 100, 200],
        "pitch_rates": [-200, 0, 200],
    },
    "kokoro": {
        "speeds": [0.75, 0.9, 1.0, 1.1, 1.25],
    },
    "edge": {
        "rates": ["-20%", "-10%", "+0%", "+10%", "+20%"],
        "pitches": ["-10Hz", "+0Hz", "+10Hz"],
    },
}


def get_voices(provider: str, config=None):
    """获取指定平台的音色列表。优先从训练配置读取，否则使用默认值。"""
    if config:
        key_map = {
            "aliyun": "aliyun_models",
            "kokoro": "kokoro_voices",
            "edge": "edge_voices",
        }
        key = key_map.get(provider)
        if key and key in config:
            return config[key]

    defaults = {
        "aliyun": ALIYUN_MODELS_DEFAULT,
        "kokoro": KOKORO_VOICES_DEFAULT,
        "edge": EDGE_VOICES_DEFAULT,
    }
    return defaults.get(provider, [])


def get_params(provider: str, config=None):
    """获取指定平台的参数组合。优先从训练配置读取，否则使用默认值。"""
    if config:
        tts_params = config.get("tts_params", {})
        if provider in tts_params:
            return tts_params[provider]
    return TTS_PARAMS_DEFAULT.get(provider, {})


def list_all_voices():
    """返回所有平台的音色列表（用于打印）"""
    return {
        "aliyun": ALIYUN_MODELS_DEFAULT,
        "kokoro": KOKORO_VOICES_DEFAULT,
        "edge": EDGE_VOICES_DEFAULT,
    }
