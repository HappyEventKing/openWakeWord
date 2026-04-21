"""
统一音色列表查询工具。
支持 Edge、Kokoro、阿里云三个平台。
"""
import asyncio
import argparse
import sys
import os

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _ROOT)
from custom.tts.voices import list_all_voices, ALIYUN_MODELS_DEFAULT, KOKORO_VOICES_DEFAULT, EDGE_VOICES_DEFAULT


async def list_edge():
    """通过 edge-tts 库动态获取所有可用音色。"""
    import edge_tts
    voices = await edge_tts.list_voices()
    en = [v for v in voices if v["Locale"].startswith("en-")]
    zh = [v for v in voices if v["Locale"].startswith("zh-")]
    other = [v for v in voices if not v["Locale"].startswith(("en-", "zh-"))]

    print(f"\n=== Edge TTS 音色 (共 {len(voices)} 个) ===")
    print(f"\n英文 ({len(en)} 个):")
    for v in en:
        print(f"  {v['ShortName']} ({v['Gender']}, {v['Locale']})")
    print(f"\n中文 ({len(zh)} 个):")
    for v in zh:
        print(f"  {v['ShortName']} ({v['Gender']}, {v['Locale']})")
    if other:
        print(f"\n其他语言 ({len(other)} 个):")
        for v in other:
            print(f"  {v['ShortName']} ({v['Gender']}, {v['Locale']})")


def list_kokoro():
    """列出 Kokoro 默认音色。"""
    print(f"\n=== Kokoro 音色 (共 {len(KOKORO_VOICES_DEFAULT)} 个) ===")
    groups = {}
    for v in KOKORO_VOICES_DEFAULT:
        prefix = v.split("_")[0]
        groups.setdefault(prefix, []).append(v)
    for prefix, voices in sorted(groups.items()):
        print(f"\n前缀 '{prefix}' ({len(voices)} 个):")
        for v in voices:
            print(f"  {v}")


def list_aliyun():
    """列出阿里云默认模型。"""
    print(f"\n=== 阿里云 TTS 模型 (共 {len(ALIYUN_MODELS_DEFAULT)} 个) ===")
    for m in ALIYUN_MODELS_DEFAULT:
        print(f"  {m}")


def main():
    parser = argparse.ArgumentParser(description="列出 TTS 平台可用音色")
    parser.add_argument("--provider", choices=["edge", "kokoro", "aliyun", "all"],
                        default="all", help="指定平台，默认列出所有")
    args = parser.parse_args()

    if args.provider in ("all", "edge"):
        asyncio.run(list_edge())
    if args.provider in ("all", "kokoro"):
        list_kokoro()
    if args.provider in ("all", "aliyun"):
        list_aliyun()


if __name__ == "__main__":
    main()
