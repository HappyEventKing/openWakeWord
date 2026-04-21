import asyncio
import edge_tts

async def main():
    voices = await edge_tts.list_voices()
    en_voices = [v for v in voices if v["Locale"].startswith("en-")]
    print(f"英文音色共 {len(en_voices)} 个：")
    for v in en_voices:
        print(f"  {v['ShortName']}  ({v['Gender']})")

    zh_voices = [v for v in voices if v["Locale"].startswith("zh-")]
    print(f"中文音色共 {len(zh_voices)} 个：")
    for v in zh_voices:
        print(f"  {v['ShortName']}  ({v['Gender']})")

asyncio.run(main())