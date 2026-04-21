"""
兼容 openwakeword/train.py 的 wrapper。
支持 CLI 直接运行和 train.py import 两种方式。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from custom.tts.generator import generate_samples, main as cli_main

if __name__ == "__main__":
    cli_main()
