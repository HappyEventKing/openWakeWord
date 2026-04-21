"""
兼容根目录运行的 wrapper。
实际逻辑位于 custom/cli/list_voices.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from custom.cli.list_voices import main

if __name__ == "__main__":
    main()
