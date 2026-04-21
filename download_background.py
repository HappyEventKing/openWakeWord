import requests
import tarfile
import os
import scipy.io.wavfile
import numpy as np
from tqdm import tqdm
import urllib3

urllib3.disable_warnings()

os.makedirs("fma", exist_ok=True)

# OpenSLR 是专门的语音数据集镜像站，国内可以访问
url = "https://www.openslr.org/resources/17/musan.tar.gz"

print("从 OpenSLR 下载 MUSAN 数据集...")
print("文件约 11GB，需要一段时间...")

r = requests.get(url, stream=True, verify=False)
total = int(r.headers.get('content-length', 0))
downloaded = 0

with open("musan.tar.gz", 'wb') as f:
    for chunk in r.iter_content(chunk_size=1024*1024):
        f.write(chunk)
        downloaded += len(chunk)
        if total:
            pct = downloaded/total*100
            print(f"  {pct:.1f}% - {downloaded/1024/1024:.0f} MB / {total/1024/1024:.0f} MB", end='\r')

print("\n解压中...")
with tarfile.open("musan.tar.gz", 'r:gz') as tar:
    tar.extractall(".")

print("✅ MUSAN 下载完成！")