import requests
import os
import zipfile

os.makedirs("mit_rirs", exist_ok=True)

# MIT RIR 数据集备用地址
url = "https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip"

print("从 MIT 官网下载 RIR...")
r = requests.get(url, stream=True, verify=False)
total = int(r.headers.get('content-length', 0))
downloaded = 0

with open("mit_rir.zip", 'wb') as f:
    for chunk in r.iter_content(chunk_size=1024*1024):
        f.write(chunk)
        downloaded += len(chunk)
        if total:
            print(f"  {downloaded/1024/1024:.1f} MB / {total/1024/1024:.1f} MB", end='\r')

print("\n解压中...")
with zipfile.ZipFile("mit_rir.zip", 'r') as z:
    z.extractall("mit_rirs")

print("✅ 完成！")