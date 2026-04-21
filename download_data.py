import requests
import os

def download_file(url, filename):
    print(f"\n开始下载: {filename}")
    r = requests.get(url, stream=True)
    total = int(r.headers.get('content-length', 0))
    downloaded = 0
    with open(filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024*1024):  # 1MB chunks
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                mb_done = downloaded / 1024 / 1024
                mb_total = total / 1024 / 1024
                print(f"  {pct:.1f}% | {mb_done:.0f} MB / {mb_total:.0f} MB", end='\r')
    print(f"\n  ✅ {filename} 下载完成！")

# 验证集（约 500MB）
download_file(
    "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy",
    "validation_set_features.npy"
)

# 训练负样本特征（约 8GB）
download_file(
    "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
    "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
)