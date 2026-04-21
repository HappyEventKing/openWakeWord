"""
训练数据自动下载模块。
整合 RIR、背景音、验证集特征的下载逻辑。
"""
import os
import requests
import tarfile
import zipfile
import urllib3
from tqdm import tqdm

urllib3.disable_warnings()


def download_file(url, filename, desc="下载中"):
    """带进度条的文件下载。"""
    if os.path.exists(filename):
        print(f"  ✅ {filename} 已存在，跳过下载")
        return True

    print(f"\n{desc}: {filename}")
    try:
        r = requests.get(url, stream=True, verify=False, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        mb_done = downloaded / 1024 / 1024
                        mb_total = total / 1024 / 1024
                        print(f"  {pct:.1f}% | {mb_done:.0f} MB / {mb_total:.0f} MB", end="\r")
        print(f"\n  ✅ {filename} 下载完成")
        return True
    except Exception as e:
        print(f"\n  ❌ 下载失败: {e}")
        return False


def download_rir(output_dir="./mit_rirs"):
    """下载 MIT Room Impulse Response 数据集。"""
    os.makedirs(output_dir, exist_ok=True)
    zip_path = "mit_rir.zip"
    url = "https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip"

    if os.path.exists(os.path.join(output_dir, "Audio")):
        print("  ✅ MIT RIR 已存在，跳过")
        return True

    if download_file(url, zip_path, "下载 MIT RIR"):
        print("  解压中...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(output_dir)
        os.remove(zip_path)
        print("  ✅ MIT RIR 准备完成")
        return True
    return False


def download_background(output_dir="."):
    """下载 MUSAN 背景音数据集。"""
    tar_path = "musan.tar.gz"
    url = "https://www.openslr.org/resources/17/musan.tar.gz"

    if os.path.exists("./musan"):
        print("  ✅ MUSAN 已存在，跳过")
        return True

    if download_file(url, tar_path, "下载 MUSAN (约 11GB)"):
        print("  解压中...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(output_dir)
        os.remove(tar_path)
        print("  ✅ MUSAN 准备完成")
        return True
    return False


def download_validation(output_path="./validation_set_features.npy"):
    """下载验证集特征文件。"""
    if os.path.exists(output_path):
        print(f"  ✅ {output_path} 已存在，跳过")
        return True

    url = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy"
    return download_file(url, output_path, "下载验证集特征")


def download_negative_features(output_path="./openwakeword_features_ACAV100M_2000_hrs_16bit.npy"):
    """下载预计算负样本特征（约 8GB）。"""
    if os.path.exists(output_path):
        print(f"  ✅ {output_path} 已存在，跳过")
        return True

    url = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    return download_file(url, output_path, "下载负样本特征 (约 8GB)")


def ensure_training_data(config=None):
    """
    根据训练配置检查并下载缺失的训练数据。
    在 openwakeword/train.py 中通过 auto_download_data 开关调用。
    """
    if config is None:
        config = {}

    print("=" * 50)
    print("检查训练数据...")
    print("=" * 50)

    # RIR
    rir_paths = config.get("rir_paths", ["./mit_rirs"])
    for p in rir_paths:
        download_rir(os.path.dirname(p) if "/" in p else p)

    # Background
    bg_paths = config.get("background_paths", ["./musan/music", "./musan/noise", "./musan/speech"])
    if bg_paths and any("musan" in p for p in bg_paths):
        download_background()

    # Validation features
    val_path = config.get("false_positive_validation_data_path", "./validation_set_features.npy")
    download_validation(val_path)

    # Negative features (optional but recommended)
    neg_paths = config.get("feature_data_files", {})
    for name, path in neg_paths.items():
        if name != "positive" and not os.path.exists(path):
            download_negative_features(path)

    print("\n训练数据检查完成！")
    return True


if __name__ == "__main__":
    ensure_training_data()
