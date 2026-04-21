"""
唤醒词模型测试脚本
用法: python test_wakeup.py --model ./hey_eventi_model/hey_eventi.tflite --audio <wav文件或目录>
"""
import argparse
import glob
import os
import wave
import numpy as np
import openwakeword


def load_wav(path):
    with wave.open(path, mode='rb') as f:
        nchannels = f.getnchannels()
        sampwidth = f.getsampwidth()
        framerate = f.getframerate()
        data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        if nchannels == 2:
            data = data.reshape(-1, 2).mean(axis=1).astype(np.int16)
        if framerate != 16000:
            print(f"[!] 警告: {path} 采样率是 {framerate}Hz，需要 16000Hz")
        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./hey_eventi_model/hey_eventi.tflite")
    parser.add_argument("--audio", required=True, help="WAV 文件或目录")
    parser.add_argument("--threshold", type=float, default=0.5, help="唤醒阈值")
    parser.add_argument("--framework", default="tflite", choices=["tflite", "onnx"])
    args = parser.parse_args()

    print(f"加载模型: {args.model} ({args.framework})")
    oww = openwakeword.Model(
        wakeword_models=[args.model],
        inference_framework=args.framework,
    )
    print(f"模型输入帧数: {oww.model_inputs}")
    print()

    if os.path.isdir(args.audio):
        files = sorted(glob.glob(os.path.join(args.audio, "*.wav")))
    else:
        files = [args.audio]

    triggered = 0
    for path in files:
        data = load_wav(path)
        padded = np.concatenate([
            np.zeros(16000, dtype=np.int16),
            data,
            np.zeros(16000, dtype=np.int16)
        ])
        predictions = oww.predict_clip(padded, padding=0, chunk_size=1280)

        scores = [p[list(p.keys())[0]] for p in predictions]
        max_score = max(scores) if scores else 0.0
        status = "[触发]" if max_score >= args.threshold else ""
        if max_score >= args.threshold:
            triggered += 1

        print(f"{os.path.basename(path):30s} max_score={max_score:.4f} {status}")
        oww.reset()

    print()
    print(f"共测试 {len(files)} 个文件，触发 {triggered} 个 (阈值 {args.threshold})")


if __name__ == "__main__":
    main()
