import torchaudio
import soundfile as sf

def info(file_path):
    info = sf.info(file_path)
    class Info:
        num_frames = info.frames
        sample_rate = info.samplerate
        channels = info.channels
    return Info()

torchaudio.info = info
