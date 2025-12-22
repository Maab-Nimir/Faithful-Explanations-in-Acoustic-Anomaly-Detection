import os
import torch
import torchaudio
import soundfile as sf

SR = 16000
FRAME_SIZE = int(0.064 * SR)   # 1024 samples
HOP_SIZE   = int(0.032 * SR)   # 512 samples
N_FFT      = 1024
N_MELS     = 128               # 128 mel bands
EPS        = 1e-10

# Mel-spectrogram transform
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR,
    n_fft=N_FFT,
    win_length=FRAME_SIZE,
    hop_length=HOP_SIZE,
    window_fn=torch.hann_window,
    n_mels=N_MELS,
    power=2.0,
)
    
def wav_to_logmel_tensor(wav_path):
    waveform, sr = sf.read(wav_path) # waveform is numpy array [n_samples]
    waveform = torch.from_numpy(waveform).float().unsqueeze(0)  # [1, n_samples]

    # Resample if needed
    if sr != SR:
        waveform = torchaudio.functional.resample(waveform, sr, SR)

    mel_spec = mel_transform(waveform)
    log_mel = torch.log10(mel_spec + EPS)
    return log_mel.squeeze(0)  # [128, ~312]

def convert_tree(input_root, output_root):
    for root, _, files in os.walk(input_root):
        rel = os.path.relpath(root, input_root)
        out_dir = os.path.join(output_root, rel)
        os.makedirs(out_dir, exist_ok=True)

        for fname in files:
            if fname.endswith(".wav"):
                wav_path = os.path.join(root, fname)
                out_path = os.path.join(out_dir, fname.replace(".wav", ".pt"))
                tensor = wav_to_logmel_tensor(wav_path)
                torch.save(tensor, out_path)
                print(f"Saved {out_path} — shape: {tuple(tensor.shape)}")

if __name__ == "__main__":
    input_root = "/home/ulaval.ca/maelr5/scratch/acoustic-monitoring/dcase2022/development_data"   # path to dcase wavs
    output_root = "/home/ulaval.ca/maelr5/scratch/acoustic-monitoring/dcase2022/dev_spectrograms" # new path for saved spectrograms
    convert_tree(input_root, output_root)
