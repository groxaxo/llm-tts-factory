import json

import torch
import torchaudio
from torch.utils.data import Dataset


SAMPLES_PER_TOKEN = 2048


class AudioDataset(Dataset):
    def __init__(self, path, sample_rate=32000):
        with open(path, encoding='utf-8') as f:
            self.dataset = json.load(f)
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text, audio_tokens, audio_path = self.dataset[idx]
        prompt = f"[TEXT]{text}[START]{''.join(f'[{token}]' for token in audio_tokens)}[STOP]"

        wav, sr = torchaudio.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.sample_rate)

        return prompt, wav.squeeze(0).to(torch.float32), len(audio_tokens)
