"""Download and convert HuggingFace dataset to LJSpeech format for llm-tts-factory."""
import os
import soundfile as sf
import numpy as np
from datasets import load_dataset

OUTPUT_DIR = "/home/op/llm-tts-factory/data/argentinian-spanish-female"
WAV_DIR = os.path.join(OUTPUT_DIR, "wavs")
os.makedirs(WAV_DIR, exist_ok=True)

print("Loading dataset (female config)...")
ds = load_dataset("ylacombe/google-argentinian-spanish", "female", split="train")
print(f"Total samples: {len(ds)}")

metadata_lines = []
for idx, sample in enumerate(ds):
    filename = f"AR_F_{idx:06d}"
    text = sample["text"]
    audio = sample["audio"]
    
    # Get audio array and sampling rate
    arr = np.array(audio["array"], dtype=np.float32)
    sr = audio["sampling_rate"]
    
    wav_path = os.path.join(WAV_DIR, f"{filename}.wav")
    sf.write(wav_path, arr, sr)
    
    # LJSpeech metadata format: filename|transcription
    metadata_lines.append(f"{filename}|{text}")
    
    if (idx + 1) % 100 == 0:
        print(f"  Processed {idx + 1}/{len(ds)} samples")

# Write metadata
meta_path = os.path.join(OUTPUT_DIR, "metadata_orig.csv")
with open(meta_path, "w", encoding="utf-8") as f:
    f.write("\n".join(metadata_lines))

print(f"\nDone! {len(metadata_lines)} samples saved.")
print(f"  WAVs: {WAV_DIR}")
print(f"  Metadata: {meta_path}")
print(f"  Sample rate: {sr}")
