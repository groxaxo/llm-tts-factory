# llm-tts-factory: End-to-End LLM-Backbone TTS Training Framework

llm-tts-factory is a full suite of end-to-end training scripts designed for building llm backbone based TTS model from scratch. 

To begin with, taking inspiration from [Soprano](https://huggingface.co/ekwek/Soprano-1.1-80M), this repository allows you to train a Soprano style TTS model from the ground up. Because of its architecture, it features an **extra Decoder training step**. Instead of generating audio directly from discrete tokens, this model uses the **hidden states of the LLM** as inputs to generate high-fidelity audio. 

I think this can be nicely setup for anybody to train llm backbone TTS models from scratch. Can spend some time to make it more modular and user friendly. Would love to hear from interested people on this!
---

## Prerequisites

- Install dependencies from `requirements.txt`.
- Some scripts download model files from Hugging Face Hub, so make sure you have access to the referenced repos/checkpoints.
- Training dataset JSON format is:
  - `[text, audio_tokens, audio_path]`
  - where `audio_tokens` is a list of codec token ids and `audio_path` is a valid waveform path.

---

## Architecture

The framework is divided into three core stages:

### 1. Codec
- **Description:** Encodes raw audio into discrete units and decodes them back.
- **Current State:** A very naive codec encoder and decoder. There is a ton of scope for improvements here (e.g., swapping in RVQ, DAC, or EnCodec).

### 2. LLM Backbone
- **Model:** Qwen-based causal language model.
- **Description:** The core sequence-to-sequence autoregressive model. Takes text and predicts discrete audio representations, passing its hidden states to the decoder.

### 3. Decoder
- **Model:** Vocos-based decoder.
- **Description:** A dedicated vocoder trained with Multi-Resolution STFT and GAN losses to synthesize the final audio waveform directly from the LLM's continuous hidden states.
- **Training strategy:** Trained in multiple stages, first to nail reconstruction and then the perception quality using gan losses.

---

## Training & Inference Commands

You can train the entire pipeline from scratch using the following step-by-step commands.

### 1. Codec Stage
Train the audio codec encoder and decoder:
```bash
python codec_train.py
```

To build dataset JSON files, both data generation scripts now default to the public `ekwek/Soprano-Encoder` checkpoint and optionally accept `--codec-ckpt-path <local_checkpoint.pt>` when you want to use a custom codec checkpoint.

### 2. LLM Stage
Train the Qwen-based causal LLM to learn the mapping from text to audio representations:
```bash
python train_llm.py \
    --input-dir <input_dir> \
    --save-dir <save_dir> \
    --pretrained-ckpt-path <optional_llm_ckpt_path> \
    --from-scratch
```

### 3. Decoder Stage
Freeze the LLM and train the Vocos decoder to reconstruct the audio from the LLM's hidden states:
```bash
python train_decoder.py \
    --input-dir <input_dir> \
    --save-dir <save_dir> \
    --llm-ckpt-path <llm_ckpt_path>
```

### 4. Inference
Run end-to-end inference using your trained LLM and Decoder pair to generate TTS:
```bash
python simple_inference.py \
    --text "hello, my name is soma siddhartha" \
    --llm-path <llm_path> \
    --decoder-path <decoder_path> \
    --out simple_inf_out.wav
```

### 5. Local Web Demo (5 trabalenguas)
After training, you can generate five Spanish tongue-twister samples and serve them in a simple local page:

```bash
mkdir -p web_demo/audio
CUDA_VISIBLE_DEVICES=0 python simple_inference.py --text "Tres tristes tigres tragan trigo en un trigal de Tristán." --llm-path ./ckpt/llm/checkpoint-2500/model.safetensors --decoder-path ./ckpt/decoder/decoder_trained.pth --out ./web_demo/audio/trabalenguas_01.wav
```

Then serve the folder:

```bash
cd web_demo
python -m http.server 8010
```

Open `http://127.0.0.1:8010` and test the generated files in `web_demo/audio/`.

### 6. Whisper Quality Check (local OpenAI-compatible endpoint)
If you have a local transcription API on `http://127.0.0.1:5092/v1`, you can sanity-check generated audio:

```bash
curl -sS -X POST http://127.0.0.1:5092/v1/audio/transcriptions \
  -F "file=@web_demo/audio/trabalenguas_01.wav" \
  -F "model=whisper-1"
```

An empty/near-empty transcription usually means the TTS model is undertrained or misaligned.

### 7. Quality-first Training Recipe (recommended)
Avoid short from-scratch runs for quality checks. Use longer runs and pass checkpoints explicitly.

LLM training (recommended start: pretrained weights, not from scratch):
```bash
CUDA_VISIBLE_DEVICES=0 python train_llm.py \
  --input-dir <input_dir> \
  --save-dir <save_dir> \
  --max-steps 150000 \
  --batch-size 64 \
  --grad-accum-steps 1 \
  --save-freq 5000
```
Monitor `train/pred_audio_unique` (logged by `train_llm.py`) to catch token collapse early.

Decoder stage 1 (reconstruction):
```bash
CUDA_VISIBLE_DEVICES=0 python train_decoder.py \
  --input-dir <input_dir> \
  --save-dir <decoder_save_dir> \
  --llm-ckpt-path <llm_checkpoint_dir_or_model.safetensors> \
  --max-steps 120000
```

Decoder stage 2 (GAN refinement):
```bash
CUDA_VISIBLE_DEVICES=0 python train_decoder.py \
  --input-dir <input_dir> \
  --save-dir <decoder_save_dir> \
  --llm-ckpt-path <llm_checkpoint_dir_or_model.safetensors> \
  --decoder-ckpt-path <decoder_stage1_ckpt.pth> \
  --use-disc \
  --max-steps 200000 \
  --start-step 120000
```
