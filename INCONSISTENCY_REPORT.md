# Project Inconsistency Analysis Report

This report summarizes current inconsistencies found in the repository and proposes practical fixes and improvements.

## Fixes applied in this change

1. **Dependency mismatch fixed**
   - `requirements.txt` now includes dependencies already imported by the codebase:
     - `safetensors`
     - `matplotlib`
     - `wandb`

2. **Missing decoder dataset module fixed**
   - Added `dataset_e2e.py` to match the existing import in `train_decoder.py`.
   - Implemented:
     - `AudioDataset` loader for `[text, audio_tokens, audio_path]` JSON rows
     - waveform loading + mono conversion + resampling
     - prompt formatting expected by decoder training
     - `SAMPLES_PER_TOKEN = 2048`

3. **Blocking debug breakpoints removed**
   - Removed active `pdb.set_trace()` calls from:
     - `train_llm.py`
     - `simple_inference.py`
     - `generate_dataset.py`

## Remaining inconsistencies and improvement opportunities

1. **Hardcoded absolute checkpoint paths**
   - Present in scripts such as `train_llm.py`, `train_decoder.py`, `codec_train.py`, and dataset generation scripts.
   - Improvement: move all checkpoint/data paths to CLI flags (or env/config), with clear defaults.

2. **Import path inconsistency in dataset generation scripts**
   - `generate_dataset.py` and `generate_dataset_from_lists.py` use `from encoder.codec import Encoder` while code lives under `codec/encoder/codec.py`.
   - Improvement: normalize imports to package-consistent paths and document execution context.

3. **Tokenizer/model identifier hardcoding**
   - `TOKENIZER_NAME` and equivalent values are fixed to `ekwek/Soprano-80M` in multiple scripts.
   - Improvement: expose via CLI argument and/or centralized config.

4. **Documentation coverage gaps**
   - README has core commands but limited mention of setup constraints (HF auth, required checkpoints, expected dataset JSON schema).
   - Improvement: add a “Prerequisites” and “Data Format” section.

5. **Limited automated validation**
   - No existing tests were discoverable in this environment.
   - Improvement: add lightweight smoke tests for:
     - dataset loaders
     - prompt formatting
     - argument parsing and script startup

## Suggested priority order

1. Parameterize hardcoded paths.
2. Normalize import paths for all entry scripts.
3. Expand README with prerequisites + dataset schema.
4. Add smoke tests for critical data and startup paths.
