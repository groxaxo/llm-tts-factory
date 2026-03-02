import torch
import torchaudio
import argparse
import os
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from safetensors.torch import load_file
# Ensure decoder module is importable
from decoder.decoder import SopranoDecoder

def load_models(llm_path, decoder_path, device='cuda'):
    print(f"Loading LLM from {llm_path}...")
    
    # Load LLM Config & Model
    config = AutoConfig.from_pretrained('ekwek/Soprano-80M')
    llm = AutoModelForCausalLM.from_config(config)
    
    # Load LLM weights
    if llm_path.endswith('.safetensors'):
        print("loading llm from custom trained model: ", llm_path)
        state_dict = load_file(llm_path)
        llm.load_state_dict(state_dict)
    else:
        # Fallback for folder/bin
        llm = AutoModelForCausalLM.from_pretrained(llm_path)
    
    llm.to(device).eval()
    
    print(f"Loading Decoder from {decoder_path}...")
    # Instantiate Decoder with defaults (matching train_decoder.py)
    # If training used non-defaults, user must manually edit this line.
    decoder = SopranoDecoder()
    
    # Load Decoder weights
    # Map to cpu first to avoid OOM or device mismatch during load
    print("Loading decoder from: ", decoder_path)
    decoder_state = torch.load(decoder_path, map_location='cpu')
    decoder.load_state_dict(decoder_state)
    decoder.to(device).eval()
    
    return llm, decoder

def generate_audio(text, llm, decoder, tokenizer, device='cuda', save_path="output.wav"):
    # 1. Format Prompt
    prompt = f"[TEXT]{text}[START]"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print(f"Prompt: {prompt}")
    print("Generating Tokens & Extracting Hidden States...")
    
    # 2. Generate with Hidden States Extraction
    if 'token_type_ids' in inputs: del inputs['token_type_ids']
    
    with torch.no_grad():
        outputs = llm.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'] if 'attention_mask' in inputs else None,
            max_new_tokens=512, 
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_hidden_states=True,
            repetition_penalty=1.2
        )
    
    # 3. Process Hidden States
    # outputs.hidden_states is a tuple of tuples.
    # Outer tuple: one per generation step.
    # Inner tuple: one per layer.
    
    hidden_states_list = []
    generated_ids = outputs.sequences[:, inputs['input_ids'].shape[1]:]
    step_hidden_states = outputs.hidden_states[1:]  # skip prefill state

    for token_id, step_states in zip(generated_ids[0].tolist(), step_hidden_states):
        if token_id == tokenizer.eos_token_id:
            break
        if 3 < token_id <= 8003:
            last_layer_state = step_states[-1][0, -1, :]
            hidden_states_list.append(last_layer_state)

    if not hidden_states_list:
        print("No audio tokens generated! Aborting.")
        return

    # Result: (B, T_audio, D)
    audio_hidden = torch.stack(hidden_states_list).unsqueeze(0)
    
    # Ensure float32
    audio_hidden = audio_hidden.to(torch.float32)

    num_audio_tokens = audio_hidden.size(1)
    print(f"Generated {num_audio_tokens} audio tokens.")

    # 4. Decode
    # Decoder expects (B, Channels, T)
    decoder_input = audio_hidden.transpose(1, 2)
    
    print(f"Decoding shape: {decoder_input.shape}...")
    with torch.no_grad():
        audio = decoder(decoder_input)
    
    # 5. Save
    audio = audio.squeeze().cpu() # (Samples,) or (1, Samples)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
        
    torchaudio.save(save_path, audio, 32000)
    print(f"Audio saved to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--llm-path", type=str, required=True, help="Path to LLM checkpoint")
    parser.add_argument("--decoder-path", type=str, required=True, help="Path to Decoder checkpoint")
    parser.add_argument("--out", type=str, default="output.wav", help="Output content")
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained('ekwek/Soprano-80M')

    tokenizer.eos_token_id = 3
    
    llm, decoder = load_models(args.llm_path, args.decoder_path, device)
    
    generate_audio(args.text, llm, decoder, tokenizer, device, args.out)

if __name__ == "__main__":
    main()
