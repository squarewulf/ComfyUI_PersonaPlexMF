import os
import sys
import json
import tarfile
import atexit
import tempfile
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
from scipy.io import wavfile
import folder_paths

# Import quantization utilities (uses PyTorch native - always available)
try:
    from .quantize import (
        check_quantization_available,
        get_quantization_error,
        get_quantization_backend,
        quantize_model_after_load,
        QUANTIZATION_AVAILABLE,
        QUANTIZATION_BACKEND,
    )
    print(f"[PersonaPlex] Quantization available (backend: {QUANTIZATION_BACKEND})")
except ImportError as e:
    print(f"[PersonaPlex] Warning: Could not load quantization module: {e}")
    QUANTIZATION_AVAILABLE = True  # PyTorch native is always available
    QUANTIZATION_BACKEND = "pytorch_native"
    def check_quantization_available(q): return q not in ("none", "None", None, "") and torch.cuda.is_available()
    def get_quantization_error(): return "Quantization module not found"
    def get_quantization_backend(): return "pytorch_native"
    def quantize_model_after_load(m, q, d): return m.to(d)
from comfy.utils import ProgressBar

__version__ = "1.3.0"

PERSONAPLEX_SRC = os.path.join(os.path.dirname(__file__), "personaplex_src", "moshi")
if PERSONAPLEX_SRC not in sys.path:
    sys.path.insert(0, PERSONAPLEX_SRC)

try:
    import sentencepiece
    import sphn
    from moshi.models import loaders, LMGen, MimiModel
    from moshi.models.lm import load_audio as lm_load_audio
    from moshi.models.lm import _iterate_audio as lm_iterate_audio
    from moshi.models.lm import encode_from_sphn as lm_encode_from_sphn
    PERSONAPLEX_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    PERSONAPLEX_AVAILABLE = False
    IMPORT_ERROR = str(e)
    print(f"[PersonaPlex] Import error: {IMPORT_ERROR}")
    print("[PersonaPlex] Make sure you've run: pip install -e personaplex_src/moshi")

VOICE_PRESETS = [
    "NATF0", "NATF1", "NATF2", "NATF3",
    "NATM0", "NATM1", "NATM2", "NATM3",
    "VARF0", "VARF1", "VARF2", "VARF3", "VARF4",
    "VARM0", "VARM1", "VARM2", "VARM3", "VARM4",
]


def wrap_with_system_tags(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def seed_all(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


PERSONAPLEX_MODELS_DIR = os.path.join(folder_paths.models_dir, "personaplex")
os.makedirs(PERSONAPLEX_MODELS_DIR, exist_ok=True)


class PersonaPlexSettings:
    """Provides all configurable settings for PersonaPlex inference and server."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Sampling settings
                "temp_audio": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Audio sampling temperature. Higher = more random."
                }),
                "temp_text": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Text sampling temperature. Higher = more random."
                }),
                "topk_audio": ("INT", {
                    "default": 250, "min": 1, "max": 2000,
                    "tooltip": "Top-K sampling for audio tokens."
                }),
                "topk_text": ("INT", {
                    "default": 25, "min": 1, "max": 500,
                    "tooltip": "Top-K sampling for text tokens."
                }),
                "use_sampling": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use sampling. If False, uses greedy decoding."
                }),
                # Timing settings
                "silence_duration": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Silence duration in seconds before response starts."
                }),
                # Reproducibility
                "seed": ("INT", {
                    "default": -1, "min": -1, "max": 2147483647,
                    "tooltip": "Random seed for reproducibility. -1 = random."
                }),
            },
            "optional": {
                # Voice settings
                "voice_preset": (["default"] + VOICE_PRESETS, {
                    "default": "default",
                    "tooltip": "Voice preset to use. 'default' uses NATF2."
                }),
                "text_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "System prompt defining the AI's persona and behavior."
                }),
                # Audio buffer settings (client-side, affects latency)
                "init_buffer_ms": ("FLOAT", {
                    "default": 400.0, "min": 50.0, "max": 1000.0, "step": 10.0,
                    "tooltip": "Initial buffer before audio playback starts (ms). Higher = more stable, higher latency."
                }),
                "partial_buffer_ms": ("FLOAT", {
                    "default": 210.0, "min": 20.0, "max": 500.0, "step": 5.0,
                    "tooltip": "Partial buffer size (ms). Lower = less latency but may cause choppy audio."
                }),
                "decoder_buffer_samples": ("FLOAT", {
                    "default": 3840.0, "min": 960.0, "max": 9600.0, "step": 240.0,
                    "tooltip": "Decoder buffer size in samples @24kHz. Default 3840 = 160ms."
                }),
                "resample_quality": ("FLOAT", {
                    "default": 5.0, "min": 0.0, "max": 10.0, "step": 1.0,
                    "tooltip": "Resampling quality (0=fast, 10=best). Higher = better audio but more CPU."
                }),
                "silence_delay_s": ("FLOAT", {
                    "default": 0.07, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Silence detection delay in seconds."
                }),
                # Server-specific settings
                "host": ("STRING", {
                    "default": "0.0.0.0",
                    "tooltip": "Host address for the server."
                }),
                "use_ssl": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable HTTPS with auto-generated certificates."
                }),
                "gradio_tunnel": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable gradio tunnel for remote access."
                }),
            },
        }
    
    RETURN_TYPES = ("PERSONAPLEX_SETTINGS",)
    RETURN_NAMES = ("settings",)
    FUNCTION = "create_settings"
    CATEGORY = "audio/PersonaPlex"
    
    def create_settings(self, temp_audio, temp_text, topk_audio, topk_text, 
                        use_sampling, silence_duration, seed,
                        voice_preset="default", text_prompt="", 
                        init_buffer_ms=400.0, partial_buffer_ms=210.0, 
                        decoder_buffer_samples=3840.0, resample_quality=5.0,
                        silence_delay_s=0.07, host="0.0.0.0",
                        use_ssl=False, gradio_tunnel=False):
        settings = {
            # Sampling
            "temp_audio": temp_audio,
            "temp_text": temp_text,
            "topk_audio": topk_audio,
            "topk_text": topk_text,
            "use_sampling": use_sampling,
            # Timing
            "silence_duration": silence_duration,
            # Reproducibility
            "seed": seed,
            # Voice
            "voice_preset": voice_preset if voice_preset != "default" else "NATF2",
            "text_prompt": text_prompt,
            # Audio buffer settings (client-side) - convert floats to ints
            "init_buffer_ms": int(init_buffer_ms),
            "partial_buffer_ms": int(partial_buffer_ms),
            "decoder_buffer_samples": int(decoder_buffer_samples),
            "resample_quality": int(resample_quality),
            "silence_delay_s": silence_delay_s,
            # Server
            "host": host,
            "use_ssl": use_ssl,
            "gradio_tunnel": gradio_tunnel,
        }
        return (settings,)


class PersonaPlexExternal:
    """Configure external PersonaPlex installation for better performance."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "python_path": ("STRING", {
                    "default": "C:\\Users\\Forge\\AppData\\Local\\Programs\\Miniconda3\\python.exe",
                    "tooltip": "Path to Python executable (e.g., Miniconda, venv)"
                }),
                "install_path": ("STRING", {
                    "default": "X:\\Code\\PersonaNew\\personaplex",
                    "tooltip": "Path to PersonaPlex installation folder"
                }),
            },
        }
    
    RETURN_TYPES = ("PERSONAPLEX_EXTERNAL",)
    RETURN_NAMES = ("external",)
    FUNCTION = "create_external"
    CATEGORY = "audio/PersonaPlex"
    
    def create_external(self, python_path, install_path):
        return ({
            "python_path": python_path.strip(),
            "install_path": install_path.strip(),
        },)


class PersonaPlexModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        model_files = []
        if os.path.exists(PERSONAPLEX_MODELS_DIR):
            for f in os.listdir(PERSONAPLEX_MODELS_DIR):
                if f.endswith(".safetensors"):
                    model_files.append(f)
        if not model_files:
            model_files = ["model.safetensors"]
        
        mimi_files = []
        if os.path.exists(PERSONAPLEX_MODELS_DIR):
            for f in os.listdir(PERSONAPLEX_MODELS_DIR):
                if f.endswith(".safetensors") and "tokenizer" in f.lower():
                    mimi_files.append(f)
                elif f.endswith(".safetensors") and "mimi" in f.lower():
                    mimi_files.append(f)
        if not mimi_files:
            mimi_files = ["tokenizer-e351c8d8-checkpoint125.safetensors"]
        
        tokenizer_files = []
        if os.path.exists(PERSONAPLEX_MODELS_DIR):
            for f in os.listdir(PERSONAPLEX_MODELS_DIR):
                if f.endswith(".model"):
                    tokenizer_files.append(f)
        if not tokenizer_files:
            tokenizer_files = ["tokenizer_spm_32k_3.model"]
        
        # Build quantize options based on availability
        quantize_options = ["none"]
        if QUANTIZATION_AVAILABLE:
            quantize_options.extend(["8bit", "4bit"])
        
        return {
            "required": {
                "moshi_model": (model_files, {"default": model_files[0] if model_files else "model.safetensors"}),
                "mimi_model": (mimi_files, {"default": mimi_files[0] if mimi_files else "tokenizer-e351c8d8-checkpoint125.safetensors"}),
                "tokenizer": (tokenizer_files, {"default": tokenizer_files[0] if tokenizer_files else "tokenizer_spm_32k_3.model"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "cpu_offload": ("BOOLEAN", {"default": False}),
                "quantize": (quantize_options, {
                    "default": "none",
                    "tooltip": "Quantization level: none (~14GB), 8bit (~8GB), 4bit (~5GB). Requires bitsandbytes."
                }),
            },
        }
    
    RETURN_TYPES = ("PERSONAPLEX_MODEL",)
    RETURN_NAMES = ("personaplex_model",)
    FUNCTION = "load_model"
    CATEGORY = "audio/PersonaPlex"

    def load_model(self, moshi_model, mimi_model, tokenizer, device, cpu_offload, quantize="none"):
        if not PERSONAPLEX_AVAILABLE:
            raise RuntimeError(f"PersonaPlex not available: {IMPORT_ERROR}")
        
        moshi_path = os.path.join(PERSONAPLEX_MODELS_DIR, moshi_model)
        mimi_path = os.path.join(PERSONAPLEX_MODELS_DIR, mimi_model)
        tokenizer_path = os.path.join(PERSONAPLEX_MODELS_DIR, tokenizer)
        
        if not os.path.exists(moshi_path):
            raise FileNotFoundError(f"Moshi model not found: {moshi_path}")
        if not os.path.exists(mimi_path):
            raise FileNotFoundError(f"Mimi model not found: {mimi_path}")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        
        voices_dir = os.path.join(PERSONAPLEX_MODELS_DIR, "voices")
        if not os.path.exists(voices_dir):
            voices_tgz = os.path.join(PERSONAPLEX_MODELS_DIR, "voices.tgz")
            if os.path.exists(voices_tgz):
                print(f"[PersonaPlex] Extracting voices to {voices_dir}")
                with tarfile.open(voices_tgz, "r:gz") as tar:
                    tar.extractall(path=PERSONAPLEX_MODELS_DIR)
            else:
                raise FileNotFoundError(f"Voices not found. Place voices.tgz or voices/ folder in {PERSONAPLEX_MODELS_DIR}")
        
        print("[PersonaPlex] Loading Mimi encoder/decoder...")
        mimi = loaders.get_mimi(mimi_path, device)
        other_mimi = loaders.get_mimi(mimi_path, device)
        
        print("[PersonaPlex] Loading tokenizer...")
        text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
        
        # Check quantization compatibility
        if quantize != "none" and cpu_offload:
            print("[PersonaPlex] Warning: Quantization and CPU offload cannot be used together. Disabling CPU offload.")
            cpu_offload = False
        
        if quantize != "none" and device == "cpu":
            print("[PersonaPlex] Warning: Quantization requires CUDA. Falling back to no quantization.")
            quantize = "none"
        
        if quantize != "none" and not check_quantization_available(quantize):
            print(f"[PersonaPlex] Warning: {get_quantization_error()}")
            print("[PersonaPlex] Falling back to no quantization.")
            quantize = "none"
        
        print("[PersonaPlex] Loading Moshi LM...")
        if quantize != "none":
            # Load to CPU first, then quantize
            print(f"[PersonaPlex] Using {quantize} quantization to reduce VRAM usage...")
            lm = loaders.get_moshi_lm(moshi_path, device="cpu", cpu_offload=False)
            lm = quantize_model_after_load(lm, quantize, device)
        else:
            lm = loaders.get_moshi_lm(moshi_path, device=device, cpu_offload=cpu_offload)
        lm.eval()
        
        model_data = {
            "mimi": mimi,
            "other_mimi": other_mimi,
            "lm": lm,
            "tokenizer": text_tokenizer,
            "device": device,
            "voices_dir": voices_dir,
            "sample_rate": int(mimi.sample_rate),
            "frame_rate": mimi.frame_rate,
        }
        
        print("[PersonaPlex] Models loaded successfully!")
        return (model_data,)


class PersonaPlexInference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "personaplex_model": ("PERSONAPLEX_MODEL",),
                "audio": ("AUDIO",),
            },
            "optional": {
                "settings": ("PERSONAPLEX_SETTINGS",),
                "voice_audio": ("AUDIO", {
                    "tooltip": "Optional audio sample for zero-shot voice cloning (3-10 seconds recommended). If provided, overrides voice_preset."
                }),
                "voice_preset": (VOICE_PRESETS, {"default": "NATF2"}),
                "text_prompt": ("STRING", {
                    "default": "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.",
                    "multiline": True
                }),
            },
        }
    
    RETURN_TYPES = ("AUDIO", "STRING",)
    RETURN_NAMES = ("audio", "text_output",)
    FUNCTION = "inference"
    CATEGORY = "audio/PersonaPlex"

    def warmup(self, mimi, other_mimi, lm_gen, device, frame_size):
        for _ in range(4):
            chunk = torch.zeros(1, 1, frame_size, dtype=torch.float32, device=device)
            codes = mimi.encode(chunk)
            _ = other_mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = lm_gen.step(codes[:, :, c : c + 1])
                if tokens is None:
                    continue
                _ = mimi.decode(tokens[:, 1:9])
                _ = other_mimi.decode(tokens[:, 1:9])
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def decode_tokens_to_pcm(self, mimi, other_mimi, tokens):
        pcm = mimi.decode(tokens[:, 1:9])
        _ = other_mimi.decode(tokens[:, 1:9])
        pcm = pcm.detach().cpu().numpy()[0, 0]
        return pcm

    def inference(self, personaplex_model, audio, settings=None, voice_audio=None, voice_preset="NATF2", text_prompt=""):
        # Extract settings from optional input or use defaults
        if settings is not None:
            temp_audio = settings.get("temp_audio", 0.8)
            temp_text = settings.get("temp_text", 0.7)
            topk_audio = settings.get("topk_audio", 250)
            topk_text = settings.get("topk_text", 25)
            use_sampling = settings.get("use_sampling", True)
            silence_duration = settings.get("silence_duration", 0.5)
            seed = settings.get("seed", -1)
            # Override voice/prompt if set in settings
            if settings.get("voice_preset") and settings["voice_preset"] != "default":
                voice_preset = settings["voice_preset"]
            if settings.get("text_prompt"):
                text_prompt = settings["text_prompt"]
        else:
            temp_audio = 0.8
            temp_text = 0.7
            topk_audio = 250
            topk_text = 25
            use_sampling = True
            silence_duration = 0.5
            seed = -1
        
        # Use default prompt if none provided
        if not text_prompt:
            text_prompt = "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."
        
        if seed != -1:
            seed_all(seed)
        
        mimi = personaplex_model["mimi"]
        other_mimi = personaplex_model["other_mimi"]
        lm = personaplex_model["lm"]
        tokenizer = personaplex_model["tokenizer"]
        device = personaplex_model["device"]
        voices_dir = personaplex_model["voices_dir"]
        sample_rate = personaplex_model["sample_rate"]
        frame_rate = personaplex_model["frame_rate"]
        
        frame_size = int(sample_rate / frame_rate)
        
        # Handle voice cloning vs preset
        temp_voice_path = None
        if voice_audio is not None:
            # Zero-shot voice cloning from audio input
            print("[PersonaPlex] Using voice cloning from audio input...")
            voice_waveform = voice_audio["waveform"]
            voice_sample_rate = voice_audio["sample_rate"]
            
            # Convert to numpy and save as temp WAV
            voice_np = voice_waveform.squeeze().cpu().numpy()
            if voice_np.ndim > 1:
                voice_np = voice_np[0]  # Take first channel if stereo
            
            # Create temp file for voice prompt
            temp_voice_file = tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            )
            temp_voice_path = temp_voice_file.name
            temp_voice_file.close()
            
            # Save as WAV (scipy expects int16 or float32)
            wavfile.write(temp_voice_path, voice_sample_rate, voice_np.astype(np.float32))
            use_preset = False
        else:
            # Use preset .pt file
            voice_prompt_path = os.path.join(voices_dir, f"{voice_preset}.pt")
            if not os.path.exists(voice_prompt_path):
                raise FileNotFoundError(f"Voice preset not found: {voice_prompt_path}")
            use_preset = True
        
        lm_gen = LMGen(
            lm,
            audio_silence_frame_cnt=int(silence_duration * frame_rate),
            sample_rate=sample_rate,
            device=device,
            frame_rate=frame_rate,
            use_sampling=use_sampling,
            temp=temp_audio,
            temp_text=temp_text,
            top_k=topk_audio,
            top_k_text=topk_text,
        )
        
        mimi.streaming_forever(1)
        other_mimi.streaming_forever(1)
        lm_gen.streaming_forever(1)
        
        print("[PersonaPlex] Warming up...")
        self.warmup(mimi, other_mimi, lm_gen, device, frame_size)
        
        print("[PersonaPlex] Loading voice prompt...")
        if use_preset:
            lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
        else:
            lm_gen.load_voice_prompt(temp_voice_path)
            # Clean up temp file after loading
            os.unlink(temp_voice_path)
        lm_gen.text_prompt_tokens = tokenizer.encode(wrap_with_system_tags(text_prompt)) if text_prompt else None
        
        mimi.reset_streaming()
        other_mimi.reset_streaming()
        lm_gen.reset_streaming()
        lm_gen.step_system_prompts(mimi)
        mimi.reset_streaming()
        
        waveform = audio["waveform"]
        audio_sr = audio["sample_rate"]
        
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        user_audio = waveform.numpy()
        
        if audio_sr != sample_rate:
            user_audio = sphn.resample(user_audio, src_sample_rate=audio_sr, dst_sample_rate=sample_rate)
        
        generated_frames: List[np.ndarray] = []
        generated_text_tokens: List[str] = []
        total_target_samples = user_audio.shape[-1]
        
        print("[PersonaPlex] Running inference...")
        pbar = ProgressBar(total_target_samples // frame_size)
        step_count = 0
        
        with torch.no_grad():
            for user_encoded in lm_encode_from_sphn(
                mimi,
                lm_iterate_audio(user_audio, sample_interval_size=frame_size, pad=True),
                max_batch=1,
            ):
                steps = user_encoded.shape[-1]
                for c in range(steps):
                    step_in = user_encoded[:, :, c : c + 1]
                    tokens = lm_gen.step(step_in)
                    if tokens is None:
                        continue
                    pcm = self.decode_tokens_to_pcm(mimi, other_mimi, tokens)
                    generated_frames.append(pcm)
                    
                    text_token = tokens[0, 0, 0].item()
                    if text_token not in (0, 3):
                        _text = tokenizer.id_to_piece(text_token)
                        _text = _text.replace("▁", " ")
                        generated_text_tokens.append(_text)
                    else:
                        text_token_map = ['EPAD', 'BOS', 'EOS', 'PAD']
                        generated_text_tokens.append(text_token_map[text_token])
                    
                    step_count += 1
                    pbar.update(1)
        
        if len(generated_frames) == 0:
            raise RuntimeError("No audio frames were generated")
        
        output_pcm = np.concatenate(generated_frames, axis=-1)
        if output_pcm.shape[-1] > total_target_samples:
            output_pcm = output_pcm[:total_target_samples]
        elif output_pcm.shape[-1] < total_target_samples:
            pad_len = total_target_samples - output_pcm.shape[-1]
            output_pcm = np.concatenate([output_pcm, np.zeros(pad_len, dtype=output_pcm.dtype)], axis=-1)
        
        output_waveform = torch.from_numpy(output_pcm).unsqueeze(0).unsqueeze(0)
        output_audio = {"waveform": output_waveform, "sample_rate": sample_rate}
        
        text_output = "".join([t for t in generated_text_tokens if t not in ['EPAD', 'BOS', 'EOS', 'PAD']])
        
        print(f"[PersonaPlex] Generated {len(generated_frames)} frames")
        
        return (output_audio, text_output)


def _cleanup_server():
    """Cleanup server process on exit."""
    if PersonaPlexConversationServer._server_process is not None:
        try:
            print("[PersonaPlex] Shutting down conversation server...")
            PersonaPlexConversationServer._server_process.terminate()
            PersonaPlexConversationServer._server_process.wait(timeout=5)
        except Exception as e:
            print(f"[PersonaPlex] Error during cleanup: {e}")
        PersonaPlexConversationServer._server_process = None

atexit.register(_cleanup_server)


class PersonaPlexConversationServer:
    _server_process = None
    _server_url = None
    
    @classmethod
    def cleanup(cls):
        """Stop the running server."""
        _cleanup_server()
    
    @classmethod
    def INPUT_TYPES(cls):
        model_files = []
        if os.path.exists(PERSONAPLEX_MODELS_DIR):
            for f in os.listdir(PERSONAPLEX_MODELS_DIR):
                if f.endswith(".safetensors") and "model" in f.lower():
                    model_files.append(f)
        if not model_files:
            model_files = ["model.safetensors"]
        
        mimi_files = []
        if os.path.exists(PERSONAPLEX_MODELS_DIR):
            for f in os.listdir(PERSONAPLEX_MODELS_DIR):
                if f.endswith(".safetensors") and ("tokenizer" in f.lower() or "mimi" in f.lower()):
                    mimi_files.append(f)
        if not mimi_files:
            mimi_files = ["tokenizer-e351c8d8-checkpoint125.safetensors"]
        
        tokenizer_files = []
        if os.path.exists(PERSONAPLEX_MODELS_DIR):
            for f in os.listdir(PERSONAPLEX_MODELS_DIR):
                if f.endswith(".model"):
                    tokenizer_files.append(f)
        if not tokenizer_files:
            tokenizer_files = ["tokenizer_spm_32k_3.model"]
        
        # Build quantize options based on availability
        quantize_options = ["none"]
        if QUANTIZATION_AVAILABLE:
            quantize_options.extend(["8bit", "4bit"])
        
        return {
            "required": {
                "moshi_model": (model_files, {"default": model_files[0]}),
                "mimi_model": (mimi_files, {"default": mimi_files[0]}),
                "tokenizer": (tokenizer_files, {"default": tokenizer_files[0]}),
                "port": ("INT", {"default": 8998, "min": 1024, "max": 65535}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "cpu_offload": ("BOOLEAN", {"default": False}),
                "quantize": (quantize_options, {
                    "default": "none",
                    "tooltip": "Quantization level: none (~14GB), 8bit (~8GB), 4bit (~5GB). Requires bitsandbytes."
                }),
                "attention": (["auto", "sage", "sdpa"], {
                    "default": "auto",
                    "tooltip": "Attention backend: auto (use SageAttention if available), sage (force SageAttention), sdpa (PyTorch native)"
                }),
                "open_browser": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "settings": ("PERSONAPLEX_SETTINGS",),
                "external": ("PERSONAPLEX_EXTERNAL",),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("server_url",)
    FUNCTION = "start_server"
    CATEGORY = "audio/PersonaPlex"

    def start_server(self, moshi_model, mimi_model, tokenizer, port, device, cpu_offload, quantize, attention, open_browser, settings=None, external=None):
        if not PERSONAPLEX_AVAILABLE:
            raise RuntimeError(f"PersonaPlex not available: {IMPORT_ERROR}")
        
        import subprocess
        import webbrowser
        import threading
        import time
        
        # Extract settings
        host = "0.0.0.0"
        use_ssl = False
        gradio_tunnel = False
        seed = -1
        # Sampling settings
        temp_audio = 0.8
        temp_text = 0.7
        topk_audio = 250
        topk_text = 25
        use_sampling = True
        # Voice and prompt defaults for web UI
        default_voice = "NATF0.pt"
        default_text_prompt = "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."
        # Audio buffer defaults (client-side)
        init_buffer_ms = 400
        partial_buffer_ms = 210
        decoder_buffer_samples = 3840
        resample_quality = 5
        silence_delay_s = 0.07
        
        if settings is not None:
            host = settings.get("host", "0.0.0.0")
            use_ssl = settings.get("use_ssl", False)
            gradio_tunnel = settings.get("gradio_tunnel", False)
            seed = settings.get("seed", -1)
            # Sampling settings from Settings node
            temp_audio = settings.get("temp_audio", 0.8)
            temp_text = settings.get("temp_text", 0.7)
            topk_audio = settings.get("topk_audio", 250)
            topk_text = settings.get("topk_text", 25)
            use_sampling = settings.get("use_sampling", True)
            # Voice and text prompt defaults for web UI
            voice_preset = settings.get("voice_preset", "NATF2")
            if voice_preset and voice_preset != "default":
                default_voice = f"{voice_preset}.pt" if not voice_preset.endswith(".pt") else voice_preset
            text_prompt = settings.get("text_prompt", "")
            if text_prompt:
                default_text_prompt = text_prompt
            # Audio buffer settings (client-side)
            init_buffer_ms = settings.get("init_buffer_ms", 400)
            partial_buffer_ms = settings.get("partial_buffer_ms", 210)
            decoder_buffer_samples = settings.get("decoder_buffer_samples", 3840)
            resample_quality = settings.get("resample_quality", 5)
            silence_delay_s = settings.get("silence_delay_s", 0.07)
        
        moshi_path = os.path.join(PERSONAPLEX_MODELS_DIR, moshi_model)
        mimi_path = os.path.join(PERSONAPLEX_MODELS_DIR, mimi_model)
        tokenizer_path = os.path.join(PERSONAPLEX_MODELS_DIR, tokenizer)
        voices_dir = os.path.join(PERSONAPLEX_MODELS_DIR, "voices")
        
        for path, name in [(moshi_path, "Moshi model"), (mimi_path, "Mimi model"), (tokenizer_path, "Tokenizer")]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} not found: {path}")
        
        if not os.path.exists(voices_dir):
            voices_tgz = os.path.join(PERSONAPLEX_MODELS_DIR, "voices.tgz")
            if os.path.exists(voices_tgz):
                print(f"[PersonaPlex] Extracting voices to {voices_dir}")
                with tarfile.open(voices_tgz, "r:gz") as tar:
                    tar.extractall(path=PERSONAPLEX_MODELS_DIR)
            else:
                raise FileNotFoundError(f"Voices not found: {voices_dir} or {voices_tgz}")
        
        if PersonaPlexConversationServer._server_process is not None:
            try:
                PersonaPlexConversationServer._server_process.terminate()
                PersonaPlexConversationServer._server_process.wait(timeout=5)
            except:
                pass
            PersonaPlexConversationServer._server_process = None
        
        # Look for dist in node folder first, then models folder
        static_dir = os.path.join(os.path.dirname(__file__), "dist")
        if not os.path.exists(static_dir):
            static_dir = os.path.join(PERSONAPLEX_MODELS_DIR, "dist")
        if not os.path.exists(static_dir):
            static_dir = None
        
        # Build server arguments
        server_args = [
            "--moshi-weight", moshi_path,
            "--mimi-weight", mimi_path,
            "--tokenizer", tokenizer_path,
            "--voice-prompt-dir", voices_dir,
            "--port", str(port),
            "--device", device,
            "--host", host,
        ]
        if static_dir:
            server_args.extend(["--static", static_dir])
        
        if cpu_offload:
            server_args.append("--cpu-offload")
        
        # Sampling settings
        server_args.extend([
            "--temp-audio", str(temp_audio),
            "--temp-text", str(temp_text),
            "--topk-audio", str(topk_audio),
            "--topk-text", str(topk_text),
        ])
        if not use_sampling:
            server_args.append("--no-sampling")
        
        # Default voice and text prompt for web UI
        server_args.extend([
            "--default-voice", default_voice,
            "--default-text-prompt", default_text_prompt,
        ])
        
        # Audio buffer settings (client-side)
        server_args.extend([
            "--init-buffer-ms", str(int(init_buffer_ms)),
            "--partial-buffer-ms", str(int(partial_buffer_ms)),
            "--decoder-buffer-samples", str(int(decoder_buffer_samples)),
            "--resample-quality", str(int(resample_quality)),
            "--silence-delay-s", str(silence_delay_s),
        ])
        
        if use_ssl:
            server_args.extend(["--ssl", "mkcert"])
        
        if gradio_tunnel:
            server_args.append("--gradio-tunnel")
        
        # Determine which installation to use
        use_external = external is not None
        external_python = external.get("python_path", "") if external else ""
        external_path = external.get("install_path", "") if external else ""
        
        if use_external and external_python and external_path:
            # Use external PersonaPlex installation (like standalone)
            external_path = external_path.strip()
            external_python = external_python.strip()
            
            if not os.path.exists(external_python):
                raise FileNotFoundError(f"External Python not found: {external_python}")
            if not os.path.exists(external_path):
                raise FileNotFoundError(f"External PersonaPlex path not found: {external_path}")
            
            # Build simpler server args for external installation
            # Use the external installation's models and static files
            ext_models_dir = os.path.join(external_path, "models")
            ext_voices_dir = os.path.join(ext_models_dir, "voices")
            ext_static_dir = os.path.join(external_path, "client", "dist")
            
            # Use external paths for models if they exist, otherwise use our models
            if os.path.exists(ext_models_dir):
                ext_moshi = os.path.join(ext_models_dir, moshi_model)
                ext_mimi = os.path.join(ext_models_dir, mimi_model) 
                ext_tokenizer = os.path.join(ext_models_dir, tokenizer)
                # Fall back to our models if external doesn't have them
                if os.path.exists(ext_moshi):
                    moshi_path = ext_moshi
                if os.path.exists(ext_mimi):
                    mimi_path = ext_mimi
                if os.path.exists(ext_tokenizer):
                    tokenizer_path = ext_tokenizer
                if os.path.exists(ext_voices_dir):
                    voices_dir = ext_voices_dir
            
            # Rebuild server args with potentially updated paths
            server_args = [
                "--moshi-weight", moshi_path,
                "--mimi-weight", mimi_path,
                "--tokenizer", tokenizer_path,
                "--voice-prompt-dir", voices_dir,
                "--port", str(port),
                "--device", device,
                "--host", host,
            ]
            
            # Use external static if available, otherwise ours
            if os.path.exists(ext_static_dir):
                server_args.extend(["--static", ext_static_dir])
            elif static_dir:
                server_args.extend(["--static", static_dir])
            
            if cpu_offload:
                server_args.append("--cpu-offload")
            
            if use_ssl:
                # Use the same SSL path as standalone
                ssl_cache = os.path.expanduser("~/.cache/personaplex-ssl")
                server_args.extend(["--ssl", ssl_cache])
            
            if gradio_tunnel:
                server_args.append("--gradio-tunnel")
            
            # Simple run_server.py style command - matches standalone behavior exactly
            run_server_path = os.path.join(external_path, "run_server.py")
            if os.path.exists(run_server_path):
                # Use the external run_server.py directly
                cmd = [external_python, run_server_path] + server_args[6:]  # Skip model args, let it use defaults
                # Actually, pass all args to override
                cmd = [external_python, run_server_path, 
                       "--moshi-weight", moshi_path,
                       "--mimi-weight", mimi_path,
                       "--tokenizer", tokenizer_path,
                       "--voice-prompt-dir", voices_dir,
                       "--port", str(port),
                       "--host", host]
                if use_ssl:
                    cmd.extend(["--ssl", ssl_cache])
            else:
                # Fall back to running moshi.server directly
                ext_moshi_src = os.path.join(external_path, "moshi")
                cmd = [external_python, "-m", "moshi.server"] + server_args
            
            env = os.environ.copy()
            # Set PYTHONPATH to external moshi
            ext_moshi_src = os.path.join(external_path, "moshi")
            env["PYTHONPATH"] = ext_moshi_src + os.pathsep + env.get("PYTHONPATH", "")
            # Match standalone: disable torch.compile
            env["NO_TORCH_COMPILE"] = "1"
            
            print(f"[PersonaPlex] Using EXTERNAL installation: {external_path}")
            print(f"[PersonaPlex] Using EXTERNAL Python: {external_python}")
            working_dir = external_path
        else:
            # Use bundled PersonaPlex installation
            # Use -c to force our moshi path before any installed version
            bootstrap_code = f'''
import sys
sys.path.insert(0, r"{PERSONAPLEX_SRC}")
# Invalidate any cached moshi module so our version loads first
for key in list(sys.modules.keys()):
    if key == "moshi" or key.startswith("moshi."):
        del sys.modules[key]
sys.argv = ["moshi.server"] + {server_args!r}
import runpy
runpy.run_module("moshi.server", run_name="__main__", alter_sys=True)
'''
            cmd = [sys.executable, "-c", bootstrap_code]
            
            env = os.environ.copy()
            # Put bundled moshi FIRST in PYTHONPATH to ensure it's found before any installed version
            env["PYTHONPATH"] = PERSONAPLEX_SRC + os.pathsep + env.get("PYTHONPATH", "")
            
            # Match standalone behavior: disable torch.compile (PyTorch has dataclass bugs that cause issues)
            env["NO_TORCH_COMPILE"] = "1"
            env.pop("NO_CUDA_GRAPH", None)     # CUDA graphs are still useful
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Better memory allocation
            
            working_dir = PERSONAPLEX_SRC
        
        # Set attention backend (for bundled version)
        if attention == "sdpa":
            env["MOSHI_NO_SAGE_ATTENTION"] = "1"  # Force SDPA
            env["MOSHI_ATTENTION"] = "sdpa"
        elif attention == "sage":
            env.pop("MOSHI_NO_SAGE_ATTENTION", None)  # Allow SageAttention
            env["MOSHI_ATTENTION"] = "sage"
        else:  # auto
            env.pop("MOSHI_NO_SAGE_ATTENTION", None)  # Allow SageAttention if available
            env["MOSHI_ATTENTION"] = "auto"
        
        print(f"[PersonaPlex] Starting conversation server on port {port}...")
        print(f"[PersonaPlex] Model path: {moshi_path}")
        if use_external:
            print(f"[PersonaPlex] Mode: EXTERNAL installation")
        else:
            print(f"[PersonaPlex] Mode: Bundled (from {PERSONAPLEX_SRC})")
            print(f"[PersonaPlex] Attention backend: {attention}")
        print(f"[PersonaPlex] Command: {' '.join(cmd[:3])}...")  # Show first part of command
        
        # Log GPU memory state before starting subprocess
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved = torch.cuda.memory_reserved(i) / 1024**3
                    total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    print(f"[PersonaPlex] GPU {i}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {total:.1f}GB total")
        except Exception as e:
            print(f"[PersonaPlex] Could not get GPU memory info: {e}")
        
        # Use unbuffered output to reduce latency
        env["PYTHONUNBUFFERED"] = "1"
        
        PersonaPlexConversationServer._server_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=working_dir,
            bufsize=0,  # Unbuffered
        )
        
        startup_output = []
        
        def stream_output():
            proc = PersonaPlexConversationServer._server_process
            if proc and proc.stdout:
                import io
                # Use raw binary reading with immediate decoding for lower latency
                reader = io.TextIOWrapper(proc.stdout, encoding='utf-8', errors='replace', line_buffering=False)
                for line in reader:
                    line_text = line.rstrip()
                    startup_output.append(line_text)
                    print(f"[PersonaPlex Server] {line_text}")
        
        output_thread = threading.Thread(target=stream_output, daemon=True)
        output_thread.start()
        
        protocol = "https" if use_ssl else "http"
        server_url = f"{protocol}://localhost:{port}"
        PersonaPlexConversationServer._server_url = server_url
        
        print(f"[PersonaPlex] Waiting for server to start (loading models)...")
        
        # Poll until server is ready (up to 120 seconds for model loading)
        import urllib.request
        import urllib.error
        max_wait = 120
        poll_interval = 2
        elapsed = 0
        server_ready = False
        
        while elapsed < max_wait:
            proc = PersonaPlexConversationServer._server_process
            if proc.poll() is not None:
                # Process exited - collect error output
                time.sleep(0.5)
                error_details = "\n".join(startup_output[-20:]) if startup_output else "No output captured"
                raise RuntimeError(
                    f"Server process exited with code {proc.returncode}.\n"
                    f"Last output:\n{error_details}"
                )
            
            # Try to connect to the server
            try:
                req = urllib.request.Request(server_url, method='HEAD')
                urllib.request.urlopen(req, timeout=2)
                server_ready = True
                break
            except (urllib.error.URLError, OSError):
                # Server not ready yet
                pass
            
            time.sleep(poll_interval)
            elapsed += poll_interval
            if elapsed % 10 == 0:
                print(f"[PersonaPlex] Still loading models... ({elapsed}s)")
        
        if not server_ready:
            print(f"[PersonaPlex] Warning: Server may not be fully ready after {max_wait}s")
        else:
            print(f"[PersonaPlex] Server is ready!")
        
        if open_browser:
            print(f"[PersonaPlex] Opening browser to {server_url}")
            webbrowser.open(server_url)
        
        print(f"[PersonaPlex] Conversation server started at {server_url}")
        print("[PersonaPlex] Server output will appear in the ComfyUI console")
        
        return (server_url,)


class PersonaPlexStopServer:
    """Stops the running PersonaPlex conversation server."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "stop_server"
    CATEGORY = "audio/PersonaPlex"

    def stop_server(self):
        if PersonaPlexConversationServer._server_process is not None:
            try:
                PersonaPlexConversationServer._server_process.terminate()
                PersonaPlexConversationServer._server_process.wait(timeout=5)
                print("[PersonaPlex] Server stopped successfully")
                status = "Server stopped"
            except Exception as e:
                print(f"[PersonaPlex] Error stopping server: {e}")
                status = f"Error: {e}"
            PersonaPlexConversationServer._server_process = None
            PersonaPlexConversationServer._server_url = None
        else:
            status = "No server running"
            print("[PersonaPlex] No server is currently running")
        
        return (status,)
