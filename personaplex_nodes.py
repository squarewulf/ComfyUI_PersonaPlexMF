import os
import sys
import json
import tarfile
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import folder_paths
from comfy.utils import ProgressBar

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


def ensure_voice_prompts(voices_dir: str, auto_download: bool):
    if os.path.exists(voices_dir):
        if any(name.endswith(".pt") for name in os.listdir(voices_dir)):
            return

    voices_tgz = os.path.join(PERSONAPLEX_MODELS_DIR, "voices.tgz")
    if os.path.exists(voices_tgz):
        print(f"[PersonaPlex] Extracting voices to {voices_dir}")
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=PERSONAPLEX_MODELS_DIR)
    elif auto_download:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise RuntimeError(
                "huggingface-hub is required to auto-download voices. "
                "Install it with: pip install huggingface-hub"
            ) from e
        try:
            voices_tgz = hf_hub_download("nvidia/personaplex-7b-v1", "voices.tgz")
        except Exception as e:
            raise RuntimeError(
                "Failed to download voices.tgz. Make sure you accepted the model "
                "license on HuggingFace and are logged in."
            ) from e
        print(f"[PersonaPlex] Downloaded voices.tgz, extracting to {voices_dir}")
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=PERSONAPLEX_MODELS_DIR)
    else:
        raise FileNotFoundError(
            f"Voices not found. Place voices.tgz or voices/ folder in {PERSONAPLEX_MODELS_DIR}"
        )

    if not os.path.exists(voices_dir) or not any(name.endswith(".pt") for name in os.listdir(voices_dir)):
        raise FileNotFoundError(f"Voice prompts not found in {voices_dir}")


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
                        voice_preset="default", text_prompt="", host="0.0.0.0",
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
            # Server
            "host": host,
            "use_ssl": use_ssl,
            "gradio_tunnel": gradio_tunnel,
        }
        return (settings,)


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
        
        return {
            "required": {
                "moshi_model": (model_files, {"default": model_files[0] if model_files else "model.safetensors"}),
                "mimi_model": (mimi_files, {"default": mimi_files[0] if mimi_files else "tokenizer-e351c8d8-checkpoint125.safetensors"}),
                "tokenizer": (tokenizer_files, {"default": tokenizer_files[0] if tokenizer_files else "tokenizer_spm_32k_3.model"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "cpu_offload": ("BOOLEAN", {"default": False}),
                "auto_download_voices": ("BOOLEAN", {"default": True}),
            },
        }
    
    RETURN_TYPES = ("PERSONAPLEX_MODEL",)
    RETURN_NAMES = ("personaplex_model",)
    FUNCTION = "load_model"
    CATEGORY = "audio/PersonaPlex"

    def load_model(self, moshi_model, mimi_model, tokenizer, device, cpu_offload, auto_download_voices=True):
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
        ensure_voice_prompts(voices_dir, auto_download_voices)
        
        print("[PersonaPlex] Loading Mimi encoder/decoder...")
        mimi = loaders.get_mimi(mimi_path, device)
        other_mimi = loaders.get_mimi(mimi_path, device)
        
        print("[PersonaPlex] Loading tokenizer...")
        text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
        
        print("[PersonaPlex] Loading Moshi LM...")
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

    def inference(self, personaplex_model, audio, settings=None, voice_preset="NATF2", text_prompt=""):
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
        
        voice_prompt_path = os.path.join(voices_dir, f"{voice_preset}.pt")
        if not os.path.exists(voice_prompt_path):
            raise FileNotFoundError(f"Voice preset not found: {voice_prompt_path}")
        
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
        lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
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
        
        user_audio = waveform.detach().cpu().numpy()
        
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


class PersonaPlexConversationServer:
    _server_process = None
    _server_url = None
    
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
        
        return {
            "required": {
                "moshi_model": (model_files, {"default": model_files[0]}),
                "mimi_model": (mimi_files, {"default": mimi_files[0]}),
                "tokenizer": (tokenizer_files, {"default": tokenizer_files[0]}),
                "port": ("INT", {"default": 8998, "min": 1024, "max": 65535}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "cpu_offload": ("BOOLEAN", {"default": False}),
                "open_browser": ("BOOLEAN", {"default": True}),
                "auto_download_voices": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "settings": ("PERSONAPLEX_SETTINGS",),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("server_url",)
    FUNCTION = "start_server"
    CATEGORY = "audio/PersonaPlex"

    def start_server(self, moshi_model, mimi_model, tokenizer, port, device, cpu_offload, open_browser, auto_download_voices, settings=None):
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
        
        if settings is not None:
            host = settings.get("host", "0.0.0.0")
            use_ssl = settings.get("use_ssl", False)
            gradio_tunnel = settings.get("gradio_tunnel", False)
            seed = settings.get("seed", -1)
        
        moshi_path = os.path.join(PERSONAPLEX_MODELS_DIR, moshi_model)
        mimi_path = os.path.join(PERSONAPLEX_MODELS_DIR, mimi_model)
        tokenizer_path = os.path.join(PERSONAPLEX_MODELS_DIR, tokenizer)
        voices_dir = os.path.join(PERSONAPLEX_MODELS_DIR, "voices")
        
        for path, name in [(moshi_path, "Moshi model"), (mimi_path, "Mimi model"), (tokenizer_path, "Tokenizer")]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} not found: {path}")
        
        ensure_voice_prompts(voices_dir, auto_download_voices)
        
        if PersonaPlexConversationServer._server_process is not None:
            try:
                PersonaPlexConversationServer._server_process.terminate()
                PersonaPlexConversationServer._server_process.wait(timeout=5)
            except:
                pass
            PersonaPlexConversationServer._server_process = None
        
        # Look for dist in: client build output, then node folder, then models folder
        static_dir = os.path.join(PERSONAPLEX_SRC, "client", "dist")
        if not os.path.exists(static_dir):
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
        
        if use_ssl:
            server_args.extend(["--ssl", "mkcert"])
        
        if gradio_tunnel:
            server_args.append("--gradio-tunnel")
        
        # Create a wrapper script that forces our bundled moshi to load
        # This is needed because Windows embedded Python ignores PYTHONPATH in some cases
        # IMPORTANT: sys.argv must be set BEFORE importing moshi.server because
        # the module calls main() at import time via: with torch.no_grad(): main()
        wrapper_code = f'''
import sys
# Set argv FIRST - server.py calls main() at module level during import
sys.argv = ["server.py"] + {server_args!r}
# Remove any cached moshi imports
for key in list(sys.modules.keys()):
    if key == "moshi" or key.startswith("moshi."):
        del sys.modules[key]
# Force our bundled path to be first
sys.path.insert(0, r"{PERSONAPLEX_SRC}")
# Import triggers main() automatically via module-level code
import moshi.server
'''
        
        cmd = [sys.executable, "-c", wrapper_code]
        
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["NO_TORCH_COMPILE"] = "1"
        
        print(f"[PersonaPlex] Starting conversation server on port {port}...")
        print(f"[PersonaPlex] Model path: {moshi_path}")
        print(f"[PersonaPlex] Static dir: {static_dir}")
        print(f"[PersonaPlex] Using bundled moshi from: {PERSONAPLEX_SRC}")
        print(f"[PersonaPlex] Server args: {' '.join(server_args)}")
        
        PersonaPlexConversationServer._server_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=PERSONAPLEX_SRC,
            bufsize=1,
            universal_newlines=True,
        )
        
        def stream_output():
            proc = PersonaPlexConversationServer._server_process
            if proc and proc.stdout:
                for line in proc.stdout:
                    print(f"[PersonaPlex Server] {line.rstrip()}")
        
        output_thread = threading.Thread(target=stream_output, daemon=True)
        output_thread.start()
        
        protocol = "https" if use_ssl else "http"
        server_url = f"{protocol}://localhost:{port}"
        PersonaPlexConversationServer._server_url = server_url
        
        print(f"[PersonaPlex] Waiting for server to be ready (loading models, warming up)...")
        
        # Poll the server until it responds, instead of a fixed sleep.
        # The server loads models, extracts voices, warms up, then starts listening.
        import urllib.request
        import ssl
        
        max_wait = 600  # 10 minutes max (models can be large)
        poll_interval = 3  # seconds between checks
        waited = 0
        server_ready = False
        
        while waited < max_wait:
            proc = PersonaPlexConversationServer._server_process
            if proc.poll() is not None:
                raise RuntimeError(
                    f"Server process exited with code {proc.returncode}. "
                    f"Check the ComfyUI console for error details."
                )
            
            try:
                ctx = None
                if use_ssl:
                    ctx = ssl.create_default_context()
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                req = urllib.request.Request(server_url, method='GET')
                resp = urllib.request.urlopen(req, timeout=2, context=ctx)
                if resp.status == 200:
                    server_ready = True
                    break
            except Exception:
                pass  # Server not ready yet
            
            time.sleep(poll_interval)
            waited += poll_interval
            if waited % 15 == 0:
                print(f"[PersonaPlex] Still waiting for server... ({waited}s elapsed)")
        
        if not server_ready:
            proc = PersonaPlexConversationServer._server_process
            if proc.poll() is not None:
                raise RuntimeError(
                    f"Server process exited with code {proc.returncode} during startup."
                )
            else:
                print(f"[PersonaPlex] Warning: Server did not respond within {max_wait}s, but process is still running. Opening anyway.")
        else:
            print(f"[PersonaPlex] Server is ready! (took ~{waited}s)")
        
        if open_browser:
            print(f"[PersonaPlex] Opening browser to {server_url}")
            webbrowser.open(server_url)
        
        print(f"[PersonaPlex] Conversation server running at {server_url}")
        print("[PersonaPlex] Server output will appear in the ComfyUI console")
        
        return (server_url,)


class PersonaPlexServerURL:
    """Node that receives a PersonaPlex server URL and can display/open it."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server_url": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "open_browser": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("url",)
    FUNCTION = "process_url"
    CATEGORY = "PersonaPlex"
    OUTPUT_NODE = True
    
    def process_url(self, server_url: str, open_browser: bool = False):
        print(f"[PersonaPlex] Server URL: {server_url}")
        
        if open_browser:
            import webbrowser
            import urllib.request
            import ssl
            import time
            
            # Wait for the server to actually be ready before opening the browser
            print(f"[PersonaPlex] Checking if server is ready at {server_url}...")
            max_wait = 300
            poll_interval = 3
            waited = 0
            ready = False
            
            while waited < max_wait:
                try:
                    ctx = None
                    if server_url.startswith("https"):
                        ctx = ssl.create_default_context()
                        ctx.check_hostname = False
                        ctx.verify_mode = ssl.CERT_NONE
                    req = urllib.request.Request(server_url, method='GET')
                    resp = urllib.request.urlopen(req, timeout=2, context=ctx)
                    if resp.status == 200:
                        ready = True
                        break
                except Exception:
                    pass
                time.sleep(poll_interval)
                waited += poll_interval
            
            if ready:
                print(f"[PersonaPlex] Server ready, opening browser to {server_url}")
            else:
                print(f"[PersonaPlex] Server not confirmed ready after {max_wait}s, opening browser anyway")
            
            webbrowser.open(server_url)
        
        return {"ui": {"text": [server_url]}, "result": (server_url,)}
