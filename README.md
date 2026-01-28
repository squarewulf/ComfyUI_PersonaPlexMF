# ComfyUI_PersonaPlexMF

A ComfyUI custom node for running [NVIDIA PersonaPlex](https://github.com/NVIDIA/personaplex) - a real-time, full-duplex speech-to-speech conversational AI with persona control.

[![GitHub](https://img.shields.io/badge/GitHub-squarewulf-blue?logo=github)](https://github.com/squarewulf/ComfyUI_PersonaPlexMF)

## Features

- **Live Conversation** — Real-time voice chat with AI using your microphone
- **Batch Inference** — Process audio files through the PersonaPlex model
- **Voice Presets** — 18 different voice options (natural and variety styles)
- **Persona Control** — Customize AI behavior with text prompts
- **Full Settings Control** — Adjust sampling, temperature, and timing parameters

## Screenshots

### PersonaPlex Conversation Server Node
Start the live conversation server with model selection and options.

![PersonaPlex Conversation Server](images/Server-Node.png)

### PersonaPlex Settings Node (Optional)
Configure all sampling, temperature, and timing parameters in one place.

![PersonaPlex Settings](images/Settings-Node.png)

### Web UI - Live Conversation
Real-time voice conversation with customizable personas and voice selection.

![Web UI](images/Web-UI.png)

## Installation

### 1. Clone or Download

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/squarewulf/ComfyUI_PersonaPlexMF.git
```

### 2. Install Dependencies

```bash
cd ComfyUI/custom_nodes/ComfyUI_PersonaPlexMF
pip install -r requirements.txt
pip install -e personaplex_src/moshi
```

### 3. Download Models

Download from [HuggingFace nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1) and place in `ComfyUI/models/personaplex/`:

| File | Description |
|------|-------------|
| `model.safetensors` | Moshi 7B LM weights (~14GB) |
| `tokenizer-e351c8d8-checkpoint125.safetensors` | Mimi encoder/decoder |
| `tokenizer_spm_32k_3.model` | Text tokenizer |
| `voices.tgz` | Voice presets (auto-extracted) |

The web UI for live conversation is included in this repo (no separate download needed).

## Nodes

### PersonaPlex Conversation Server

Starts a WebSocket server with a web UI for live voice conversation.

**Inputs:**
| Input | Type | Description |
|-------|------|-------------|
| `moshi_model` | dropdown | Moshi LM weights file |
| `mimi_model` | dropdown | Mimi encoder/decoder file |
| `tokenizer` | dropdown | Text tokenizer file |
| `port` | int | Server port (default: 8998) |
| `device` | dropdown | cuda or cpu |
| `cpu_offload` | bool | Enable for low VRAM GPUs |
| `open_browser` | bool | Auto-open browser |
| `settings` | optional | Connect PersonaPlex Settings node |

**Output:** `server_url` — URL to access the conversation UI

### PersonaPlex Model Loader

Loads PersonaPlex models for batch inference.

**Inputs:**
| Input | Type | Description |
|-------|------|-------------|
| `moshi_model` | dropdown | Moshi LM weights file |
| `mimi_model` | dropdown | Mimi encoder/decoder file |
| `tokenizer` | dropdown | Text tokenizer file |
| `device` | dropdown | cuda or cpu |
| `cpu_offload` | bool | Enable for low VRAM GPUs |

**Output:** `personaplex_model` — Model bundle for inference

### PersonaPlex Inference

Runs speech-to-speech inference on audio input.

**Inputs:**
| Input | Type | Description |
|-------|------|-------------|
| `personaplex_model` | model | From Model Loader |
| `audio` | AUDIO | Input audio |
| `settings` | optional | Connect PersonaPlex Settings node |
| `voice_preset` | dropdown | Voice to use |
| `text_prompt` | string | Persona/behavior prompt |

**Outputs:** 
- `audio` — Generated speech response
- `text_output` — Transcription of response

### PersonaPlex Settings

Configure all inference parameters. Connect to Inference or Server nodes.

**Sampling Settings:**
| Setting | Default | Description |
|---------|---------|-------------|
| `temp_audio` | 0.8 | Audio sampling temperature |
| `temp_text` | 0.7 | Text sampling temperature |
| `topk_audio` | 250 | Top-K for audio tokens |
| `topk_text` | 25 | Top-K for text tokens |
| `use_sampling` | true | Use sampling vs greedy |

**Timing Settings:**
| Setting | Default | Description |
|---------|---------|-------------|
| `silence_duration` | 0.5 | Seconds before response starts |

**Other Settings:**
| Setting | Default | Description |
|---------|---------|-------------|
| `seed` | -1 | Random seed (-1 = random) |
| `voice_preset` | default | Voice override |
| `text_prompt` | "" | Persona prompt override |
| `host` | 0.0.0.0 | Server bind address |
| `use_ssl` | false | Enable HTTPS |
| `gradio_tunnel` | false | Enable remote access |

## Voice Presets

| Category | Voices |
|----------|--------|
| Natural Female | NATF0, NATF1, NATF2, NATF3 |
| Natural Male | NATM0, NATM1, NATM2, NATM3 |
| Variety Female | VARF0, VARF1, VARF2, VARF3, VARF4 |
| Variety Male | VARM0, VARM1, VARM2, VARM3, VARM4 |

## Example Prompts

**Assistant:**
```
You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.
```

**Customer Service:**
```
You work for AeroRentals Pro which is a drone rental company and your name is Alex.
```

**Casual Conversation:**
```
You enjoy having a good conversation. Have a casual discussion about technology.
```

## Hardware Requirements

| Configuration | VRAM | Notes |
|--------------|------|-------|
| Recommended | 24GB+ | Full speed inference |
| Minimum | 12GB | Use `cpu_offload=True` (slower) |

## Troubleshooting

**"Torch not compiled with CUDA"**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**"No module named 'sphn'"**
```bash
pip install sphn sentencepiece aiohttp
```

**Server won't start**
- Check ComfyUI console for error messages
- Ensure all model files are in `ComfyUI/models/personaplex/`

## License

- PersonaPlex model weights: [NVIDIA Open Model License](https://huggingface.co/nvidia/personaplex-7b-v1)
- This ComfyUI wrapper: MIT License

## Author

[squarewulf](https://github.com/squarewulf)
