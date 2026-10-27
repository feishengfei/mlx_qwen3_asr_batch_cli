# qwen3‑asr‑mlx‑batch‑cli

> 从 ComfyUI `Qwen3ASRMLXBatchFolder` 节点剥离的独立命令行工具，Apple Silicon Mac MLX 后端。
批量识别TTS分片音频（`tts_000.wav` / `tts_001.wav` …），自动累加时间偏移，合并输出完整字幕。
支持输出：`txt / srt / vtt / json / lrc`。

## 环境要求
- Apple Silicon Mac（M1/M2/M3/M4），MLX框架
- Python >=3.10

### 安装依赖
```bash
pip install soundfile mlx-audio

