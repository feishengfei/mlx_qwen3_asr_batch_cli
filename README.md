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


```
### 下载模型
下载 Qwen3‑ASR‑1.7B‑bf16 MLX权重，默认路径写在脚本头部常量：
```python
DEFAULT_MODEL_DIR = "/Users/felix/mlx/Qwen3-ASR-1.7B-bf16"
```
> ⚠️开源使用者可修改该常量为自己的模型存放路径，或者运行时通过 `-m` 参数指定模型目录。

## 音频目录格式
每个待处理目录内部存放顺序分片音频，命名格式：
```
东晋_王羲之_丧乱贴/
├─ tts_000.wav
├─ tts_001.wav
├─ tts_002.wav
└─ ...
```

输出规则：
输入目录：`~/Downloads/TODO/东晋_王羲之_丧乱贴/东晋_王羲之_丧乱贴`
输出字幕生成在**输入目录的父目录下**，文件名与目录名保持一致。
> 输出示例：`~/Downloads/TODO/东晋_王羲之_丧乱贴/东晋_王羲之_丧乱贴.srt`

## 使用示例

### 最简调用（单个目录，默认输出srt，language‑hint=zh）
```bash
python qwen3_asr_batch_cli.py ~/Downloads/TODO/东晋_王羲之_丧乱贴/东晋_王羲之_丧乱贴
```

### 一次性批量处理多个目录
```bash
python qwen3_asr_batch_cli.py dir1 dir2 dir3
```

### 输出 LRC歌词字幕，附带元信息
```bash
python qwen3_asr_batch_cli.py ~/Downloads/TODO/东晋_王羲之_丧乱贴/东晋_王羲之_丧乱贴 \
  -f lrc \
  --lrc-title "丧乱帖" \
  --lrc-artist "王羲之"
```

### 指定自定义模型目录，关闭verbose日志
```bash
python qwen3_asr_batch_cli.py my_audio_dir \
  -m /path/to/your/Qwen3-ASR-1.7B-bf16 \
  -f json \
  --no-verbose
```

## 参数参考
```
positional arguments:
  audio_dirs            一个或多个分片音频目录(存放tts_000.wav等)

options:
  -h, --help            show this help message and exit
  -m MODEL_DIR, --model-dir MODEL_DIR
                        Qwen3‑ASR‑MLX模型本地目录 (default: /Users/felix/mlx/Qwen3-ASR-1.7B-bf16)
  -p FILE_PREFIX, --file-prefix FILE_PREFIX
                        分片文件名前缀，匹配 {prefix}_000.wav / mp3 (default: tts)
  -f {txt,srt,vtt,json,lrc}, --out-format {txt,srt,vtt,json,lrc}
                        输出字幕格式 (default: srt)
  -l {auto,zh,en,ja,ko,ru,fr}, --language-hint {auto,zh,en,ja,ko,ru,fr}
                        语言提示，中文TTS建议保持zh (default: zh)
  --verbose             打印详细日志 (default: True)
  --no-verbose          关闭详细日志输出

LRC标签参数（仅输出格式为lrc时生效）:
  --lrc-title LRC_TITLE
                        LRC ti 标题标签
  --lrc-artist LRC_ARTIST
                        LRC ar 艺术家标签
  --lrc-album LRC_ALBUM
                        LRC al 专辑标签
  --lrc-by LRC_BY       LRC by生成标识 (default: Qwen3‑ASR‑MLX)
```

## 特性
1. **模型只加载一次**：多目录批量处理时不会重复加载权重，节省内存与等待时间。
2. 临时识别JSON写入系统临时目录，任务结束自动清理，不污染业务文件夹。
3. 目录处理失败不会整体崩溃，跳过错误目录继续执行下一个任务，统计成功/失败计数。
4. 完整复刻原ComfyUI节点时间偏移逻辑，分片拼接后时间轴连续正确。
5. LRC支持头部元标签；segment内部多行文本会在时间段内均匀分配每行时间戳。

## 注意事项
1. 仅支持 Apple Silicon Mac，MLX库不支持Intel / Windows / Linux。
2. 分片文件命名必须遵守 `prefix_数字.wav/mp3`，依靠数字序号排序。
3. 输入路径支持 `~` 用户家目录简写
   

# Wrapper 包装脚本 mlx_qwen3_asr_batch_cli_wrapper.py
提供递归扫描、自动跳过已生成字幕、强制重跑能力，适合批量遍历整个TODO文件夹。

## Wrapper 参数
- `-r / --recursive`：开启递归扫描，从传入根目录向下查找所有包含 *.wav 的父目录（分片目录）
- `-f / --force`：强制重跑，**不跳过已存在的字幕文件**，覆盖输出

> ⚠️ 注意区分：
> wrapper 的 `--out-format` 是字幕格式（srt/lrc等）
> wrapper 的 `-f` 是 force 强制标记，二者不再冲突

## 典型调用
```bash
# 递归扫描TODO目录，已经有srt的自动跳过
mlx-asr-batch -r ~/Downloads/TODO/

# 递归扫描 + 强制全部重生成lrc
mlx-asr-batch -r -f --out-format lrc ~/Downloads/TODO/


# TIPS for zsh

```shell
alias mlx-asr-batch='python ~/bin/mlx_qwen3_asr_batch_cli/mlx_qwen3_asr_batch_cli_wrapper.py'
```

