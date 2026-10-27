#!/usr/bin/env python3
"""
Qwen3‑ASR‑MLX Batch CLI
从ComfyUI节点 Qwen3ASRMLXBatchFolder 提取独立命令行工具
批量处理多个分片音频目录，自动累加时间偏移，输出字幕 txt/srt/vtt/json/lrc
Only for Apple Silicon Mac (MLX backend)
新增：基于词级时间戳做字幕二次切分，中文标点优先，字符/时长双重上限保护
"""
import os
import re
import json
import tempfile
import argparse
from pathlib import Path
from soundfile import info
from mlx_audio.stt.utils import load_model
from mlx_audio.stt.generate import generate_transcription

# =============================================================================
# ⚠️ 开源使用者请修改此处默认模型路径
DEFAULT_MODEL_DIR = "/Users/felix/mlx/Qwen3-ASR-1.7B-bf16"
# =============================================================================

# ====================== 字幕后处理切分配置（大写常量可修改） ==================
SUBTITLE_MAX_CHARS = 35        # 单条字幕最大汉字字符数
SUBTITLE_MAX_DURATION_S = 6.0  # 单条字幕最大时长(秒)
SUBTITLE_SPLIT_PUNCT = "，。！？；"  # 优先切分的中文标点
# =============================================================================


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS,mmm format for SRT"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")


def format_vtt_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format for VTT"""
    return format_timestamp(seconds).replace(",", ".")


def format_lrc_timestamp(seconds: float) -> str:
    """LRC格式：mm:ss.xx"""
    mm = int(seconds // 60)
    ss = seconds % 60
    return f"{mm:02d}:{ss:05.2f}"


def save_as_lrc(all_segments, output_path: str, title: str = "", artist: str = "", album: str = "", lrc_by: str = ""):
    """增强版LRC生成：元标签 + segment内多行均匀分配时间戳"""
    lrc_lines = []
    if title.strip():
        lrc_lines.append(f"[ti:{title.strip()}]")
    if artist.strip():
        lrc_lines.append(f"[ar:{artist.strip()}]")
    if album.strip():
        lrc_lines.append(f"[al:{album.strip()}]")
    if lrc_by.strip():
        lrc_lines.append(f"[by:{lrc_by.strip()}]")
    if lrc_lines:
        lrc_lines.append("")
    for seg in all_segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_text = seg["text"].rstrip("%").strip()
        sub_lines = [line.strip() for line in seg_text.splitlines() if line.strip()]
        if not sub_lines:
            continue
        duration_total = seg_end - seg_start
        line_count = len(sub_lines)
        if line_count == 1:
            t_str = format_lrc_timestamp(seg_start)
            lrc_lines.append(f"[{t_str}]{sub_lines[0]}")
        else:
            step = duration_total / line_count
            for idx, sub_text in enumerate(sub_lines):
                line_time = seg_start + step * idx
                t_str = format_lrc_timestamp(line_time)
                lrc_lines.append(f"[{t_str}]{sub_text}")
    lrc_content = "\n".join(lrc_lines)
    with open(f"{output_path}.lrc", "w", encoding="utf-8") as f:
        f.write(lrc_content)


def save_as_txt_merged(all_segments, output_path: str):
    full_text = "\n".join(s["text"].strip() for s in all_segments)
    with open(f"{output_path}.txt", "w", encoding="utf-8") as f:
        f.write(full_text)


def save_as_srt_merged(all_segments, output_path: str):
    lines = []
    for idx, seg in enumerate(all_segments, start=1):
        s = format_timestamp(seg["start"])
        e = format_timestamp(seg["end"])
        lines.append(f"{idx}\n{s} --> {e}\n{seg['text'].strip()}\n")
    with open(f"{output_path}.srt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_as_vtt_merged(all_segments, output_path: str):
    lines = ["WEBVTT", ""]
    for idx, seg in enumerate(all_segments, start=1):
        s = format_vtt_timestamp(seg["start"])
        e = format_vtt_timestamp(seg["end"])
        lines.append(f"{idx}\n{s} --> {e}\n{seg['text'].strip()}\n")
    with open(f"{output_path}.vtt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_as_json_merged(all_segments, output_path: str):
    payload = {
        "segments": all_segments,
        "full_text": "\n".join(s["text"].strip() for s in all_segments)
    }
    with open(f"{output_path}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def split_segments_by_words(seg_items, max_chars: int, max_dur: float, puncts: str):
    """
    基于词级时间戳对单分片内原始segments做字幕切分
    规则：中文标点优先断开，再做字符数/时长上限强制截断
    返回：新的局部segment列表（未加offset），dict格式 {"start":float,"end":float,"text":str}
    防御：words为空则直接返回原始片段，不做切分
    """
    out = []
    punct_set = set(puncts)

    for seg in seg_items:
        words = None
        if isinstance(seg, dict):
            words = seg.get("words")
            seg_start = seg.get("start", 0.0)
            seg_end = seg.get("end", 0.0)
            seg_text = seg.get("text", "").strip()
        else:
            words = getattr(seg, "words", None)
            seg_start = getattr(seg, "start", 0.0)
            seg_end = getattr(seg, "end", 0.0)
            seg_text = getattr(seg, "text", "").strip()

        # 没有词级时间戳，降级，原样保留该段
        if not words or len(words) == 0:
            out.append({"start": seg_start, "end": seg_end, "text": seg_text})
            continue

        buf_words = []
        buf_text = []

        for w in words:
            w_text = w.get("text", "").strip()
            w_start = w.get("start")
            w_end = w.get("end")
            if not w_text:
                continue
            buf_words.append((w_start, w_end, w_text))
            buf_text.append(w_text)
            current_text = "".join(buf_text)
            current_len = len(current_text)
            current_dur = buf_words[-1][1] - buf_words[0][0]

            hit_punct = any(ch in punct_set for ch in w_text)
            need_split = hit_punct or (current_len >= max_chars) or (current_dur >= max_dur)

            if need_split:
                piece_text = "".join(buf_text).strip()
                piece_start = buf_words[0][0]
                piece_end = buf_words[-1][1]
                if piece_text:
                    out.append({"start": piece_start, "end": piece_end, "text": piece_text})
                buf_words.clear()
                buf_text.clear()

        # 处理缓冲区剩余部分
        if buf_words:
            piece_text = "".join(buf_text).strip()
            piece_start = buf_words[0][0]
            piece_end = buf_words[-1][1]
            if piece_text:
                out.append({"start": piece_start, "end": piece_end, "text": piece_text})
    return out


def process_single_folder(
    model,
    audio_folder: Path,
    file_prefix: str,
    out_format: str,
    language_hint: str,
    verbose: bool,
    lrc_title: str,
    lrc_artist: str,
    lrc_album: str,
    lrc_by: str,
    no_split_subtitle: bool
):
    """处理单个分片音频目录；返回输出文件Path"""
    print(f"\n{'='*60}")
    print(f"📂 处理目录: {audio_folder}")
    print(f"{'='*60}")
    if not audio_folder.is_dir():
        raise NotADirectoryError(f"目录不存在: {audio_folder}")
    escaped_pre = re.escape(file_prefix)
    pat = re.compile(rf"^{escaped_pre}_(\d+)\.(wav|mp3)$", re.IGNORECASE)
    matched = []
    for fname in os.listdir(audio_folder):
        m = pat.match(fname)
        if m:
            seq = int(m.group(1))
            matched.append((seq, fname))
    if len(matched) == 0:
        raise FileNotFoundError(f"目录内未找到匹配 `{file_prefix}_000.wav/mp3` 的分片音频")
    matched.sort(key=lambda x: x[0])
    file_list = [str(audio_folder / fn) for _, fn in matched]
    print(f"🔊 找到分片数量: {len(file_list)}")
    for fpath in file_list:
        print(f"   > {Path(fpath).name}")
    total_offset = 0.0
    all_merged_segments = []
    lang_arg = None if language_hint == "auto" else language_hint
    # 使用系统临时目录存放临时json，处理完自动清理
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, audio_file in enumerate(file_list):
            info_obj = info(audio_file)
            dur = info_obj.duration
            print(f"\n--分片 {Path(audio_file).name} | 时长={dur:.2f}s | offset={total_offset:.2f}s")
            temp_base = os.path.join(tmpdir, f"_temp_asr_part_{idx}")
            temp_json = f"{temp_base}.json"
            seg_obj = generate_transcription(
                model=model,
                audio=audio_file,
                output_path=temp_base,
                format="json",
                verbose=verbose,
                language=lang_arg,
                return_timestamps=True,  # 开启获取词级时间戳，用于字幕切分
            )
            seg_items = []
            if hasattr(seg_obj, "segments"):
                seg_items = seg_obj.segments
            elif hasattr(seg_obj, "sentences"):
                seg_items = seg_obj.sentences
            else:
                # fallback读取临时json
                with open(temp_json, "r", encoding="utf-8") as f:
                    jdata = json.load(f)
                seg_items = jdata.get("segments", [])

            if not no_split_subtitle:
                local_segs = split_segments_by_words(
                    seg_items,
                    max_chars=SUBTITLE_MAX_CHARS,
                    max_dur=SUBTITLE_MAX_DURATION_S,
                    puncts=SUBTITLE_SPLIT_PUNCT
                )
            else:
                # 不切分，原样转换为dict
                local_segs = []
                for s in seg_items:
                    d = {}
                    if isinstance(s, dict):
                        d["start"] = s.get("start", 0.0)
                        d["end"] = s.get("end", 0.0)
                        d["text"] = s.get("text", "").strip()
                    else:
                        d["start"] = getattr(s, "start", 0.0)
                        d["end"] = getattr(s, "end", 0.0)
                        d["text"] = getattr(s, "text", "").strip()
                    if d["text"]:
                        local_segs.append(d)

            # 叠加分片全局时间偏移，加入总集合
            for d in local_segs:
                d["start"] = d["start"] + total_offset
                d["end"] = d["end"] + total_offset
                if d["text"]:
                    all_merged_segments.append(d)

            total_offset += dur
    print(f"\n✅分片全部识别完成；合并总时长 {total_offset:.2f}s；有效字幕段：{len(all_merged_segments)}")
    # 输出路径推导：输入目录C → 输出到C.parent，文件名=C.name
    out_parent = audio_folder.parent
    out_stem = audio_folder.name
    out_base = str(out_parent / out_stem)
    if out_format == "txt":
        save_as_txt_merged(all_merged_segments, out_base)
    elif out_format == "srt":
        save_as_srt_merged(all_merged_segments, out_base)
    elif out_format == "vtt":
        save_as_vtt_merged(all_segments, out_base)
    elif out_format == "json":
        save_as_json_merged(all_merged_segments, out_base)
    elif out_format == "lrc":
        save_as_lrc(
            all_merged_segments,
            out_base,
            title=lrc_title,
            artist=lrc_artist,
            album=lrc_album,
            lrc_by=lrc_by
        )
    else:
        raise ValueError(f"不支持输出格式 {out_format}")
    final_out = Path(f"{out_base}.{out_format}")
    print(f"📄输出字幕文件: {final_out.resolve()}")
    return final_out


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3‑ASR‑MLX Batch CLI，批量处理TTS分片音频目录，合并输出字幕，支持词级时间戳自动切分字幕",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # 位置参数：多个音频目录
    parser.add_argument(
        "audio_dirs",
        nargs="+",
        help="一个或多个分片音频目录(存放tts_000.wav等)"
    )
    parser.add_argument(
        "-m", "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Qwen3‑ASR‑MLX模型本地目录"
    )
    parser.add_argument(
        "-p", "--file-prefix",
        default="tts",
        help="分片文件名前缀，匹配 {prefix}_000.wav / mp3"
    )
    parser.add_argument(
        "-f", "--out-format",
        choices=["txt", "srt", "vtt", "json", "lrc"],
        default="srt",
        help="输出字幕格式"
    )
    parser.add_argument(
        "-l", "--language-hint",
        choices=["auto", "zh", "en", "ja", "ko", "ru", "fr"],
        default="zh",
        help="语言提示，中文TTS建议保持zh"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="打印详细日志"
    )
    parser.add_argument(
        "--no-verbose",
        dest="verbose",
        action="store_false",
        help="关闭详细日志输出"
    )
    parser.add_argument(
        "--no-split-subtitle",
        action="store_true",
        default=False,
        help="关闭自动字幕切分，输出ASR原始segment，用于对比调试"
    )
    # LRC元标签参数
    parser.add_argument("--lrc-title", default="", help="LRC ti 标题标签，仅format=lrc生效")
    parser.add_argument("--lrc-artist", default="", help="LRC ar 艺术家标签，仅format=lrc生效")
    parser.add_argument("--lrc-album", default="", help="LRC al 专辑标签，仅format=lrc生效")
    parser.add_argument("--lrc-by", default="Qwen3‑ASR‑MLX", help="LRC by生成标识，仅format=lrc生效")

    args = parser.parse_args()
    # 路径展开 ~
    audio_dir_list = [Path(p).expanduser() for p in args.audio_dirs]
    print(f"🚀 Qwen3‑ASR‑MLX Batch CLI")
    if not args.no_split_subtitle:
        print(f"✂️ 字幕自动切分开启：max_chars={SUBTITLE_MAX_CHARS}, max_dur={SUBTITLE_MAX_DURATION_S}s")
    else:
        print(f"⚠️ 字幕自动切分已关闭，输出原始ASR segments")
    print(f"🔧 加载模型: {args.model_dir}")
    # 模型只加载一次，多个目录复用
    model = load_model(args.model_dir)
    print(f"✅模型加载完成")
    success_count = 0
    fail_count = 0
    for folder in audio_dir_list:
        try:
            process_single_folder(
                model=model,
                audio_folder=folder,
                file_prefix=args.file_prefix,
                out_format=args.out_format,
                language_hint=args.language_hint,
                verbose=args.verbose,
                lrc_title=args.lrc_title,
                lrc_artist=args.lrc_artist,
                lrc_album=args.lrc_album,
                lrc_by=args.lrc_by,
                no_split_subtitle=args.no_split_subtitle
            )
            success_count += 1
        except Exception as e:
            print(f"\n❌处理目录 [{folder}] 失败: {str(e)}")
            fail_count += 1
            continue
    print(f"\n{'='*60}")
    print(f"🏁全部任务结束。成功:{success_count} 失败:{fail_count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
