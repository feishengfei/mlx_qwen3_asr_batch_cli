#!/usr/bin/env python3
"""
Qwen3‑ASR‑MLX Single File CLI
单音频文件听写工具，仅支持 wav / mp3，输出字幕 txt/srt/vtt/json/lrc
Only for Apple Silicon Mac (MLX backend)
复用 mlx_qwen3_asr_batch_cli 的切分逻辑、保存函数、常量
⚠️不支持视频/m4a/flac等；格式转换请用户外部完成
"""
import argparse
import tempfile
from pathlib import Path
from soundfile import info
from mlx_audio.stt.utils import load_model
from mlx_audio.stt.generate import generate_transcription

# 从batch cli导入：常量、切分函数、各个格式保存函数、时间格式化工具
from mlx_qwen3_asr_batch_cli import (
    DEFAULT_MODEL_DIR,
    SUBTITLE_MAX_CHARS,
    SUBTITLE_MAX_DURATION_S,
    SUBTITLE_SPLIT_PUNCT,
    SUBTITLE_SPLIT_PUNCT_EN,
    TRIM_PUNCT_EXCEPT,
    split_by_text_ratio_interpolate,
    format_timestamp,
    format_vtt_timestamp,
    format_lrc_timestamp,
    save_as_lrc,
    save_as_txt_merged,
    save_as_srt_merged,
    save_as_vtt_merged,
    save_as_json_merged,
)


def process_single_file(
    model,
    audio_file: Path,
    out_format: str,
    language_hint: str,
    verbose: bool,
    lrc_title: str,
    lrc_artist: str,
    lrc_album: str,
    lrc_by: str,
    no_split_subtitle: bool
) -> Path:
    """处理单个wav/mp3音频文件，返回输出字幕Path"""
    print(f"\n{'='*60}")
    print(f"🎵 处理音频文件: {audio_file}")
    print(f"{'='*60}")

    if not audio_file.is_file():
        raise FileNotFoundError(f"音频文件不存在: {audio_file}")

    suffix = audio_file.suffix.lower()
    if suffix not in (".wav", ".mp3"):
        raise RuntimeError(f"不支持文件格式 {suffix}，仅支持 .wav / .mp3，请外部转码后重试。")

    info_obj = info(str(audio_file))
    dur = info_obj.duration
    print(f"⏱️ 音频总时长 = {dur:.2f}s")

    lang_arg = None if language_hint == "auto" else language_hint
    all_segments = []

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_base = Path(tmpdir) / "_temp_asr_single"
        temp_json = f"{temp_base}.json"

        seg_obj = generate_transcription(
            model=model,
            audio=str(audio_file),
            output_path=str(temp_base),
            format="json",
            verbose=verbose,
            language=lang_arg,
            return_timestamps=True,
        )

        seg_items = []
        if hasattr(seg_obj, "segments"):
            seg_items = seg_obj.segments
        elif hasattr(seg_obj, "sentences"):
            seg_items = seg_obj.sentences
        else:
            # fallback读取临时json
            import json
            with open(temp_json, "r", encoding="utf-8") as f:
                jdata = json.load(f)
            seg_items = jdata.get("segments", [])

        if not no_split_subtitle:
            all_punct = SUBTITLE_SPLIT_PUNCT + SUBTITLE_SPLIT_PUNCT_EN
            local_segs = split_by_text_ratio_interpolate(
                seg_items,
                max_chars=SUBTITLE_MAX_CHARS,
                max_dur=SUBTITLE_MAX_DURATION_S,
                puncts=all_punct
            )
        else:
            # 关闭切分，原样转为dict
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

        # 单文件无分片偏移，直接使用
        all_segments = local_segs

    print(f"\n✅识别完成；有效字幕段：{len(all_segments)}")

    # 输出策略A：同目录，使用输入文件stem
    out_stem = audio_file.stem
    out_parent = audio_file.parent
    out_base = str(out_parent / out_stem)

    if out_format == "txt":
        save_as_txt_merged(all_segments, out_base)
    elif out_format == "srt":
        save_as_srt_merged(all_segments, out_base)
    elif out_format == "vtt":
        save_as_vtt_merged(all_segments, out_base)
    elif out_format == "json":
        save_as_json_merged(all_segments, out_base)
    elif out_format == "lrc":
        save_as_lrc(
            all_segments,
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
        description="Qwen3‑ASR‑MLX Single File CLI：听写单个wav/mp3音频文件",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "audio_file",
        help="输入单个音频文件，仅支持 wav / mp3"
    )
    parser.add_argument(
        "-m", "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Qwen3‑ASR‑MLX模型本地目录"
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
        help="语言提示"
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
    audio_path = Path(args.audio_file).expanduser().resolve()

    print(f"🚀 Qwen3‑ASR‑MLX Single‑File CLI")
    if not args.no_split_subtitle:
        print(f"✂️ 字幕自动切分开启：max_chars={SUBTITLE_MAX_CHARS}, max_dur={SUBTITLE_MAX_DURATION_S}s")
    else:
        print(f"⚠️ 字幕自动切分已关闭，输出原始ASR segments")

    print(f"🔧 加载模型: {args.model_dir}")
    model = load_model(args.model_dir)
    print(f"✅模型加载完成")

    try:
        process_single_file(
            model=model,
            audio_file=audio_path,
            out_format=args.out_format,
            language_hint=args.language_hint,
            verbose=args.verbose,
            lrc_title=args.lrc_title,
            lrc_artist=args.lrc_artist,
            lrc_album=args.lrc_album,
            lrc_by=args.lrc_by,
            no_split_subtitle=args.no_split_subtitle
        )
        print(f"\n{'='*60}")
        print(f"🏁任务执行成功")
        print(f"{'='*60}")
    except Exception as e:
        print(f"\n❌处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
