#!/usr/bin/env python3
"""
Wrapper for mlx_qwen3_asr_batch_cli.py
新增递归扫描、自动跳过已存在字幕、force强制重跑、dry-run预演
"""
import argparse
from pathlib import Path
from typing import Set, List

# 导入原版cli里的处理函数、常量
from mlx_qwen3_asr_batch_cli import (
    DEFAULT_MODEL_DIR,
    process_single_folder,
    load_model
)


def collect_wav_parent_dirs(root: Path) -> Set[Path]:
    """
    递归遍历root，找到所有包含wav文件的父目录
    等价于 find root -name '*.wav' 取dirname，自动去重
    """
    parent_dirs = set()
    for wav_path in root.rglob("*.wav"):
        parent_dirs.add(wav_path.parent.resolve())
    return parent_dirs


def should_skip(folder: Path, out_format: str) -> bool:
    """
    判断是否跳过：输出字幕文件已经存在就返回True（跳过）
    out_path = folder.parent / f"{folder.name}.{out_format}"
    """
    out_path = folder.parent / f"{folder.name}.{out_format}"
    return out_path.exists()


def main():
    parser = argparse.ArgumentParser(
        description="Wrapper：递归扫描目录 + 自动跳过已存在字幕，调用Qwen3ASR批量识别",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # wrapper专属参数
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="开启递归扫描，从root目录向下搜索所有包含wav的分片目录"
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="强制模式：不跳过已有字幕文件，全部重新生成"
    )
    parser.add_argument(
        "-n", "--dry-run",
        action="store_true",
        help="预演模式：只列出待处理/跳过清单，不加载模型、不运行ASR、不写入文件"
    )
    parser.add_argument(
        "scan_root",
        nargs="+",
        help="扫描根目录；-r开启后才会递归搜索wav父目录；不使用-r则等价原版，直接作为目录列表传入"
    )
    parser.add_argument(
        "--no-split-subtitle",
        action="store_true",
        default=False,
        help="关闭自动字幕切分，输出ASR原始segment，用于对比调试"
    )

    # 透传给底层ASR的参数（和原版cli对齐）
    parser.add_argument("-m", "--model-dir", default=DEFAULT_MODEL_DIR, help="Qwen3-ASR MLX模型目录")
    parser.add_argument("-p", "--file-prefix", default="tts", help="分片音频前缀 tts_000.wav")
    parser.add_argument("--out-format", choices=["txt", "srt", "vtt", "json", "lrc"], default="srt", help="输出字幕格式")
    parser.add_argument("-l", "--language-hint", choices=["auto", "zh", "en", "ja", "ko", "ru", "fr"], default="zh")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--no-verbose", dest="verbose", action="store_false")
    parser.add_argument("--lrc-title", default="")
    parser.add_argument("--lrc-artist", default="")
    parser.add_argument("--lrc-album", default="")
    parser.add_argument("--lrc-by", default="Qwen3‑ASR‑MLX")

    args = parser.parse_args()

    # 收集待处理目录列表
    target_folders: List[Path] = []
    for raw_root in args.scan_root:
        scan_root = Path(raw_root).expanduser().resolve()
        if not scan_root.exists():
            print(f"⚠️ 扫描根目录不存在，跳过：{scan_root}")
            continue

        if args.recursive:
            print(f"🔍 递归扫描: {scan_root}")
            dir_set = collect_wav_parent_dirs(scan_root)
            dir_list = sorted(list(dir_set))
            print(f"✅ 找到包含wav的分片目录总数：{len(dir_list)}")
            target_folders.extend(dir_list)
        else:
            # 不开启递归，直接把传入路径当成分片目录（原版行为）
            target_folders.append(scan_root)

    # 去重（防止多次扫描命中同一个目录）
    target_folders = list({p.resolve() for p in target_folders})
    target_folders.sort()
    if not target_folders:
        print("❌ 没有找到任何待处理分片目录，退出")
        return

    print(f"\n📋 扫描得到分片目录清单（共 {len(target_folders)} 个）")
    for d in target_folders:
        print(f"   - {d}")

    # dry-run：只做预判，不加载模型、不跑识别
    if args.dry_run:
        print("\n🧪 ====== DRY-RUN 预演模式，不会执行识别和写文件 ======")
        dry_skip = 0
        dry_todo = 0
        out_fmt = args.out_format
        for folder in target_folders:
            out_file = folder.parent / f"{folder.name}.{out_fmt}"
            if (not args.force) and should_skip(folder, out_fmt):
                print(f"⏭️ [DRY] 跳过 {folder.name} → {out_file} 已存在")
                dry_skip += 1
            else:
                print(f"▶️  [DRY] 将处理 {folder.name} → {out_file}")
                dry_todo += 1
        print("\n" + "="*70)
        print(f"📊 预演汇总：待处理={dry_todo} | 将跳过={dry_skip}")
        print("="*70)
        return

    # 正常运行：加载模型一次，全局复用
    print(f"\n🚀 加载模型：{args.model_dir}")
    model = load_model(args.model_dir)
    print("✅ 模型加载完成\n")

    success_count = 0
    skip_count = 0
    fail_count = 0
    out_fmt = args.out_format

    for folder in target_folders:
        out_file = folder.parent / f"{folder.name}.{out_fmt}"

        if (not args.force) and should_skip(folder, out_fmt):
            print(f"⏭️ 跳过 [{folder.name}]：字幕文件已存在 -> {out_file}")
            skip_count += 1
            continue

        try:
            print(f"\n▶️ 开始处理：{folder}")
            process_single_folder(
                model=model,
                audio_folder=folder,
                file_prefix=args.file_prefix,
                out_format=out_fmt,
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
            print(f"❌ 处理失败 [{folder}]：{str(e)}")
            fail_count += 1

    print("\n" + "="*70)
    print(f"🏁 汇总：成功={success_count} | 跳过={skip_count} | 失败={fail_count}")
    print("="*70)


if __name__ == "__main__":
    main()
