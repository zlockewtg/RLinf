#!/usr/bin/env python3
"""Concatenate numbered video segments such as 0_0.mp4, 0_1.mp4, ..."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_INPUT_DIR = Path(
    "/mnt/public/tgy/repo/RLinf/logs/"
    "20260514-06:44:26-behavior_ppo_openpi05_eval_skill_chain/"
    "video/eval/seed_1"
)


def natural_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", path.stem)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def collect_segments(input_dir: Path, prefix: str, output: Path) -> list[Path]:
    output = output.resolve()
    segments = [
        path
        for path in input_dir.glob(f"{prefix}*.mp4")
        if path.is_file() and path.resolve() != output
    ]
    return sorted(segments, key=natural_key)


def ffmpeg_escape(path: Path) -> str:
    return str(path).replace("'", r"'\''")


def concat_with_ffmpeg(segments: list[Path], output: Path, reencode: bool) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as file_list:
        list_path = Path(file_list.name)
        for segment in segments:
            file_list.write(f"file '{ffmpeg_escape(segment.resolve())}'\n")

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
        ]
        if reencode:
            cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"]
        else:
            cmd += ["-c", "copy"]
        cmd.append(str(output))
        subprocess.run(cmd, check=True)
    finally:
        list_path.unlink(missing_ok=True)


def concat_with_opencv(segments: list[Path], output: Path, fps_override: float | None) -> None:
    import cv2

    first = cv2.VideoCapture(str(segments[0]))
    if not first.isOpened():
        raise RuntimeError(f"Cannot open first video: {segments[0]}")

    width = int(first.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = fps_override or float(first.get(cv2.CAP_PROP_FPS)) or 30.0
    first.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output}")

    total_frames = 0
    try:
        for segment in segments:
            cap = cv2.VideoCapture(str(segment))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {segment}")

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(frame)
                total_frames += 1
            cap.release()
    finally:
        writer.release()

    if total_frames == 0:
        output.unlink(missing_ok=True)
        raise RuntimeError("No frames were written to the output video.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate mp4 segments with the same prefix in natural numeric order."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing video segments. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--prefix",
        default="0_",
        help="Only concatenate files whose names start with this prefix. Default: 0_",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output mp4 path. Default: <input-dir>/<prefix without trailing _>_merged.mp4",
    )
    parser.add_argument(
        "--method",
        choices=("auto", "ffmpeg", "opencv"),
        default="auto",
        help="Concatenation backend. Default: auto",
    )
    parser.add_argument(
        "--reencode",
        action="store_true",
        help="When using ffmpeg, re-encode instead of stream-copying.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="FPS override for the OpenCV backend.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    output = args.output
    if output is None:
        output_name = f"{args.prefix.rstrip('_') or 'merged'}_merged.mp4"
        output = input_dir / output_name
    else:
        output = output.expanduser()
        if not output.is_absolute():
            output = input_dir / output
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    segments = collect_segments(input_dir, args.prefix, output)
    if not segments:
        print(f"No mp4 segments found in {input_dir} with prefix {args.prefix!r}.", file=sys.stderr)
        return 1

    print("Segments:")
    for segment in segments:
        print(f"  {segment.name}")
    print(f"Output: {output}")

    ffmpeg_path = shutil.which("ffmpeg")
    try:
        if args.method == "ffmpeg" or (args.method == "auto" and ffmpeg_path):
            concat_with_ffmpeg(segments, output, args.reencode)
        elif args.method == "auto":
            print("ffmpeg not found; falling back to OpenCV.")
            concat_with_opencv(segments, output, args.fps)
        else:
            concat_with_opencv(segments, output, args.fps)
    except subprocess.CalledProcessError as exc:
        if args.method == "auto" and not args.reencode:
            print("ffmpeg stream copy failed; retrying with OpenCV.")
            concat_with_opencv(segments, output, args.fps)
        else:
            print(f"Video concat failed: {exc}", file=sys.stderr)
            return 1
    except Exception as exc:
        print(f"Video concat failed: {exc}", file=sys.stderr)
        return 1

    size_mb = os.path.getsize(output) / (1024 * 1024)
    print(f"Done: {output} ({size_mb:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
