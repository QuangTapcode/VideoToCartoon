"""Read and save basic video metadata with OpenCV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def fourcc_to_text(value: float) -> str:
    code = int(value)
    chars = [chr((code >> (8 * index)) & 0xFF) for index in range(4)]
    text = "".join(chars).replace("\x00", "")
    return text if text.strip() else "????"


def inspect_video(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Không thể mở video: {path}")

    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        codec = fourcc_to_text(capture.get(cv2.CAP_PROP_FOURCC))
    finally:
        capture.release()

    if width <= 0 or height <= 0:
        raise RuntimeError("Video không có kích thước hợp lệ.")
    duration = frame_count / fps if fps > 0 and frame_count > 0 else None
    return {
        "path": str(path),
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "fps": fps,
        "frame_count_reported_by_opencv": frame_count,
        "duration_seconds_estimated": duration,
        "codec_reported_by_opencv": codec,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Đọc metadata video bằng OpenCV")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Không tìm thấy input: {args.input}")
    metadata = inspect_video(args.input)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Đã ghi report: {args.report}")


if __name__ == "__main__":
    main()

