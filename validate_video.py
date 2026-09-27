"""Validate an output video and create input/output frame comparisons."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from inspect_video import inspect_video


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def read_all_frames(path: Path) -> tuple[int, list[np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Không thể mở video: {path}")
    count = 0
    samples: list[np.ndarray] = []
    sample_indices: set[int] = set()
    reported_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if reported_count > 0:
        sample_indices = {0, reported_count // 2, reported_count - 1}
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if count in sample_indices:
                samples.append(frame.copy())
            count += 1
    finally:
        capture.release()
    return count, samples


def black_ratio(frame: np.ndarray) -> float:
    return float(np.mean(np.max(frame, axis=2) < 8))


def create_comparison(
    input_path: Path, output_path: Path, comparison_path: Path
) -> None:
    input_capture = cv2.VideoCapture(str(input_path))
    output_capture = cv2.VideoCapture(str(output_path))
    if not input_capture.isOpened() or not output_capture.isOpened():
        raise RuntimeError("Không thể mở video để tạo ảnh so sánh.")

    input_count = int(input_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_count = int(output_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sorted({0, input_count // 2, max(0, input_count - 1)})
    panels: list[np.ndarray] = []
    try:
        for index in indices:
            input_capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            output_capture.set(cv2.CAP_PROP_POS_FRAMES, min(index, max(0, output_count - 1)))
            input_ok, input_frame = input_capture.read()
            output_ok, output_frame = output_capture.read()
            if not input_ok or not output_ok:
                continue
            if input_frame.shape[:2] != output_frame.shape[:2]:
                output_frame = cv2.resize(
                    output_frame,
                    (input_frame.shape[1], input_frame.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
            label_input = input_frame.copy()
            label_output = output_frame.copy()
            cv2.putText(label_input, f"input frame {index}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(label_output, f"output frame {index}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            panels.append(np.hstack((label_input, label_output)))
    finally:
        input_capture.release()
        output_capture.release()

    if not panels:
        raise RuntimeError("Không tạo được frame so sánh.")
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(comparison_path), np.vstack(panels))


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiểm tra video output AnimeGANv2")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--comparison", type=Path)
    args = parser.parse_args()

    if not args.input.is_file() or not args.output.is_file():
        raise SystemExit("Input và output phải tồn tại trước khi validate.")

    input_metadata = inspect_video(args.input)
    output_metadata = inspect_video(args.output)
    input_count, _ = read_all_frames(args.input)
    output_count, output_samples = read_all_frames(args.output)

    output_black_ratio = max(
        (black_ratio(frame) for frame in output_samples), default=1.0
    )
    report = {
        "input": input_metadata,
        "output": output_metadata,
        "sequential_read": {
            "input_frames_read": input_count,
            "output_frames_read": output_count,
            "frame_count_difference": output_count - input_count,
        },
        "checks": {
            "input_opened": True,
            "output_opened": True,
            "same_resolution": (
                input_metadata["width"] == output_metadata["width"]
                and input_metadata["height"] == output_metadata["height"]
            ),
            "fps_difference": abs(input_metadata["fps"] - output_metadata["fps"]),
            "output_frame_count_matches_input": output_count == input_count,
            "output_not_mostly_black": output_black_ratio < 0.98,
        },
        "observations": [
            "Cần xem ảnh comparison và video thực tế để kết luận màu sắc, méo hình và phong cách.",
            "Flicker do suy luận độc lập từng frame là hiện tượng cần ghi nhận riêng.",
        ],
    }
    if args.comparison:
        create_comparison(args.input, args.output, args.comparison)
        report["comparison_image"] = str(args.comparison)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Đã ghi report: {args.report}")

    failed_checks = [
        name
        for name, passed in report["checks"].items()
        if name in {"same_resolution", "output_not_mostly_black"} and not passed
    ]
    if failed_checks:
        raise SystemExit(f"FAIL: {', '.join(failed_checks)}")


if __name__ == "__main__":
    main()

