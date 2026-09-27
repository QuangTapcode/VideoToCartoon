"""Convert a video frame-by-frame with an AnimeGANv2 PyTorch checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from inspect_video import inspect_video
from models.animegan2 import Generator


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Đã yêu cầu CUDA nhưng PyTorch không thấy GPU CUDA.")
    return torch.device(requested)


def load_model(model_path: Path, device: torch.device) -> Generator:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {model_path}. "
            "Tải một checkpoint AnimeGANv2 .pt và đặt vào thư mục models."
        )

    model = Generator()
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "generator" in checkpoint:
        state_dict = checkpoint["generator"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise RuntimeError("Checkpoint không phải state_dict PyTorch hợp lệ.")

    cleaned_state_dict = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint không khớp kiến trúc AnimeGANv2. "
            f"missing={list(missing)[:5]}, unexpected={list(unexpected)[:5]}"
        )
    model.to(device).eval()
    return model


def nearest_multiple(value: float, multiple: int = 32) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def resize_for_model(frame: np.ndarray, load_size: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = load_size / float(max(height, width))
    model_width = nearest_multiple(width * scale)
    model_height = nearest_multiple(height * scale)
    if (model_width, model_height) == (width, height):
        return frame
    return cv2.resize(frame, (model_width, model_height), interpolation=cv2.INTER_AREA)


def frames_to_tensor(
    frames_bgr: list[np.ndarray], device: torch.device
) -> torch.Tensor:
    """Convert an ordered list of OpenCV BGR frames to an AnimeGANv2 batch."""
    if not frames_bgr:
        raise ValueError("Need at least one frame to create a tensor batch.")

    frames_rgb = np.stack(
        [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr], axis=0
    )
    values = np.ascontiguousarray(frames_rgb.astype(np.float32) / 255.0)
    tensor = torch.from_numpy(values).permute(0, 3, 1, 2)
    return (tensor * 2.0 - 1.0).to(device, non_blocking=device.type == "cuda")


def frame_to_tensor(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Backward-compatible single-frame wrapper around the batch conversion."""
    return frames_to_tensor([frame_bgr], device)


def tensor_to_frames(
    output: torch.Tensor, original_size: tuple[int, int]
) -> list[np.ndarray]:
    if isinstance(output, (tuple, list)):
        output = output[0]
    if output.ndim != 4:
        raise RuntimeError(f"AnimeGANv2 output must be BCHW, got {tuple(output.shape)}")

    values = output.detach().float().clamp(-1.0, 1.0)
    values = ((values + 1.0) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()
    width, height = original_size
    frames_bgr = []
    for frame_rgb in values:
        frame_bgr = cv2.cvtColor(np.ascontiguousarray(frame_rgb), cv2.COLOR_RGB2BGR)
        if frame_bgr.shape[1] != width or frame_bgr.shape[0] != height:
            frame_bgr = cv2.resize(
                frame_bgr, (width, height), interpolation=cv2.INTER_LANCZOS4
            )
        frames_bgr.append(frame_bgr)
    return frames_bgr


def tensor_to_frame(output: torch.Tensor, original_size: tuple[int, int]) -> np.ndarray:
    """Backward-compatible single-frame wrapper around the batch conversion."""
    return tensor_to_frames(output, original_size)[0]


def get_device_info(device: torch.device) -> dict:
    gpu_name = None
    if device.type == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(device)
    return {
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "gpu_name": gpu_name,
    }


def synchronize_device(device: torch.device) -> None:
    """Make GPU timing include queued CUDA kernels when CUDA is active."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def find_ffmpeg() -> str | None:
    """Find FFmpeg in PATH or in the standard Windows WinGet package folder."""
    executable = shutil.which("ffmpeg")
    if executable:
        return executable

    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        return None
    packages_dir = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
    candidates = sorted(packages_dir.glob("Gyan.FFmpeg.Shared*/*/bin/ffmpeg.exe"))
    return str(candidates[0]) if candidates else None


def _temporary_sibling(path: Path, suffix: str | None = None) -> Path:
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{path.stem}_", suffix=suffix or path.suffix or ".mp4", dir=str(path.parent)
    )
    os.close(file_descriptor)
    os.unlink(temporary_name)
    return Path(temporary_name)


def _run_ffmpeg(command: list[str], action: str) -> None:
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.CalledProcessError as error:
        details = (error.stderr or "").strip()
        if len(details) > 1200:
            details = details[-1200:]
        suffix = f"\nChi tiết FFmpeg: {details}" if details else ""
        raise RuntimeError(f"FFmpeg không thể {action}.{suffix}") from error


def transcode_video_for_browser(source: Path, output: Path) -> None:
    """Encode video as H.264 so Chrome/Streamlit can play the preview."""
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError(
            "Không tìm thấy FFmpeg để tạo preview H.264. "
            "Cài FFmpeg hoặc bỏ tùy chọn browser_compatible."
        )

    temporary_output = _temporary_sibling(output)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(temporary_output),
    ]
    try:
        _run_ffmpeg(command, "mã hóa preview H.264")
        os.replace(temporary_output, output)
    finally:
        if temporary_output.exists():
            temporary_output.unlink()


def mux_audio(
    video_without_audio: Path,
    audio_source: Path,
    output: Path,
    browser_compatible: bool = False,
) -> None:
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError(
            "Không tìm thấy FFmpeg trong PATH. Bỏ --audio-from hoặc cài FFmpeg trước."
        )
    video_codec = (
        [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
        ]
        if browser_compatible
        else ["-c:v", "copy"]
    )
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_without_audio),
        "-i",
        str(audio_source),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        *video_codec,
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output),
    ]
    _run_ffmpeg(command, "ghép audio vào video")


def process_video(args: argparse.Namespace) -> dict:
    input_path: Path = args.input
    output_path: Path = args.output
    report_path: Path = args.report or output_path.with_suffix(".report.json")
    device = choose_device(args.device)

    input_metadata = inspect_video(input_path)
    if input_metadata["fps"] <= 0:
        raise RuntimeError("Video không có FPS hợp lệ.")
    if args.load_size < 32:
        raise ValueError("--load-size phải từ 32 trở lên.")

    batch_size = int(getattr(args, "batch_size", 1))
    if batch_size <= 0:
        raise ValueError("--batch-size phải lớn hơn 0.")
    max_frames = int(getattr(args, "max_frames", 0))
    if max_frames < 0:
        raise ValueError("--max-frames không được âm.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model_load_started = time.perf_counter()
    model = load_model(args.model, device)
    model_load_seconds = time.perf_counter() - model_load_started

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Không thể mở video: {input_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    writer_path = output_path
    temporary_video: Path | None = None
    if args.audio_from:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f"{output_path.stem}_no_audio_",
            suffix=output_path.suffix or ".mp4",
            dir=str(output_path.parent),
        )
        os.close(file_descriptor)
        os.unlink(temporary_name)
        temporary_video = Path(temporary_name)
        writer_path = temporary_video

    writer = cv2.VideoWriter(
        str(writer_path),
        cv2.VideoWriter_fourcc(*args.codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        if temporary_video and temporary_video.exists():
            temporary_video.unlink()
        raise RuntimeError(
            f"Không thể mở VideoWriter với codec {args.codec!r}. "
            "Thử --codec mp4v hoặc codec được backend OpenCV hỗ trợ."
        )

    processed_frames = 0
    model_seconds = 0.0
    read_seconds = 0.0
    preprocess_seconds = 0.0
    postprocess_seconds = 0.0
    write_seconds = 0.0
    batch_count = 0
    started = time.perf_counter()
    progress_callback = getattr(args, "progress_callback", None)
    reported_frame_count = input_metadata["frame_count_reported_by_opencv"]
    progress_total = (
        min(reported_frame_count, max_frames) if max_frames else reported_frame_count
    )
    try:
        with torch.inference_mode():
            while True:
                model_frames = []
                reached_eof = False
                read_started = time.perf_counter()
                while len(model_frames) < batch_size:
                    if max_frames and processed_frames + len(model_frames) >= max_frames:
                        break
                    ok, frame = capture.read()
                    if not ok:
                        reached_eof = True
                        break
                    model_frames.append(resize_for_model(frame, args.load_size))
                read_seconds += time.perf_counter() - read_started

                if not model_frames:
                    break

                preprocess_started = time.perf_counter()
                tensor = frames_to_tensor(model_frames, device)
                preprocess_seconds += time.perf_counter() - preprocess_started
                synchronize_device(device)
                inference_started = time.perf_counter()
                output = model(tensor, align_corners=args.align_corners)
                synchronize_device(device)
                model_seconds += time.perf_counter() - inference_started
                postprocess_started = time.perf_counter()
                cartoon_frames = tensor_to_frames(output, (width, height))
                postprocess_seconds += time.perf_counter() - postprocess_started
                if len(cartoon_frames) != len(model_frames):
                    raise RuntimeError(
                        "Số frame output của AnimeGANv2 không khớp batch input."
                    )
                batch_count += 1
                write_started = time.perf_counter()
                for cartoon_frame in cartoon_frames:
                    writer.write(cartoon_frame)
                    processed_frames += 1
                    if progress_callback:
                        progress_callback(processed_frames, progress_total)
                write_seconds += time.perf_counter() - write_started

                if processed_frames % args.log_every == 0 or len(model_frames) < batch_size:
                    print(f"Đã xử lý {processed_frames} frame")
                if reached_eof or (max_frames and processed_frames >= max_frames):
                    break
    finally:
        capture.release()
        writer.release()

    if processed_frames == 0:
        if temporary_video and temporary_video.exists():
            temporary_video.unlink()
        raise RuntimeError("Không đọc được frame nào từ video đầu vào.")

    browser_compatible = bool(getattr(args, "browser_compatible", False))
    if temporary_video:
        try:
            mux_audio(
                temporary_video,
                args.audio_from,
                output_path,
                browser_compatible=browser_compatible,
            )
        finally:
            if temporary_video.exists():
                temporary_video.unlink()
    elif browser_compatible:
        transcode_video_for_browser(output_path, output_path)

    output_metadata = inspect_video(output_path)
    elapsed_seconds = time.perf_counter() - started
    report = {
        "input": input_metadata,
        "output": output_metadata,
        "model": {
            "checkpoint": str(args.model),
            "style": args.model.stem,
            "device": str(device),
            "load_size": args.load_size,
            "model_load_seconds": model_load_seconds,
            "device_info": get_device_info(device),
        },
        "processing": {
            "processed_frames": processed_frames,
            "batch_size": batch_size,
            "batch_count": batch_count,
            "processing_seconds": elapsed_seconds,
            "model_inference_seconds": model_seconds,
            "read_seconds": read_seconds,
            "preprocess_seconds": preprocess_seconds,
            "postprocess_seconds": postprocess_seconds,
            "write_seconds": write_seconds,
            "seconds_per_frame": elapsed_seconds / processed_frames,
            "audio_source": str(args.audio_from) if args.audio_from else None,
            "codec_requested": args.codec,
            "browser_compatible_h264": browser_compatible,
            "frame_limit": max_frames,
        },
        "checks": {
            "same_resolution": (
                input_metadata["width"] == output_metadata["width"]
                and input_metadata["height"] == output_metadata["height"]
            ),
            "fps_difference": abs(input_metadata["fps"] - output_metadata["fps"]),
            "frame_count_difference_from_reported_input": (
                output_metadata["frame_count_reported_by_opencv"]
                - processed_frames
            ),
        },
        "notes": [
            "AnimeGANv2 xử lý các frame theo batch nhưng vẫn ghi theo đúng thứ tự; cần xem video để đánh giá flicker.",
            "Batch size chỉ tăng hiệu suất suy luận; không thay đổi FPS, số frame hay tốc độ phát video.",
            "Output OpenCV không tự giữ audio; audio chỉ có nếu --audio-from và FFmpeg thành công.",
            "Browser preview dùng H.264/yuv420p khi browser_compatible được bật; mp4v có thể không phát trực tiếp trên một số trình duyệt.",
        ],
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Đã ghi report: {report_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chuyển video thành video hoạt hình bằng AnimeGANv2 + PyTorch"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--load-size", type=int, default=512)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Số frame suy luận cùng lúc; GPU thường nhanh hơn nhưng tốn VRAM hơn.",
    )
    parser.add_argument(
        "--align-corners",
        action="store_true",
        help="Dùng bilinear upsample align_corners=True; mặc định False theo checkpoint PyTorch.",
    )
    parser.add_argument("--codec", default="mp4v", help="4 ký tự codec OpenCV, mặc định mp4v")
    parser.add_argument("--audio-from", type=Path)
    parser.add_argument(
        "--browser-compatible",
        action="store_true",
        help="Mã hóa output H.264/yuv420p để trình duyệt phát trực tiếp; cần FFmpeg.",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--max-frames", type=int, default=0, help="0 = xử lý toàn bộ video")
    parser.add_argument("--log-every", type=int, default=30)
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Không tìm thấy input: {args.input}")
    if args.audio_from and not args.audio_from.is_file():
        parser.error(f"Không tìm thấy audio source: {args.audio_from}")
    if args.log_every <= 0:
        parser.error("--log-every phải lớn hơn 0")
    if args.max_frames < 0:
        parser.error("--max-frames không được âm")
    if args.batch_size <= 0:
        parser.error("--batch-size phải lớn hơn 0")
    if len(args.codec) != 4:
        parser.error("--codec phải có đúng 4 ký tự, ví dụ mp4v")
    return args


if __name__ == "__main__":
    process_video(parse_args())

