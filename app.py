"""Streamlit interface for converting uploaded videos with AnimeGANv2."""

from __future__ import annotations

import json
import hashlib
import re
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import streamlit as st
import torch

from cartoonize_video import find_ffmpeg, process_video
from inspect_video import inspect_video


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
SUPPORTED_VIDEO_TYPES = ["mp4", "mov", "avi", "mkv", "webm", "m4v"]
VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
}
MODEL_LABELS = {
    "paprika": "Paprika, phong cách anime tổng quát",
    "face_paint_512_v1": "Face Paint v1, thiên về chân dung",
    "face_paint_512_v2": "Face Paint v2 (512), ưu tiên chân dung",
    "celeba_distill": "CelebA Distill, khuôn mặt",
}
MODEL_PRIORITY = {
    "face_paint_512_v2": 0,
    "face_paint_512_v1": 1,
    "paprika": 2,
    "celeba_distill": 3,
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #f4f7f8;
            --muted: #9aa8ad;
            --muted-strong: #c3ced1;
            --canvas: #0f1316;
            --surface: #171d21;
            --surface-soft: #1d2529;
            --line: #2b373c;
            --accent: #8ee3ff;
            --accent-strong: #50c9ee;
            --danger: #ff9f9f;
        }

        [data-testid="stAppViewContainer"] {
            background: var(--canvas);
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        [data-testid="stToolbar"] {
            visibility: hidden;
        }
        .block-container {
            max-width: 1240px;
            padding: 2.6rem 2.5rem 4rem;
        }
        .frame-topline {
            color: var(--accent);
            font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
            font-size: 0.7rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            margin-bottom: 1.2rem;
        }
        .frame-topline span {
            color: var(--muted);
            padding: 0 0.45rem;
        }
        .frame-title {
            color: var(--ink);
            font-size: clamp(2.4rem, 5vw, 4.8rem);
            font-weight: 650;
            letter-spacing: -0.065em;
            line-height: 0.98;
            max-width: 780px;
            margin: 0;
        }
        .frame-intro {
            color: var(--muted-strong);
            font-size: 1rem;
            line-height: 1.55;
            max-width: 570px;
            margin: 1.35rem 0 2.7rem;
        }
        .section-label {
            color: var(--muted);
            font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
            font-size: 0.7rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin: 0 0 0.75rem;
        }
        .surface-note {
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.5;
            margin-top: 0.6rem;
        }
        .empty-preview {
            min-height: 340px;
            border: 1px solid var(--line);
            border-radius: 14px;
            background: var(--surface);
            display: flex;
            flex-direction: column;
            justify-content: center;
            padding: 2rem;
        }
        .empty-preview strong {
            color: var(--ink);
            font-size: 1.35rem;
            font-weight: 550;
        }
        .empty-preview p {
            color: var(--muted);
            line-height: 1.55;
            max-width: 390px;
            margin: 0.6rem 0 0;
        }
        .result-heading {
            color: var(--ink);
            font-size: 1.4rem;
            font-weight: 600;
            letter-spacing: -0.03em;
            margin: 1.2rem 0 0.8rem;
        }
        .status-line {
            color: var(--muted-strong);
            font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
            font-size: 0.75rem;
        }
        .stProgress > div > div > div > div {
            background: var(--accent-strong);
        }
        .stButton > button,
        .stDownloadButton > button {
            border: 1px solid var(--accent-strong);
            border-radius: 10px;
            background: var(--accent);
            color: #071116;
            font-weight: 700;
            min-height: 2.7rem;
            transition: transform 160ms ease, background 160ms ease;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover {
            background: #b9efff;
            color: #071116;
            transform: translateY(-1px);
        }
        .stButton > button:active,
        .stDownloadButton > button:active {
            transform: scale(0.98);
        }
        [data-testid="stFileUploader"] section {
            border: 1px dashed #4a626a;
            border-radius: 12px;
            background: var(--surface);
        }
        [data-testid="stFileUploader"] section:hover {
            border-color: var(--accent);
        }
        [data-testid="stMetric"] {
            border-top: 1px solid var(--line);
            padding-top: 0.65rem;
        }
        [data-testid="stMetricLabel"] {
            color: var(--muted);
        }
        [data-testid="stMetricValue"] {
            color: var(--ink);
            font-size: 2rem;
        }
        [data-testid="stAlert"] {
            border-radius: 10px;
        }
        @media (max-width: 767px) {
            .block-container {
                padding: 1.4rem 1rem 3rem;
            }
            .frame-title {
                font-size: 3rem;
            }
            .frame-intro {
                margin-bottom: 1.8rem;
            }
        }
        @media (prefers-reduced-motion: reduce) {
            .stButton > button,
            .stDownloadButton > button {
                transition: none;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_stem(name: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(name).stem).strip("_")
    return stem or "video"


def video_mime_type(name: str) -> str:
    return VIDEO_MIME_TYPES.get(Path(name).suffix.lower(), "video/mp4")


def validate_facebook_url(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("Hãy nhập link Facebook Reel.")
    if not normalized.startswith(("https://", "http://")):
        normalized = f"https://{normalized}"

    parsed = urlparse(normalized)
    hostname = (parsed.hostname or "").lower().rstrip(".")
    allowed = (
        hostname == "facebook.com"
        or hostname.endswith(".facebook.com")
        or hostname == "fb.watch"
        or hostname == "instagram.com"
        or hostname.endswith(".instagram.com")
        or hostname == "ig.watch"
    )
    if parsed.scheme != "https" or not allowed:
        raise ValueError("Chỉ hỗ trợ URL HTTPS thuộc Facebook, fb.watch hoặc Instagram.")
    return normalized


def available_models() -> list[tuple[str, Path]]:
    models = []
    for checkpoint in sorted(MODEL_DIR.glob("*.pt")):
        key = checkpoint.stem
        label = MODEL_LABELS.get(key, key)
        models.append((label, checkpoint))
    return sorted(
        models,
        key=lambda item: (
            MODEL_PRIORITY.get(item[1].stem, 99),
            item[1].name.lower(),
        ),
    )


def save_upload(uploaded_file, directory: Path) -> Path:
    path = directory / Path(uploaded_file.name).name
    path.write_bytes(bytes(uploaded_file.getbuffer()))
    return path


def save_video_bytes(video_bytes: bytes, filename: str, directory: Path) -> Path:
    safe_name = Path(filename).name or "video.mp4"
    path = directory / safe_name
    path.write_bytes(video_bytes)
    return path


def preview_metadata_bytes(video_bytes: bytes, filename: str) -> dict | None:
    try:
        with tempfile.TemporaryDirectory(prefix="animegan_preview_") as temp_dir:
            path = save_video_bytes(video_bytes, filename, Path(temp_dir))
            return inspect_video(path)
    except Exception:
        return None


def trim_video_bytes(
    video_bytes: bytes,
    filename: str,
    start_seconds: float,
    end_seconds: float,
) -> tuple[str, bytes]:
    """Trim a video accurately with FFmpeg while keeping video and audio."""
    if end_seconds <= start_seconds:
        raise ValueError("Thời điểm kết thúc phải lớn hơn thời điểm bắt đầu.")

    ffmpeg_path = find_ffmpeg()
    if not ffmpeg_path:
        raise RuntimeError(
            "Cần FFmpeg để cắt video và giữ audio. Hãy cài FFmpeg rồi khởi động lại Tool."
        )

    duration = end_seconds - start_seconds
    trimmed_name = f"{safe_stem(filename)}_trimmed.mp4"
    with tempfile.TemporaryDirectory(prefix="animegan_trim_") as temp_dir:
        work_dir = Path(temp_dir)
        input_path = save_video_bytes(video_bytes, filename, work_dir)
        output_path = work_dir / trimmed_name
        command = [
            str(ffmpeg_path),
            "-y",
            "-i",
            str(input_path),
            "-ss",
            f"{start_seconds:.3f}",
            "-t",
            f"{duration:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0 or not output_path.is_file():
            details = result.stderr.strip().splitlines()[-1] if result.stderr else ""
            suffix = f" Chi tiết: {details}" if details else ""
            raise RuntimeError(f"FFmpeg không thể cắt video.{suffix}")
        return trimmed_name, output_path.read_bytes()


def download_facebook_video(url: str) -> tuple[str, bytes]:
    normalized_url = validate_facebook_url(url)
    try:
        from yt_dlp import YoutubeDL
    except ImportError as error:
        raise RuntimeError(
            "Chưa cài yt-dlp. Chạy `python -m pip install -r requirements.txt` rồi khởi động lại Tool."
        ) from error

    ffmpeg_path = find_ffmpeg()
    with tempfile.TemporaryDirectory(prefix="animegan_facebook_") as temp_dir:
        work_dir = Path(temp_dir)
        options = {
            "outtmpl": str(work_dir / "facebook_reel.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "retries": 2,
            "socket_timeout": 30,
            "restrictfilenames": True,
        }
        if ffmpeg_path:
            options.update(
                {
                    "ffmpeg_location": str(Path(ffmpeg_path).parent),
                    "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                    "merge_output_format": "mp4",
                }
            )
        else:
            options["format"] = "best[ext=mp4]/best"

        try:
            with YoutubeDL(options) as downloader:
                downloader.download([normalized_url])
        except Exception as error:
            raise RuntimeError(
                "Không tải được Reel. Link có thể yêu cầu đăng nhập, bị giới hạn khu vực "
                f"hoặc Facebook đã thay đổi dữ liệu. Chi tiết: {error}"
            ) from error

        candidates = [
            path
            for path in work_dir.glob("facebook_reel.*")
            if path.is_file() and path.stat().st_size > 0 and path.suffix.lower() in {
                ".mp4",
                ".mov",
                ".mkv",
                ".webm",
                ".m4v",
                ".avi",
            }
        ]
        if not candidates:
            raise RuntimeError("Facebook không trả về file video có thể xử lý.")

        selected = next(
            (path for path in candidates if path.suffix.lower() == ".mp4"),
            candidates[0],
        )
        return selected.name, selected.read_bytes()


def preview_metadata(uploaded_file) -> dict | None:
    return preview_metadata_bytes(uploaded_file.getvalue(), uploaded_file.name)


def render_metadata(metadata: dict) -> None:
    cols = st.columns([1.25, 1, 1, 1], gap="small")
    cols[0].metric("Độ phân giải", metadata.get("resolution", "-") )
    fps = metadata.get("fps", 0)
    cols[1].metric("FPS", f"{fps:.2f}" if fps else "-")
    frame_count = metadata.get("frame_count_reported_by_opencv", 0)
    cols[2].metric("Số frame video", f"{frame_count:,}" if frame_count else "-")
    duration = metadata.get("duration_seconds_estimated")
    cols[3].metric("Thời lượng video", f"{duration:.1f}s" if duration else "-")


def process_source_video(
    video_bytes: bytes, source_name: str, model_path: Path, options: dict
) -> None:
    progress = st.progress(0, text="Đang chuẩn bị model...")
    status = st.empty()

    def update_progress(done: int, total: int) -> None:
        ratio = min(done / total, 1.0) if total > 0 else 0.0
        progress.progress(ratio, text=f"Đang chuyển đổi frame {done}/{total or '?'}")
        status.markdown(f"<span class='status-line'>FRAME {done:04d}</span>", unsafe_allow_html=True)

    try:
        with tempfile.TemporaryDirectory(prefix="animegan_run_") as temp_dir:
            work_dir = Path(temp_dir)
            input_path = save_video_bytes(video_bytes, source_name, work_dir)
            output_path = work_dir / f"{safe_stem(source_name)}_animeganv2.mp4"
            report_path = work_dir / "report.json"
            args = SimpleNamespace(
                input=input_path,
                output=output_path,
                model=model_path,
                device=options["device"],
                load_size=options["load_size"],
                align_corners=options["align_corners"],
                codec="mp4v",
                audio_from=input_path if options["keep_audio"] else None,
                report=report_path,
                max_frames=0,
                batch_size=max(1, int(options.get("batch_size", 1))),
                browser_compatible=bool(options.get("browser_compatible", False)),
                log_every=max(1, options["log_every"]),
                progress_callback=update_progress,
            )
            report = process_video(args)
            output_bytes = output_path.read_bytes()
            st.session_state["result_video"] = output_bytes
            st.session_state["result_name"] = output_path.name
            st.session_state["result_report"] = report
            st.session_state["result_source"] = source_name
        progress.progress(1.0, text="Hoàn tất chuyển đổi")
        status.empty()
    except Exception as error:
        progress.empty()
        status.empty()
        raise RuntimeError(str(error)) from error


def process_uploaded_video(uploaded_file, model_path: Path, options: dict) -> None:
    process_source_video(uploaded_file.getvalue(), uploaded_file.name, model_path, options)


def render_result() -> None:
    output_bytes = st.session_state.get("result_video")
    report = st.session_state.get("result_report")
    if not output_bytes or not report:
        return

    st.markdown("<div class='result-heading'>Video hoạt hình</div>", unsafe_allow_html=True)
    st.video(output_bytes, format="video/mp4", width="stretch")
    output_name = st.session_state.get("result_name", "animeganv2_output.mp4")
    st.download_button(
        "Tải video kết quả",
        data=output_bytes,
        file_name=output_name,
        mime="video/mp4",
        width="stretch",
    )
    output_meta = report.get("output", {})
    processing = report.get("processing", {})
    cols = st.columns(4)
    cols[0].metric("Độ phân giải", output_meta.get("resolution", "-"))
    cols[1].metric("FPS", f"{output_meta.get('fps', 0):.2f}")
    cols[2].metric("Số frame output", f"{processing.get('processed_frames', 0):,}")
    output_duration = output_meta.get("duration_seconds_estimated")
    cols[3].metric(
        "Thời lượng video",
        f"{output_duration:.1f}s" if output_duration else "-",
    )
    model_info = report.get("model", {})
    device_info = model_info.get("device_info", {})
    gpu_label = device_info.get("gpu_name") or "CPU"
    st.caption(
        f"Thời gian xử lý: {processing.get('processing_seconds', 0):.1f}s · "
        f"Thiết bị: {model_info.get('device', '-')} ({gpu_label}) · "
        f"Batch: {processing.get('batch_size', 1)} · "
        "FPS output được giữ nguyên theo video input."
    )
    st.download_button(
        "Tải report JSON",
        data=json.dumps(report, ensure_ascii=False, indent=2),
        file_name="animeganv2_report.json",
        mime="application/json",
        width="stretch",
    )
    st.info("AnimeGANv2 xử lý từng frame độc lập. Hãy xem đoạn chuyển động liên tục để kiểm tra hiện tượng nhấp nháy.")


def main() -> None:
    st.set_page_config(
        page_title="Frame Studio | AnimeGANv2",
        page_icon="F",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_styles()
    cuda_available = torch.cuda.is_available()

    st.markdown(
        """
        <div class="frame-topline">Frame Studio <span>/</span> AnimeGANv2</div>
        <h1 class="frame-title">Biến video thường thành chuyển động anime.</h1>
        <p class="frame-intro">Tải video lên, chọn phong cách, rồi xuất một bản video hoạt hình bằng AnimeGANv2 trên PyTorch.</p>
        """,
        unsafe_allow_html=True,
    )

    models = available_models()
    left, right = st.columns([0.94, 1.06], gap="large")
    with left:
        st.markdown("<div class='section-label'>Input video</div>", unsafe_allow_html=True)
        source_mode = st.segmented_control(
            "Nguồn video",
            ["Tải file từ máy", "Reel"],
            default="Tải file từ máy",
            key="input_source_mode",
        )
        source_bytes: bytes | None = None
        source_name: str | None = None
        source_metadata: dict | None = None

        if source_mode == "Reel":
            with st.form("facebook_url_form", border=False):
                facebook_url = st.text_input(
                    "Link Facebook Reel",
                    placeholder="https://www.facebook.com/reel/...",
                    help="Chỉ tải được video công khai mà Facebook/yt-dlp cho phép truy cập.",
                )
                fetch_video = st.form_submit_button(
                    "Lấy video từ Reel",
                    type="secondary",
                    width="stretch",
                )
            if fetch_video:
                try:
                    with st.status("Đang lấy video từ Facebook...", expanded=False) as status:
                        remote_name, remote_bytes = download_facebook_video(facebook_url)
                        status.update(label="Đã lấy video từ Facebook", state="complete")
                    st.session_state["remote_video_name"] = remote_name
                    st.session_state["remote_video_bytes"] = remote_bytes
                    st.session_state["remote_video_url"] = facebook_url.strip()
                    st.session_state.pop("result_video", None)
                    st.session_state.pop("result_report", None)
                except Exception as error:
                    st.session_state.pop("remote_video_name", None)
                    st.session_state.pop("remote_video_bytes", None)
                    st.error(str(error))
                    st.info(
                        "Nếu Reel yêu cầu đăng nhập, hãy tải video bằng trình duyệt rồi chọn "
                        "Tải file từ máy. Tool không tự lấy cookie Facebook."
                    )

            source_bytes = st.session_state.get("remote_video_bytes")
            source_name = st.session_state.get("remote_video_name")
            if source_bytes and source_name:
                metadata = preview_metadata_bytes(source_bytes, source_name)
                source_metadata = metadata
                st.caption(f"{source_name}  |  {len(source_bytes) / (1024 * 1024):.1f} MB")
                if metadata:
                    st.video(
                        source_bytes,
                        format=video_mime_type(source_name),
                        width="stretch",
                    )
                    render_metadata(metadata)
                    st.caption(
                        f"Nguồn: {st.session_state.get('remote_video_url', 'Facebook Reel')}"
                    )
                else:
                    st.warning("Đã tải file nhưng OpenCV chưa đọc được metadata video.")
            elif not fetch_video:
                st.caption("Dán link Facebook Reel rồi bấm `Lấy video từ Facebook`.")
        else:
            uploaded_file = st.file_uploader(
                "Chọn video từ máy",
                type=SUPPORTED_VIDEO_TYPES,
                label_visibility="collapsed",
                help="MP4, MOV, AVI, MKV, WEBM hoặc M4V",
            )
            if uploaded_file:
                source_bytes = uploaded_file.getvalue()
                source_name = uploaded_file.name
                metadata = preview_metadata(uploaded_file)
                source_metadata = metadata
                st.caption(f"{uploaded_file.name}  |  {uploaded_file.size / (1024 * 1024):.1f} MB")
                if metadata:
                    st.video(
                        source_bytes,
                        format=video_mime_type(source_name),
                        width="stretch",
                    )
                    render_metadata(metadata)
                else:
                    st.warning("Đã nhận file nhưng OpenCV chưa đọc được metadata video.")

        ffmpeg_available = find_ffmpeg() is not None
        working_bytes = source_bytes
        working_name = source_name
        trim_enabled = False
        trim_applied = False
        if source_bytes and source_name:
            source_signature = hashlib.sha1(source_bytes).hexdigest()
            if st.session_state.get("trimmed_source_signature") != source_signature:
                st.session_state.pop("trimmed_video_bytes", None)
                st.session_state.pop("trimmed_video_name", None)
                st.session_state.pop("trimmed_source_signature", None)

            trimmed_bytes = st.session_state.get("trimmed_video_bytes")
            trimmed_name = st.session_state.get("trimmed_video_name")
            trim_enabled = st.checkbox(
                "Cắt ngắn video trước khi xử lý",
                value=False,
                key="trim_enabled",
                disabled=not ffmpeg_available or not source_metadata,
                help="Chọn khoảng thời gian cần giữ lại. FFmpeg sẽ cắt video và giữ audio.",
            )
            trim_range: tuple[float, float] | None = None
            if trim_enabled and source_metadata:
                original_duration = float(source_metadata.get("duration_seconds_estimated") or 0)
                if original_duration > 0:
                    trim_range = st.slider(
                        "Khoảng thời gian giữ lại (giây)",
                        min_value=0.0,
                        max_value=original_duration,
                        value=(0.0, min(original_duration, 30.0)),
                        step=0.1,
                        help="Kéo hai đầu mốc để chọn thời điểm bắt đầu và kết thúc.",
                    )
                    trim_start, trim_end = trim_range
                    if trim_end <= trim_start:
                        st.error("Thời điểm kết thúc phải lớn hơn thời điểm bắt đầu.")
                    elif st.button("Áp dụng đoạn cắt", width="stretch"):
                        try:
                            with st.status("Đang cắt video...", expanded=False) as status:
                                trimmed_name, trimmed_bytes = trim_video_bytes(
                                    source_bytes,
                                    source_name,
                                    trim_start,
                                    trim_end,
                                )
                                status.update(label="Đã cắt video", state="complete")
                            st.session_state["trimmed_video_name"] = trimmed_name
                            st.session_state["trimmed_video_bytes"] = trimmed_bytes
                            st.session_state["trimmed_source_signature"] = source_signature
                            working_name = trimmed_name
                            working_bytes = trimmed_bytes
                            trim_applied = True
                            st.session_state.pop("result_video", None)
                            st.session_state.pop("result_report", None)
                            st.success(
                                f"Đã tạo đoạn video {trim_end - trim_start:.1f} giây."
                            )
                        except Exception as error:
                            st.error(f"Không thể cắt video: {error}")
                else:
                    st.warning("Không xác định được thời lượng để cắt video.")

            if trim_enabled and trimmed_bytes and trimmed_name:
                working_bytes = trimmed_bytes
                working_name = trimmed_name
                trim_applied = True
                trimmed_metadata = preview_metadata_bytes(trimmed_bytes, trimmed_name)
                st.caption(f"Preview đoạn đã cắt: {trimmed_name}")
                if trimmed_metadata:
                    st.video(
                        trimmed_bytes,
                        format=video_mime_type(trimmed_name),
                        width="stretch",
                    )
                    render_metadata(trimmed_metadata)
            elif trim_enabled:
                st.caption("Chọn khoảng thời gian rồi bấm `Áp dụng đoạn cắt` trước khi xử lý.")

        st.markdown("<div class='section-label'>Model setup</div>", unsafe_allow_html=True)
        if not models:
            st.error("Chưa có checkpoint trong thư mục models.")
            st.code("models/face_paint_512_v2.pt", language="text")
            st.caption(
                "Tải checkpoint Face Paint v2 theo hướng dẫn trong README.md rồi "
                "khởi động lại tool. Có thể dùng paprika.pt cho cảnh không tập trung vào khuôn mặt."
            )
        else:
            labels = [label for label, _ in models]
            selected_label = st.selectbox("Phong cách", labels, label_visibility="collapsed")
            selected_model = dict(models)[selected_label]
            st.caption(f"Checkpoint: {selected_model.name}")

        device_options = ["Tự động", "CPU"]
        if cuda_available:
            device_options.append("CUDA")
        device_label = st.selectbox(
            "Thiết bị suy luận",
            device_options,
            index=0,
            help="Tự động sẽ ưu tiên CUDA nếu PyTorch nhận diện được GPU.",
        )
        device = {"Tự động": "auto", "CPU": "cpu", "CUDA": "cuda"}[device_label]
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            st.success(f"GPU sẵn sàng: {gpu_name}")
        else:
            st.warning(
                "PyTorch hiện không nhận CUDA nên đang chạy CPU. Cài bản PyTorch có CUDA rồi khởi động lại Tool để dùng GPU."
            )
        batch_size = st.select_slider(
            "Số frame xử lý mỗi lượt",
            options=[1, 2, 4, 8],
            value=4 if cuda_available else 1,
            help="Batch lớn hơn thường nhanh hơn trên GPU nhưng tốn VRAM hơn. Không làm thay đổi FPS hoặc số frame output.",
        )
        load_size = st.select_slider(
            "Kích thước suy luận",
            options=[256, 320, 384, 448, 512, 640, 768],
            value=640,
            help="640 là mức cân bằng giữa độ nét và tốc độ; 768 thường nét hơn nhưng chậm hơn và tốn VRAM hơn.",
        )
        keep_audio = st.checkbox(
            "Giữ audio từ video gốc",
            value=ffmpeg_available,
            disabled=not ffmpeg_available,
            help="Cần FFmpeg trong PATH. OpenCV không ghi audio.",
        )
        if not ffmpeg_available:
            st.caption("FFmpeg chưa có trong PATH. Output sẽ chỉ có hình.")

        ready = bool(working_bytes and working_name and models and (not trim_enabled or trim_applied))
        st.markdown("<div class='surface-note'>Video được xử lý tuần tự. Với video dài, CPU có thể mất nhiều thời gian.</div>", unsafe_allow_html=True)
        if st.button("Chuyển thành video hoạt hình", type="primary", width="stretch", disabled=not ready):
            st.session_state.pop("result_video", None)
            try:
                process_source_video(
                    working_bytes,
                    working_name,
                    selected_model,
                    {
                        "device": device,
                        "batch_size": batch_size,
                        "load_size": load_size,
                        "align_corners": False,
                        "keep_audio": keep_audio,
                        "browser_compatible": ffmpeg_available,
                        "log_every": 30,
                    },
                )
                st.success("Đã chuyển đổi xong.")
            except Exception as error:
                st.error(f"Không thể chuyển đổi: {error}")

    with right:
        st.markdown("<div class='section-label'>Preview</div>", unsafe_allow_html=True)
        if st.session_state.get("result_video"):
            render_result()
        else:
            st.markdown(
                """
                <div class="empty-preview">
                    <strong>Chưa có video kết quả</strong>
                    <p>Chọn một video ở bên trái. Bản preview và nút tải xuống sẽ xuất hiện sau khi AnimeGANv2 hoàn tất.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()

