# VideoToCartoon

Tool chuyển video thường thành video hoạt hình bằng **AnimeGANv2 + PyTorch**. Ứng dụng có giao diện Streamlit để tải video từ máy hoặc lấy một Facebook Reel công khai, xử lý từng frame trên CPU/GPU, ghép thành video mới và cho phép xem/tải kết quả ngay trên trình duyệt.

## Tính năng

- Chuyển đổi video theo từng frame bằng AnimeGANv2 PyTorch.
- Giữ nguyên độ phân giải, FPS, thứ tự frame và thời lượng hợp lý của video đầu vào.
- Chọn checkpoint/style, thiết bị `Tự động`, `CPU` hoặc `CUDA`, batch size và kích thước suy luận.
- Có thể giữ audio bằng FFmpeg.
- Tạo video H.264/yuv420p để trình duyệt phát preview ổn định.
- Nhập video từ máy hoặc dán URL Facebook Reel công khai qua `yt-dlp`.
- Hiển thị metadata, thời gian xử lý, thiết bị GPU và cho tải report JSON.
- Có công cụ CLI để kiểm tra metadata và validate output độc lập với giao diện.

## Luồng xử lý

```text
Video đầu vào
    │
    ├─ OpenCV đọc tuần tự từng frame
    ├─ BGR → RGB, resize về kích thước suy luận, chuẩn hóa [-1, 1]
    ├─ AnimeGANv2 Generator suy luận theo batch trên CPU/CUDA
    ├─ tensor output → RGB/BGR, resize Lanczos về kích thước gốc
    ├─ OpenCV ghi video và giữ FPS đầu vào
    └─ FFmpeg tùy chọn: ghép audio + tạo bản H.264 cho trình duyệt
```

AnimeGANv2 xử lý độc lập từng frame. Vì vậy, các vùng chi tiết hoặc chuyển động nhanh có thể xuất hiện nhấp nháy giữa các frame; đây là giới hạn của bài test frame-by-frame, không phải lỗi mất frame của OpenCV.

## Công nghệ và model

| Thành phần | Vai trò |
| --- | --- |
| Python | Ngôn ngữ triển khai |
| OpenCV | Đọc/ghi video, lấy metadata và tiền xử lý frame |
| PyTorch | Nạp model, chọn CPU/CUDA và suy luận batch |
| AnimeGANv2 | Chuyển phong cách ảnh/video sang hoạt hình |
| Streamlit | Giao diện upload, preview, progress và download |
| FFmpeg | Ghép audio và mã hóa H.264 tương thích trình duyệt |
| yt-dlp | Lấy video từ URL Facebook Reel công khai |

Model mặc định là checkpoint `paprika.pt` từ [bryandlee/animegan2-pytorch](https://github.com/bryandlee/animegan2-pytorch). Kiến trúc Generator tương thích được đặt tại [`models/animegan2.py`](models/animegan2.py). Checkpoint không được commit vào repository; mỗi máy tải riêng vào `models/paprika.pt`.

## Yêu cầu

- Windows 10/11 hoặc môi trường Python tương đương.
- Python 3.10 trở lên; dự án đã được kiểm tra với Python 3.14.
- RAM đủ cho video và batch đang chọn.
- GPU NVIDIA/CUDA là tùy chọn. CPU vẫn chạy được nhưng chậm hơn.
- FFmpeg được khuyến nghị nếu cần giữ audio, lấy video Facebook ở định dạng đầy đủ hoặc xem output trực tiếp trên trình duyệt.

## Cài đặt

Mở PowerShell tại thư mục `VideotoCartoon`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu dùng NVIDIA GPU, cài profile PyTorch CUDA thay cho bản CPU mặc định:

```powershell
pip install -r requirements-gpu.txt
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Cài FFmpeg và bảo đảm lệnh sau chạy được:

```powershell
ffmpeg -version
```

## Tải checkpoint

Tạo thư mục `models` nếu chưa có, sau đó tải checkpoint `paprika.pt`:

```powershell
Invoke-WebRequest `
  -Uri "https://github.com/bryandlee/animegan2-pytorch/raw/main/weights/paprika.pt" `
  -OutFile "models\paprika.pt"
```

Có thể dùng checkpoint khác nếu kiến trúc tương thích, nhưng cần đặt file vào `models/` và chọn nó trong giao diện. Không đưa checkpoint, video cá nhân hoặc video kết quả vào Git.

## Chạy giao diện

```powershell
python -m streamlit run app.py
```

Hoặc:

```powershell
.\run_app.ps1
```

Mở địa chỉ Streamlit hiển thị trong terminal, thường là `http://localhost:8501`.

### Quy trình trong giao diện

1. Chọn `Tải file từ máy` để upload video hoặc chọn `Link Facebook Reel`.
2. Với Facebook, dán URL Reel công khai rồi bấm `Lấy video từ Facebook`.
3. Kiểm tra preview và metadata: độ phân giải, FPS, số frame, thời lượng.
4. Chọn style/checkpoint, thiết bị, batch size và kích thước suy luận.
5. Bật `Giữ audio từ video gốc` nếu đã cài FFmpeg.
6. Bấm `Chuyển thành video hoạt hình`.
7. Xem video đầu vào/kết quả trong trình duyệt, tải video và report JSON.

### Chọn tham số chất lượng và tốc độ

- `Kích thước suy luận` cao hơn thường giữ chi tiết tốt hơn nhưng tốn VRAM và thời gian hơn. Giá trị cân bằng là `640`; dùng `768` khi GPU đủ mạnh.
- `Batch` lớn thường tận dụng GPU tốt hơn, nhưng cần nhiều VRAM hơn. Nếu gặp lỗi out-of-memory, giảm batch xuống `1` hoặc `2`.
- `Tự động` ưu tiên CUDA khi PyTorch nhận diện được GPU, sau đó tự chuyển về CPU.
- Output được resize về đúng kích thước video đầu vào; tăng `load-size` không tạo thêm chi tiết đã mất trong video gốc.

## Chạy bằng CLI

### Chuyển video không audio

```powershell
python cartoonize_video.py `
  --input input\sample.mp4 `
  --output output\sample_anime.mp4 `
  --model models\paprika.pt `
  --device auto `
  --load-size 640 `
  --batch-size 4 `
  --report reports\sample_report.json
```

### Giữ audio và tạo output tương thích trình duyệt

```powershell
python cartoonize_video.py `
  --input input\sample.mp4 `
  --output output\sample_anime_h264.mp4 `
  --model models\paprika.pt `
  --device cuda `
  --load-size 640 `
  --batch-size 4 `
  --audio-from input\sample.mp4 `
  --browser-compatible `
  --report reports\sample_report.json
```

`cv2.VideoWriter` không ghi audio. Tùy chọn `--audio-from` yêu cầu FFmpeg để ghép audio từ file nguồn. Tùy chọn `--browser-compatible` tạo H.264/yuv420p; nếu bỏ tùy chọn này, một số trình duyệt có thể không phát được codec `mp4v` trực tiếp.

## Facebook Reel: quy tắc sử dụng

Ứng dụng chỉ nhận URL HTTPS thuộc `facebook.com`, subdomain của Facebook hoặc `fb.watch`, ví dụ:

```text
https://www.facebook.com/reel/1075380275136838
https://fb.watch/xxxxxxxx/
```

Video phải là nội dung công khai và tài khoản/mạng hiện tại phải được Facebook cho phép truy cập. Tool không tự lấy cookie, không vượt qua đăng nhập, vùng hạn chế, DRM, paywall hay quyền riêng tư. Người dùng chịu trách nhiệm về quyền tải xuống, bản quyền và quyền sử dụng video. Nếu Facebook đổi cơ chế phân phối hoặc URL yêu cầu đăng nhập, hãy tải video hợp pháp về máy rồi dùng nguồn `Tải file từ máy`.

## Kiểm tra metadata và output

Đọc thông tin video đầu vào:

```powershell
python inspect_video.py `
  --input input\sample.mp4 `
  --report reports\input_metadata.json
```

Validate số frame, FPS, độ phân giải, tỷ lệ frame đen và tạo ảnh so sánh đầu/cuối/giữa video:

```powershell
python validate_video.py `
  --input input\sample.mp4 `
  --output output\sample_anime_h264.mp4 `
  --report reports\validation.json `
  --comparison reports\input_output_comparison.jpg
```

Report JSON của giao diện/CLI ghi lại metadata input-output, thiết bị, GPU, batch size, load size, thời gian xử lý, audio và các cảnh báo. Đây là dữ liệu kiểm thử/đánh giá, không phải video preview.

## Cấu trúc dự án

```text
VideotoCartoon/
├── app.py                  # Giao diện Streamlit và workflow upload/Facebook
├── cartoonize_video.py     # Pipeline AnimeGANv2, OpenCV, audio và H.264
├── models/animegan2.py     # Kiến trúc Generator AnimeGANv2
├── inspect_video.py        # Đọc resolution/FPS/frame count/codec
├── validate_video.py       # Validate output và tạo ảnh so sánh
├── run_app.ps1             # Chạy Streamlit trên PowerShell
├── requirements.txt        # Dependencies dùng chung
├── requirements-gpu.txt    # Profile PyTorch CUDA cho NVIDIA
├── .streamlit/config.toml  # Cấu hình giao diện/upload
├── input/.gitkeep          # Nơi đặt video đầu vào cục bộ
├── models/.gitkeep         # Nơi đặt checkpoint cục bộ
├── output/.gitkeep         # Nơi sinh video kết quả cục bộ
├── reports/.gitkeep        # Nơi sinh JSON/ảnh kiểm thử cục bộ
├── TODO.md                 # Checklist và ghi chú kiểm thử
└── README.md
```

## Xử lý lỗi thường gặp

| Hiện tượng | Cách xử lý |
| --- | --- |
| `Chưa có checkpoint` | Tải `paprika.pt` vào `models/paprika.pt`. |
| CUDA không được dùng | Kiểm tra `torch.cuda.is_available()` và cài đúng profile `requirements-gpu.txt`. |
| CUDA out of memory | Giảm batch size hoặc load size. |
| Không giữ được audio | Cài FFmpeg và bật giữ audio; OpenCV không tự ghi audio. |
| Trình duyệt không phát output | Chạy với `--browser-compatible` hoặc dùng nút tải video để mở bằng trình phát khác. |
| Facebook không tải được | Dùng Reel công khai, URL HTTPS hợp lệ; nếu vẫn lỗi, tải file hợp pháp về máy rồi upload local. |
| Video hơi mờ | Tăng load size lên `768` nếu đủ VRAM; chất lượng vẫn bị giới hạn bởi video nguồn và checkpoint. |
| Nhấp nháy giữa các frame | Đây là giới hạn của suy luận từng frame; pipeline hiện chưa có temporal consistency. |

## Giới hạn và hướng phát triển

- AnimeGANv2 không được thiết kế riêng cho tính nhất quán theo thời gian của video.
- Mỗi lần chạy cần nạp model và xử lý toàn bộ frame; video dài sẽ tốn thời gian.
- Tăng độ phân giải suy luận chỉ cải thiện cách model nhìn frame, không khôi phục thông tin đã mất.
- Có thể phát triển thêm temporal smoothing, optical-flow consistency, hàng đợi xử lý nền và các model video chuyên dụng.

## Credits

- [AnimeGANv2 PyTorch by bryandlee](https://github.com/bryandlee/animegan2-pytorch)
- [AnimeGANv2 paper/project](https://github.com/Tachibana961/AnimeGANv2)
- [PyTorch](https://pytorch.org/), [OpenCV](https://opencv.org/), [Streamlit](https://streamlit.io/), [FFmpeg](https://ffmpeg.org/) và [yt-dlp](https://github.com/yt-dlp/yt-dlp)

## License

Repository hiện chưa thêm file license. Trước khi phát hành hoặc tái sử dụng công khai, hãy bổ sung license phù hợp và kiểm tra license của model/checkpoint cùng video đầu vào.
