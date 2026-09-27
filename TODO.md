# TODO bài test Video thường → video hoạt hình

> Mỗi bước có **ghi chú**, **hướng dẫn thực hiện** và **kết quả cần ghi**. Đánh dấu `[x]` chỉ sau khi đã kiểm tra thật trên video; không đánh dấu chỉ vì chương trình đã được viết.

## Trạng thái hiện tại

- [x] Đã tạo khung thư mục, script xử lý, script kiểm tra và tài liệu hướng dẫn.
- [x] Đã đổi pipeline sang AnimeGANv2 + PyTorch.
- [x] Đã tạo giao diện Streamlit để upload, xem trực tiếp video gốc/video hoạt hình và tải kết quả.
- [x] Đã tạo và dùng video smoke test trong `input/`.
- [x] Đã đặt checkpoint AnimeGANv2 tại `models/paprika.pt`.
- [x] Đã chạy suy luận thực tế trên smoke test.
- [x] Đã chạy trên video thực tế mong muốn của người dùng.
- [x] Đã điều chỉnh chất lượng hình: mặc định `load-size 640`, hỗ trợ chọn `768` và resize output bằng Lanczos4.
- [x] Đã thêm nguồn input Facebook Reel công khai: tải bằng `yt-dlp`, preview và đưa vào pipeline xử lý.

## Bước 0 — Mở giao diện Tool

- [x] Cài dependencies bằng `pip install -r requirements.txt`.
- [x] Đặt checkpoint vào `models/paprika.pt`.
- [x] Chạy:

  ```powershell
  python -m streamlit run app.py
  ```

  Có thể dùng nhanh `./run_app.ps1` trên PowerShell. Server đã kiểm tra trả HTTP 200.

- [x] Upload video smoke test ở vùng `Input video`.
- [x] Chọn style Paprika, CPU và kích thước suy luận 512; audio được tắt do máy chưa có FFmpeg.
- [x] Bấm `Chuyển thành video hoạt hình`.
- [x] Render player trực tiếp cho video gốc sau upload và video hoạt hình sau xử lý; tạo hai nút tải video kết quả/report JSON.
- [x] Upload video thực tế mong muốn của người dùng, chạy bằng CUDA và đánh giá chất lượng hình ảnh.

**Ghi chú:** smoke test đã xử lý 17/17 frame; video thật đã được upload qua AppTest, chuyển đổi bằng CUDA batch 4, giữ audio và tạo đủ player/download. Video output của giao diện được FFmpeg mã hóa H.264/yuv420p để Chrome phát trực tiếp; output CLI vẫn giữ mặc định `mp4v` nếu không bật `--browser-compatible`. Giao diện giữ video kết quả trong phiên Streamlit hiện tại. Nếu refresh trang, có thể cần chạy lại.

**Giới hạn hiện tại:** AnimeGANv2 vẫn suy luận độc lập từng frame nên có thể flicker; output thực tế có màu ấm và mềm hơn input. FFmpeg đã được cài để giữ audio.

**Ghi chú về độ nét:** `load-size 512` cho hình mềm hơn do phải thu nhỏ rồi phóng lại. Đo trên mẫu thử: sharpness Laplacian trung bình tăng từ khoảng 18.4 ở 512 lên 25.5 ở 640 và 40.6 ở 768; đây là chỉ số tham khảo, chất lượng cảm nhận còn phụ thuộc checkpoint và video.

**Kết quả cần ghi:** đã tạo `reports/step0_input_metadata.json`, `reports/step0_processing.json`, `reports/step0_validation.json`, `reports/step0_frame_compare.jpg` và output `output/step0_smoke_test_anime.mp4`.

## Bước 1 — Chọn và ghi nhận video đầu vào

- [x] Chọn video ngắn 5–15 giây, có hình ảnh đủ sáng và chuyển động vừa phải: `istockphoto-1751442586-640_adpp_is.mp4`.
- [x] Đặt video tại `input/istockphoto-1751442586-640_adpp_is.mp4`.
- [x] Chạy:

  ```powershell
  python inspect_video.py --input input\istockphoto-1751442586-640_adpp_is.mp4 --report reports\step1_input_metadata.json
  ```

- [x] Chạy lệnh kiểm tra và ghi lại: tên file, độ phân giải `width × height`, FPS, số frame, thời lượng, codec, có audio hay không.

**Ghi chú:** video có thời lượng 13.88 giây, nằm trong khoảng test 5–15 giây. OpenCV đọc được codec H.264. Thống kê 347 frame cho thấy độ sáng trung bình 123.38/255, chênh lệch frame liên tiếp trung bình 3.80/255; đây là dấu hiệu video đủ sáng và chuyển động không quá gắt. File có track `soun`/`mp4a` nên được ghi nhận có audio AAC. Máy chưa có FFmpeg/FFprobe để xác nhận bằng lệnh chuyên dụng.

**Kết quả đã ghi:** `reports/step1_input_metadata.json`.

| Thuộc tính | Kết quả |
|---|---|
| Input | `input/istockphoto-1751442586-640_adpp_is.mp4` |
| Kích thước | 768 × 432 |
| FPS | 25.0 |
| Số frame | 347 |
| Thời lượng ước tính | 13.88 giây |
| Codec video | H.264 (`h264`) |
| Audio | Có, AAC (`mp4a`) |
| Đọc tuần tự | 347/347 frame |

## Bước 2 — Chọn và xác nhận AnimeGANv2 + PyTorch

- [x] Chọn implementation PyTorch `bryandlee/animegan2-pytorch`.
- [x] Chọn style tổng quát `paprika` cho bài test video.
- [x] Tải `paprika.pt` và đặt đúng tại `models/paprika.pt`.
- [x] Kiểm tra file tồn tại và dung lượng: 8,603,556 bytes.
- [x] Xác nhận framework bằng:

  ```powershell
  python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
  ```

**Ghi chú:** model dùng generator AnimeGANv2, input RGB `BCHW` trong `[-1,1]`, output qua `tanh`. Checkpoint đã nạp thành công vào `Generator`, model ở chế độ `eval`, có 2,143,552 tham số. Đã đổi sang `torch 2.11.0+cu128` và `torchvision 0.26.0+cu128`; CUDA runtime 12.8 nhận RTX 4060 Laptop GPU, `auto` chọn CUDA.

**Kết quả đã ghi:** `reports/step2_model_metadata.json`; style `paprika`, PyTorch `2.11.0+cu128`, device `cuda`, suy luận batch 4 thành công, VRAM cực đại khoảng 172.46 MB trong smoke test.

## Bước 3 — Cài môi trường

- [x] Tạo virtual environment `.venv` bằng `python -m venv .venv --system-site-packages`.
- [x] Cài/kiểm tra profile GPU bằng `.\\.venv\\Scripts\\python.exe -m pip install -r requirements-gpu.txt`.
- [x] Xác nhận Streamlit mở được bằng `python -m streamlit run app.py`; server trả HTTP 200.
- [x] Kiểm tra import OpenCV, NumPy, PyTorch và Streamlit.
- [x] Cài FFmpeg 9.0.2 và xác nhận `ffmpeg -version`, `ffprobe -version`.

**Ghi chú:** `.venv` dùng `--system-site-packages` để tái sử dụng bộ CUDA lớn đã cài trên máy; khi triển khai máy khác có thể dùng `requirements-gpu.txt` để cài lại. Pipeline không cần TensorFlow.

**Kết quả đã ghi:** Python 3.14.4, PyTorch `2.11.0+cu128`, CUDA 12.8, OpenCV 4.13.0, NumPy 2.4.4, Streamlit 1.64.0, FFmpeg 9.0.2.

## Bước 4 — Mở video và đọc frame tuần tự

- [x] Script mở video bằng `cv2.VideoCapture`.
- [x] Kiểm tra `frame_index` tăng liên tục từ 0 đến 346.
- [x] Xác nhận không có frame `None` giữa chừng.

**Ghi chú:** xử lý tuần tự giúp giữ đúng thứ tự frame và không cần nạp cả video vào RAM.

**Kết quả đã ghi:** `reports/step4_frame_read.json`; đọc 347/347 frame, chỉ số liên tục.

## Bước 5 — Tiền xử lý frame

- [x] Đổi frame BGR của OpenCV sang RGB cho AnimeGANv2.
- [x] Resize cạnh dài về `--load-size`, làm tròn kích thước về bội số của 32.
- [x] Chuyển `uint8 [0,255]` thành tensor `float32 [-1,1]` dạng `NCHW`.
- [x] Kiểm tra bằng report rằng kích thước model không làm biến dạng tỷ lệ khung hình.

**Ghi chú:** frame sau suy luận được đổi RGB → BGR và resize về đúng `(width, height)` gốc trước khi ghi video.

**Kết quả đã ghi:** `reports/step5_preprocessing.json`; `load-size=512`, tensor batch `[4,3,288,512]`, output trả về 768×432, giữ tỷ lệ 16:9.

## Bước 6 — Suy luận AnimeGANv2 từng frame

- [x] Load checkpoint `paprika.pt` ở chế độ inference.
- [x] Hỗ trợ `--device auto`, `cpu`, `cuda`.
- [x] Hỗ trợ `--batch-size`; batch smoke test 4 đã xử lý đủ 17/17 frame.
- [x] Chạy batch 4 trên CUDA thật với RTX 4060.
- [x] Chạy toàn bộ video với style Paprika.
- [x] Đánh giá checkpoint face: không cần vì video là cảnh ngựa/toàn cảnh, không phải video chủ yếu là chân dung.

**Ghi chú:** mỗi frame là một lần suy luận độc lập; đây là nguyên nhân có thể gây flicker theo thời gian.

**Kết quả đã ghi:** `reports/full_gpu_b8_processing.json`; device `cuda`, style Paprika, 347 frame, 13.564 giây, 0.0391 giây/frame, không có lỗi.

## Bước 7 — Ghi video output

- [x] Dùng `cv2.VideoWriter` với FPS đầu vào và kích thước đầu ra bằng input.
- [x] Xác nhận output mở được bằng `validate_video.py`.
- [x] Xác nhận codec ghi thực tế là MPEG-4/`mp4v` trong file MP4.

**Ghi chú:** codec ghi mặc định là `mp4v`; có thể đổi bằng `--codec` nếu backend máy hỗ trợ codec khác.

**Kết quả đã ghi:** `output/istockphoto-1751442586-640_adpp_is_anime_gpu_b8.mp4`, 768×432, 25 FPS, 347 frame, 13.88 giây.

## Bước 8 — Giữ audio bằng FFmpeg (nếu yêu cầu)

- [x] Chạy lại với `--audio-from input\\istockphoto-1751442586-640_adpp_is.mp4`.
- [x] Xác nhận FFmpeg ghép thành công và output có cả hình lẫn tiếng.
- [x] Audio được yêu cầu và đã giữ lại: AAC, 48 kHz, stereo, thời lượng 13.8667 giây.

**Ghi chú:** pipeline tạo video hình không tiếng trước, sau đó map video mới và audio gốc bằng FFmpeg. Nếu FFmpeg không có trong PATH, bước này sẽ dừng với thông báo hướng dẫn.

**Kết quả đã ghi:** `reports/media_streams.json`; FFmpeg 9.0.2 mux thành công. Hai file FFprobe raw cũng được giữ lại để đối chiếu.

## Bước 9 — Kiểm tra chất lượng và frame bị thiếu

- [x] Chạy `validate_video.py`.
- [x] Mở `reports/full_gpu_frame_compare.jpg`, so sánh frame đầu/giữa/cuối.
- [x] Kiểm tra màu, vùng đen, méo hình, crop ngoài ý muốn và frame bị thiếu.
- [x] Xác nhận frame count/FPS hợp lý so với input.

**Ghi chú:** validation đạt: output mở được, đọc 347/347 frame, cùng 768×432 và 25 FPS, không phải video đen. So sánh hình cho thấy style anime, không crop/méo nghiêm trọng; màu ấm hơn và ảnh mềm hơn input. `reports/full_gpu_temporal_grid.jpg` không cho thấy nhấp nháy đột ngột trong 8 frame liên tiếp.

**Kết quả đã ghi:** `reports/full_gpu_b8_validation.json`, `reports/full_gpu_b8_frame_compare.jpg`, `reports/full_gpu_temporal_grid.jpg`.

## Bước 10 — Đo thời gian và kiểm tra flicker

- [x] Ghi `processing_seconds` và `seconds_per_frame` từ report.
- [x] Xem chuỗi frame liên tiếp bằng temporal grid.
- [x] Ghi nhận flicker: chưa thấy nhấp nháy đột ngột trong mẫu kiểm tra; vẫn cần xem toàn video khi đánh giá cuối.

**Ghi chú:** flicker là hạn chế dự kiến của việc chạy AnimeGANv2 độc lập trên từng frame. Bài test cơ bản chỉ cần ghi nhận; không tự coi đó là lỗi pipeline.

**Kết quả đã ghi:** CUDA batch 8 xử lý 347 frame trong 13.564 giây, 0.0391 giây/frame. Mức flicker trong mẫu quan sát: `không thấy rõ`; chi tiết tại `reports/full_gpu_flicker.json`. Vẫn ghi nhận rủi ro flicker do model suy luận độc lập từng frame.

## Tiêu chí đạt cuối cùng

- [x] Video output phát được và decode hết frame.
- [x] FPS, độ phân giải và thứ tự frame hợp lý so với input.
- [x] Có phong cách anime, không đen toàn bộ, không sai màu hoặc biến dạng nghiêm trọng.
- [x] Audio được giữ lại theo yêu cầu.
- [x] Report có metadata, thời gian xử lý và nhận xét flicker.

## Nhật ký kết quả

| Mục | Giá trị |
|---|---|
| Input | `input/istockphoto-1751442586-640_adpp_is.mp4` |
| Model/style | AnimeGANv2 / Paprika / `models/paprika.pt` |
| Device | CUDA / RTX 4060 Laptop GPU / PyTorch `2.11.0+cu128` |
| Input resolution/FPS/frames | 768x432 / 25 FPS / 347 frame |
| Output resolution/FPS/frames | 768x432 / 25 FPS / 347 frame |
| Processing time | 13.564 giây, batch 8, `load-size 512` |
| Audio | Đã giữ AAC, 48 kHz, stereo |
| Flicker | Chưa thấy rõ trong temporal grid 8 frame; vẫn có rủi ro do xử lý độc lập |
| Kết luận | Đạt; output GPU có audio, đủ frame, đúng FPS/độ phân giải |

## Cập nhật batch/GPU — 27/09/2026

- [x] Bổ sung batch inference: đọc nhiều frame mỗi lượt nhưng ghi lại tuần tự.
- [x] Giữ nguyên FPS, độ phân giải, thứ tự frame và số frame output.
- [x] Thêm lựa chọn `Số frame xử lý mỗi lượt` trên giao diện.
- [x] Report ghi `device_info`, `batch_size` và `batch_count`.
- [x] Smoke test batch 4: 17/17 frame, 96x64, 8 FPS, 2.125 giây, validation đạt.
- [x] GPU thực tế: đã cài `torch 2.11.0+cu128`, CUDA 12.8; PyTorch nhận NVIDIA RTX 4060 Laptop GPU và `torch.cuda.is_available() = True`.
- [x] Benchmark hiệu năng cùng cấu hình: CPU batch 4 = 24.00 giây/64 frame; CUDA batch 4 = 4.14 giây/64 frame; CUDA batch 8 = 2.96 giây/64 frame; Auto = CUDA.
- [x] Chạy toàn bộ video thật bằng CUDA batch 8: 347 frame trong 13.564 giây, có audio.
- [x] Bổ sung player xem trực tiếp video gốc sau upload và output sau xử lý; output giao diện là H.264/yuv420p để Chrome phát được.

**Kết luận:** không nên chia hoặc bỏ frame theo FPS. FPS chỉ quyết định tốc độ phát; batch là cách tăng tốc suy luận. GPU đã được xác nhận bằng suy luận AnimeGANv2 batch 4 thật trên `cuda:0`. Batch không tự khử flicker theo thời gian.
