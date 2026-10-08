# Chạy Lab 22 trên Colab T4 và lấy bài nộp

NB0 đã có lời giải; các kết quả NB1–NB4 phải được sinh bằng lần chạy thật trên GPU.
Notebook mặc định chỉ chạy phần bắt buộc. Không cần khoá API.

## 1. Mở đúng notebook

1. Vào Google Colab, chọn **File → Upload notebook**.
2. Chọn file **`colab/Lab22_DPO_T4.ipynb` trong repo hiện tại**.
3. Chọn **Runtime → Change runtime type → T4 GPU**.
4. Giữ `RUN_BONUS = False` trong cell cấu hình.
5. Chọn **Runtime → Run all**. Thời gian lab ước tính 1,5–2 giờ, tuỳ phiên GPU.

Notebook tự ghi mã nguồn vào `/content/lab22`; bạn không cần clone repo trên Colab.
Nếu Colab yêu cầu khởi động lại sau khi cài thư viện, khởi động lại rồi chạy từ đầu.
Nếu hết VRAM, bật dòng `os.environ["MAX_LEN"] = "512"` ở cell cấu hình,
rồi khởi động lại runtime và chạy từ đầu để dữ liệu và mô hình dùng cùng cấu hình.
Giữ 800 cặp train, 100 cặp held-out và ít nhất 50 câu hỏi được chấm cho bài chính thức.

## 2. Theo dõi các mốc

| Phần | Bằng chứng cần thấy |
|---|---|
| NB0 | Khớp loss tham chiếu; loss lúc policy = reference bằng khoảng 0,693 |
| NB1 | Biểu đồ loss; lưu adapter và `models/sft-merged/` |
| NB2 | Không trùng câu hỏi; ba cặp mẫu; tỉ lệ chosen dài hơn rejected |
| NB3 | Reward chosen/rejected trên cả train và held-out; diagnosis và metrics |
| NB4 | 8 câu cố định + ít nhất 50 câu held-out; summary của hội đồng giám khảo |

Đọc đủ ba cặp NB2 và ghi nhận xét riêng cho từng cặp: chosen có chính xác,
đầy đủ, đúng yêu cầu hơn không, hay chỉ dài hơn? Không coi nhãn tự động luôn đúng.
Trong NB3, vài phút tính reference log-prob trước thanh tiến trình là bình thường.

## 3. Tải kết quả trước khi mất phiên

Cell cuối tự tạo và tải **`lab22-submission.zip`**. Nếu trình duyệt chặn tải,
tải thủ công `/content/lab22/lab22-submission.zip` trong bảng Files.
ZIP giữ cấu trúc thư mục gồm ảnh, kết quả đánh giá, metrics, dữ liệu chia,
config SFT nhỏ và mã nguồn; không chứa trọng số mô hình hay `.env`.

Tải thêm **File → Download → Download .ipynb** để giữ output của notebook Colab.
ZIP không tự chứa notebook Colab đang mở. Lưu notebook đã chạy ở
`colab/Lab22_DPO_T4_executed.ipynb` trong repo; giữ nguyên notebook nguồn.

Giải nén ZIP vào thư mục gốc repo. Khi có file trùng, giữ mã nguồn mới nhất
và bài REFLECTION bạn đã viết. Không sửa đường dẫn reference trong adapter config:
verifier kiểm tra bằng đường dẫn gốc cùng SHA-256 của config SFT khi chuyển từ Colab về máy.
Config nhỏ chỉ là bằng chứng nguồn mô hình; muốn tiếp tục huấn luyện cần trọng số SFT còn trong phiên.

## 4. Điền REFLECTION từ các file

| Nội dung | Nguồn |
|---|---|
| GPU, dữ liệu và loss SFT | `adapters/sft-mini/sft_metrics.json` |
| Số cặp train/eval, β, lr, epoch | `adapters/dpo/dpo_metrics.json` |
| Tỉ lệ chosen dài hơn rejected | `data/pref/stats.json`: `chosen_longer_frac` |
| Thời gian NB3 | `dpo_metrics.json`: `train_runtime_seconds` (chia 60 ra phút) |
| VRAM cao nhất NB3 | `dpo_metrics.json`: `peak_vram_gb` (bộ nhớ tensor CUDA được cấp phát) |
| Đường reward và chẩn đoán | `train_reward_history`, `eval_reward_history`, `diagnosis` và ảnh NB3 |
| Win rate, CI, sanity, thiên vị độ dài | `data/eval/judge_summary.json` |
| Hai ví dụ hữu ích/an toàn | `data/eval/side_by_side.jsonl` và `judge_results_rm.json` |

Độ dài trung bình SFT/DPO được tính bằng trung bình `len(row["sft"])` và
`len(row["dpo"])` trên cùng các dòng trong `side_by_side.jsonl`.
Phần §3 cần ít nhất 100 từ; §6 cần ít nhất 150 từ. Không dùng kết quả NB0 đồ chơi
để thay cho kết quả huấn luyện NB3. CI chứa 0,5 hoặc DPO không cải thiện vẫn là
kết quả hợp lệ nếu phân tích trung thực. Các mục bonus có thể để trống nếu không chạy.

## 5. Kiểm tra và nộp

Trên Colab, sau khi điền REFLECTION, chạy một cell:

```python
!python /content/lab22/scripts/verify.py
```

Trên Windows có thể chạy `python -X utf8 scripts/verify.py` thay cho `make verify`.
Chỉ khi toàn bộ bằng chứng đã có và REFLECTION đã điền, verifier mới báo đạt.
Nếu sửa REFLECTION trên Colab, chạy lại cell tải ZIP để lấy bản mới nhất.
Commit notebook đã chạy, ảnh, JSON, dữ liệu chia, config nhỏ và REFLECTION
vào repo public của bạn; nộp URL repo vào LMS.

Khi gửi kết quả để được hỗ trợ hoàn thiện phản tư, cung cấp ZIP và notebook
đã chạy, kèm nhận xét của bạn về ba cặp NB2 và lựa chọn quan trọng cho §6.
