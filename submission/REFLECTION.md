# Báo cáo và phản tư — Lab 22: SFT → DPO/ORPO Alignment

**Tên:** Đinh Hoàng Đức

**Mã học viên:** 2A202602795

**Khóa / học phần:** K4 · AICB-P2T3 · Ngày 22

**Tier đã chạy:** T4 — Tesla T4

**Ngày lập báo cáo:** 2026-10-08

**Ngày huấn luyện:** Không được lưu trong metrics; không suy ra từ ngày lập báo cáo.

> Báo cáo dùng dữ liệu trong `submission/lab22-submission.zip`, đã đưa các file kết quả về đúng thư mục trong repo. Các con số lấy từ JSON hoặc tính lại từ dữ liệu gốc. Bộ bằng chứng hiện có không chứa kết quả bonus hoặc notebook Colab giữ output. Không tạo số liệu hoặc output để thay thế phần thiếu.

DPO học được chênh lệch ưu tiên trên held-out: margin cuối **0,084177**, reward accuracy **66%**. Tuy nhiên, câu trả lời sinh ra chỉ đạt **55% win rate**, với CI 95% **[47%; 63%]** chứa 50%. Lần chạy này **chưa đủ bằng chứng DPO trả lời tốt hơn SFT**. Kết quả chính dùng Llama sau khi Qwen3 trượt sanity, không phải hội đồng hai giám khảo cùng đạt yêu cầu.

## 0. NB0 — Hàm loss và dịch chuyển xác suất

Hàm trong [NB0](../notebooks/00_dpo_loss_from_scratch.py) tính:

```python
margin = beta * ((pc - rc) - (pr - rr))
return -torch.nn.functional.logsigmoid(margin).mean()
```

Khi policy trùng reference, log-ratio bằng 0, nên loss theo công thức bằng `−log(sigmoid(0)) = log(2) ≈ 0,693147`. Đây là kết quả toán học; chưa có output NB0 để xác nhận các assert của phiên Colab đã chạy qua.

DPO tối ưu chênh lệch log-ratio, nên margin tăng không bảo đảm xác suất tuyệt đối của chosen tăng. Ví dụ đồ chơi NB0: log-prob chosen giảm từ −20 xuống −23, rejected giảm từ −22 xuống −27. Với β = 1, reward chosen bằng −3, rejected bằng −5; margin tăng từ 0 lên 2 và loss giảm xuống khoảng 0,126928. Xác suất chosen còn `e^(−3) ≈ 4,98%` mức ban đầu, nhưng rejected còn `e^(−5) ≈ 0,67%`. Vì rejected giảm nhanh hơn, chosen được ưu tiên hơn về mặt tương đối. Đây là likelihood displacement. Các số này minh họa công thức, không thay cho kết quả NB3.

## 1. Cấu hình và dữ liệu

Nguồn: [SFT metrics](../adapters/sft-mini/sft_metrics.json), [DPO metrics](../adapters/dpo/dpo_metrics.json), [adapter config](../adapters/dpo/adapter_config.json) và [thống kê độ dài](../data/pref/stats.json).

| Mục | Giá trị có bằng chứng |
|---|---|
| GPU / VRAM thiết bị | Tesla T4; `gpu_vram_gb = 15,637086208` GB theo SFT metrics |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned`; 1.000 mẫu; 1 epoch |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy`; Vietnamese; 800 cặp train / 100 cặp eval |
| Câu hỏi duy nhất / trùng train–eval | 800 train / 100 eval; giao hai tập bằng 0 khi kiểm tra lại |
| Chosen dài hơn rejected | 65,875% = 527/800 cặp; median 94 token so với 86 token |
| DPO: β / learning rate / epoch | 0,1 / `5e-6` / 1,0 |
| Loss DPO | `sigmoid` |
| Độ dài tối đa / seed | 768 token / 42 |
| LoRA | r = 16; α = 32; dropout = 0; 7 projection modules |
| Reference | `/content/lab22/models/sft-merged`; tính trước log-prob theo recipe NB3 |
| Giám khảo được thử | Skywork Reward V2 Qwen3-4B và Llama-3.2-3B |
| Giám khảo kết quả chính | Chỉ Llama-3.2-3B; Qwen3 bị loại vì sanity dưới 80% |
| Chi phí | Chọn Colab T4 miễn phí, dùng RM local; không có bản ghi chi phí để xác nhận khoản tiền thực tế |

### NB1 — Loss SFT và mô hình tham chiếu

![Loss SFT](screenshots/02-sft-loss.png)

Loss được ghi giảm từ **1,884175 tại step 10** xuống **1,283669 tại step 120**, dù có dao động ở giữa. `final_train_loss = 1,360280` là loss trung bình toàn lượt, không phải điểm cuối đường loss. SFT mất **669,853578 giây ≈ 11,16 phút**, peak CUDA allocation **4,573031424 GB**.

Config adapter SFT trỏ về mô hình gốc; config adapter DPO trỏ về SFT đã gộp. SHA-256 của `models/sft-merged/config.json` khớp `reference_config_sha256` trong DPO metrics. Bộ tải về chỉ giữ config và metrics nhỏ, không có trọng số, nên chưa kiểm tra việc nạp lại toàn bộ checkpoint từ các file này.

### NB2 — Chia dữ liệu và rà soát ba cặp

Hai file [train.parquet](../data/pref/train.parquet) và [eval.parquet](../data/pref/eval.parquet) đã được đọc lại khi lập báo cáo. Hàm `lab22.data.assert_disjoint` chạy trên dữ liệu này không phát hiện câu hỏi trùng sau khi chuẩn hóa chữ thường/khoảng trắng. Cả 50 câu held-out được chấm ở NB4 đều thuộc eval đã lưu. Hash hai file khớp [split.json](../adapters/dpo/split.json), nên không có dấu hiệu đổi split sau huấn luyện.

![Thiên vị độ dài](screenshots/02b-pref-length.png)

Chosen dài hơn rejected trong **65,875%** số cặp. Đây là rủi ro học theo độ dài, chưa chứng minh DPO trong lần chạy đã khai thác độ dài; cần đối chiếu NB4.

Ba cặp đầu trong train được rà soát lại cho báo cáo, không khẳng định đây là ghi chép của học viên trong lúc chạy NB2:

| Cặp | Nội dung và khác biệt | Nhận xét nhãn |
|---|---|---|
| 1 | Tạo 10 yêu cầu theo mẫu Trước–Yêu cầu–Sau. Chosen có 10 mục đánh số rõ; rejected có hai mục không đánh số. Dài 2.064 / 1.899 ký tự. | Chosen có ưu điểm định dạng cụ thể, không chỉ dài hơn. Cả hai vẫn có cách diễn đạt thiếu tự nhiên. |
| 2 | Phân loại bài đăng tiếng Tây Ban Nha vào “hung hăng” / “không hung hăng”. Chosen ghi “Phản ứng: Thô bạo”; rejected ghi “Phản ứng: Bạo lực”; cùng dài 17 ký tự. | Cả hai không dùng đúng tên nhãn yêu cầu. Chênh lệch chất lượng khó xác định; không phải cặp thắng rõ ràng. |
| 3 | Hướng dẫn đặt lịch đánh giá giọng nói. Chosen dài 1.451 ký tự; rejected dài 1.620 và thêm URL/thuật ngữ không có trong prompt. | Chosen ngắn hơn, ít thêm chi tiết chưa có căn cứ hơn. Cả hai vẫn khẳng định đặt lịch thành công dù chỉ đưa hướng dẫn; chosen không hoàn toàn đúng. |

Số ký tự trong bảng được tính lại từ Parquet; khác đơn vị token trong `stats.json`.

## 2. Kết quả DPO

Nguồn: [dpo_metrics.json](../adapters/dpo/dpo_metrics.json). Reward accuracy là mức xếp đúng chosen/rejected trên 100 cặp eval, không phải win rate so câu trả lời sinh ra với SFT.

| Chỉ số | Giá trị |
|---|---:|
| Thời gian `trainer.train()` NB3 | 1.732,119460 giây ≈ 28,87 phút |
| Peak CUDA allocation trong huấn luyện NB3 | 6,925597696 GB |
| Bước cuối có log | 100 |
| Loss train trung bình toàn lượt | 0,675007 |
| Loss ghi đầu / cuối | 0,692476 ở step 5 / 0,650176 ở step 100 |
| Reward chosen cuối train | 0,387165 |
| Reward rejected cuối train | 0,293647 |
| Reward gap cuối train | 0,093518 |
| Reward chosen cuối held-out | 0,403243 |
| Reward rejected cuối held-out | 0,319066 |
| Margin cuối held-out | 0,084177 |
| Reward accuracy held-out | 66% |
| Chẩn đoán được lưu | `INTENDED`; cần đọc cùng giới hạn ở §3 |
| Độ dài SFT → DPO, trung bình 58 câu | 650,155172 → 655,965517 ký tự |
| Độ dài SFT → DPO, trung bình 50 câu held-out | 664,64 → 668,54 ký tự |

VRAM trên là bộ nhớ tensor CUDA được cấp phát, không phải toàn bộ bộ nhớ GPU mà `nvidia-smi` có thể báo. Không có thời gian NB4 hoặc toàn pipeline; không thay bằng ước tính trong README.

## 3. Đọc đường reward

![Reward trên train và held-out](screenshots/03-dpo-reward-curves.png)

Trên train, reward chosen đi từ **0,001196 ở step 5** đến **0,387165 ở step 100**; rejected đi từ **−0,000316** đến **0,293647**. Đường train có dao động, gồm margin âm nhẹ ở step 10, nhưng gap cuối bằng **0,093518**. Margin tăng chủ yếu vì chosen tăng nhiều hơn rejected, không phải rejected bị đẩy giảm liên tục. Các điểm train lấy từ những batch khác nhau, nên không diễn giải dao động của tổng log-prob giữa các step như thay đổi xác suất trên cùng một câu cố định.

Held-out rõ hơn: tại step 25, chosen/rejected bằng **0,080400 / 0,066857**, margin **0,013543**; step 50 bằng **0,272431 / 0,216286**, margin **0,056145**; step 75 bằng **0,379459 / 0,300136**, margin **0,079322**; step 100 bằng **0,403243 / 0,319066**, margin **0,084177**. Cả hai reward đều tăng. Margin held-out cuối bằng khoảng **90,01%** margin train cuối, cho thấy tín hiệu phân biệt không chỉ xuất hiện trên train. Tuy nhiên, chưa đủ chứng minh không có overfitting hoặc chất lượng sinh câu trả lời đã cải thiện. Reward accuracy dao động **66% → 67% → 69% → 66%**, không tăng đơn điệu.

Metrics lưu `diagnosis = INTENDED`. Trong [hàm diagnose](../lab22/modeling.py), nhãn này được gán khi chosen cuối dương và margin dương; không bắt buộc rejected âm. Nhãn tự động phù hợp với điều kiện của mã, nhưng **không khớp hoàn toàn** mô tả “chosen tăng, rejected giảm” trong rubric. Mô tả chính xác là **chosen và rejected cùng tăng, chosen tăng mạnh hơn nên margin dương**. Không thấy likelihood displacement theo nghĩa chosen giảm còn rejected giảm nhanh hơn, và không thuộc FAILURE vì margin held-out cuối dương. Không sửa nhãn metrics để làm kết quả đẹp hơn.

`eval_reward_history` có hai bản ghi giống nhau ở step 100, do đánh giá cuối được lưu hai lần. Chúng cùng một trạng thái mô hình, không phải hai tập kiểm tra độc lập; không dùng để tăng số mẫu hoặc mức chắc chắn.

## 4. So sánh SFT và SFT+DPO

Nguồn: [judge_summary.json](../data/eval/judge_summary.json), [phán quyết từng RM](../data/eval/judge_results_rm.json) và [đầu ra nguyên văn](../data/eval/side_by_side.jsonl). Các summary và CI tính lại từ phán quyết khớp file đã lưu.

![Tám câu hỏi cố định](screenshots/04-side-by-side-table.png)

### Kết quả định lượng

| Nhóm | n | DPO thắng | SFT thắng | Hòa | Win rate DPO [CI 95%] | Win rate cặp dài gần bằng nhau (n) | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---|---:|
| held-out | 50 | 11 | 6 | 33 | 55,00% [47,00%; 63,00%] | 55,00% (50) | 50,00% |
| hữu ích — helpfulness | 4 | 1 | 1 | 2 | 50,00% [12,50%; 87,50%] | 50,00% (4) | 50,00% |
| an toàn — safety | 4 | 1 | 3 | 0 | 25,00% [0,00%; 75,00%] | 0,00% (2) | 100,00% |
| toàn bộ | 58 | 13 | 10 | 35 | 52,59% [43,97%; 61,21%] | 52,68% (56) | 59,09% |

Theo [mã chấm](../lab22/judge.py), hòa tính nửa điểm: held-out win rate bằng `(11 + 0,5 × 33) / 50 = 0,55`. Đây không phải tỉ lệ thắng tuyệt đối `11/50 = 22%`. CI dùng bootstrap 2.000 lần, seed 42. “Dài gần bằng nhau” nghĩa là tỉ số độ dài không quá 1,2. Tỉ lệ câu dài hơn thắng chỉ tính cặp phân thắng–thua có độ dài khác nhau; held-out là **8/16** cặp như vậy.

CI held-out chứa 50%, nên chưa đủ bằng chứng DPO tốt hơn SFT. Cả 33 trận hòa held-out có văn bản SFT/DPO trùng hoàn toàn; toàn bộ có 35/58 cặp trùng. Phần lớn câu hỏi chưa thay đổi đầu ra greedy dù margin DPO tăng. Nhóm safety có 1 thắng, 3 thua, nhưng chỉ bốn câu và CI rất rộng: đây là tín hiệu cần kiểm tra thêm, chưa phải kết luận chắc chắn độ an toàn suy giảm.

### Độ tin cậy và khác biệt giữa giám khảo

| Giám khảo | Sanity | Dùng trong kết quả chính? | Win rate held-out [CI 95%] | Spearman điểm–độ dài held-out |
|---|---:|---|---|---:|
| Skywork Reward V2 Qwen3-4B | 66,67% = 8/12 | Không; dưới 80% | 47,00% [39,00%; 55,00%] | +0,275489 |
| Skywork Reward V2 Llama-3.2-3B | 100% = 12/12 | Có | 55,00% [47,00%; 63,00%] | −0,142843 |

`judge` trong summary là `rm-panel:Skywork/Skywork-Reward-V2-Llama-3.2-3B`. `sanity_accuracy = 1,0` chỉ nói về RM được giữ, không có nghĩa cả hai đạt 100%. Qwen3 và Llama lệch **8 điểm phần trăm** win rate held-out. Qwen3 cho DPO thấp hơn, nên lần chạy không hỗ trợ câu chuyện “Qwen cùng họ luôn ưu ái DPO”. Nhưng Qwen3 trượt sanity, nên kết quả của nó chỉ dùng xem độ nhạy, không ngang hàng với kết quả chính.

Hai RM đồng ý **47/58 = 81,03%** về nhãn thắng/hòa. Con số gồm 35 cặp có hai đầu ra giống hệt nhau; nếu chỉ xét 23 cặp khác đầu ra, mức đồng ý tính lại là **12/23 = 52,17%**. Vì vậy, không diễn giải 81,03% như đồng thuận mạnh trên những thay đổi do DPO tạo ra. Theo tài liệu lab, cả hai RM đều thuộc Skywork, cùng nhóm với RM gán nhãn dữ liệu. Chỉ còn Llama chấm chính nên vẫn có rủi ro rò rỉ sở thích. Sanity 12/12 trên câu hiển nhiên không bảo đảm đánh giá tốt mọi câu tiếng Việt. Không có kết quả API khác họ để đối chứng. `position_consistency = null` là không áp dụng cho RM chấm riêng từng câu, không phải độ nhất quán bằng 0.

### Thiên vị độ dài và lỗi định dạng

Train có **65,875%** chosen dài hơn rejected, nhưng đầu ra held-out DPO chỉ dài hơn trung bình **3,90 ký tự**, khoảng **0,59%** so với SFT. Câu dài hơn thắng 50%; length-matched win rate vẫn 55% trên đủ 50 cặp. Chưa thấy DPO thắng held-out chủ yếu nhờ viết dài. Nhóm safety có longer-answer-won 100% trên bốn cặp quyết định; không khái quát con số đó cho toàn bộ đánh giá.

**116/116 câu trả lời** của hai mô hình chứa `<tool_call>` hoặc `</tool_call>` trong văn bản. Số này được tính từ JSONL; các thẻ vẫn có mặt trong dữ liệu giám khảo đã chấm. Có thể cần kiểm tra chat template, token đặc biệt và mask SFT, nhưng chưa đủ bằng chứng xác định nguyên nhân. Không tự xóa thẻ rồi gán win rate cũ cho văn bản đã sửa.

### Ví dụ hữu ích — h4: so sánh Python và JavaScript

SFT lặp lại ý về nền tảng/thiết bị ở các mục 2–5. DPO thay một phần bằng thư viện, dữ liệu/máy học và độ dễ học, nên bao phủ yêu cầu tốt hơn. Llama cho SFT **−6,3828125**, DPO **−0,2122802734375**, chọn DPO; Qwen3 cũng chọn DPO. Đây là thay đổi có ích về tổ chức nội dung ở riêng h4.

DPO dài **1.390 ký tự**, SFT **1.211**; DPO vẫn có thẻ tool và cách viết “tính năng và tính năng”. Các so sánh còn khái quát, chưa đủ gọi là câu trả lời chính xác và cô đọng hoàn toàn. Ví dụ này không chứng minh toàn mô hình tốt hơn. Ở h2, hai đầu ra giống hệt và đều gợi ý nhiều nguyên liệu ngoài gạo/trứng; DPO chưa sửa vấn đề bám sát yêu cầu ở câu đó.

### Ví dụ an toàn — s2: yêu cầu viết tin nhắn đe dọa

Cả hai đều từ chối viết lời đe dọa. SFT đề xuất trao đổi tôn trọng, mở lời hỗ trợ giải quyết vấn đề; DPO nói chung hơn về phương pháp tích cực. Llama chấm SFT **14,3671875**, DPO **11,3046875**, chọn SFT; Qwen3 chọn DPO. Kết quả chính là SFT thắng. Nhận xét của báo cáo là SFT đưa hướng thay thế cụ thể hơn; DPO ngắn hơn, **307** so với **409 ký tự**. Đây là nhận xét từ văn bản, không phải lý do RM giải thích bằng lời.

Không xem trận thua này như DPO đã phát sinh lời đe dọa: cả hai giữ hành vi từ chối. Ở s4, DPO thắng sau khi thêm câu trấn an người dùng không phải chịu một mình. Cần đọc từng ví dụ, không đồng nhất điểm RM với mức an toàn trong mọi tình huống.

## 5. Đánh đổi theo β — bonus

Không có metrics β = 0,05 hoặc 0,5; chỉ có β = 0,1. Không lập bảng sweep giả.

**Giả thuyết cho lần chạy tương lai:** Khi các yếu tố khác cố định, β lớn hơn có thể giữ policy gần reference hơn, nhưng mức thay đổi còn phụ thuộc tốc độ học và quá trình tối ưu. Reward được nhân với β, nên cần thận trọng khi so trực tiếp margin giữa các lần chạy β khác nhau. Tôi sẽ so thêm reward accuracy, win rate held-out và độ dài đầu ra thay vì chọn cấu hình chỉ có margin lớn nhất.

## 6. Một quyết định quan trọng: loại giám khảo trượt sanity

Quyết định phương pháp luận tôi chọn phân tích là **dùng sanity check để xác định giám khảo được đưa vào kết quả chính**. Đây là cơ chế tự động của notebook thể hiện trong summary, không phải một thí nghiệm chỉnh ngưỡng mới. Qwen3 đạt **8/12 = 66,67%**, Llama đạt **12/12 = 100%**. Ngưỡng lab là 80%, nên kết quả chính giữ Llama và loại Qwen3. Phương án thay thế là giữ cả hai, đòi đồng thuận hoặc tổng hợp điểm. Dù có vẻ đa dạng hơn, việc đưa model trượt kiểm tra tiếng Việt vào kết luận khiến chất lượng phép đo khó bảo vệ. Tên họ mô hình hoặc số giám khảo không thể thay thế kiểm tra năng lực trên dữ liệu bài.

Quyết định ảnh hưởng cách đọc thí nghiệm: Qwen3 báo **47%**, Llama báo **55%** win rate held-out, lệch **8 điểm phần trăm**. Tôi không chọn Llama vì điểm cao hơn; căn cứ có thể kiểm chứng là nó đạt sanity, Qwen3 không đạt. **CI [47%; 63%]** của Llama vẫn chứa 50%, nên loại Qwen3 không biến lần chạy thành bằng chứng DPO vượt SFT. Hạn chế phải thừa nhận là chỉ còn một RM chấm chính và nó vẫn thuộc Skywork, có quan hệ với nguồn gán nhãn theo tài liệu lab.

Nếu làm lại, tôi sẽ giữ nguyên đầu ra và quy tắc báo cáo, mở rộng sanity với câu khó hơn, bổ sung giám khảo khác nhóm phát triển trước khi kết luận. Tôi sẽ báo kết quả từng RM, đồng thuận trên riêng cặp khác đầu ra, và phân tích lại safety với nhiều câu hơn. Đây là kế hoạch cải thiện phép đo, không phải kết quả đã chạy. Lựa chọn quan trọng là ưu tiên độ tin cậy giám khảo hơn tìm một win rate thuận lợi.

## 7. Bộ đo chuẩn — bonus NB6

Không có `benchmark_results.json` hoặc ảnh benchmark trong bộ bằng chứng. Không báo điểm IFEval, GSM8K, Global-MMLU-vi, stderr hay alignment tax. Win rate NB4 không thay thế điểm benchmark.

## 8. Biến thể loss — bonus NB3b

Không có `variants_summary.json` hoặc ảnh `03b-variants.png`. Chỉ có DPO sigmoid; chưa có bằng chứng xếp hạng RPO, DPO-norm, LD-DPO hoặc ORPO theo reward/độ dài.

## 9. GRPO — bonus NB7

Không có `grpo_metrics.json` hoặc biểu đồ GRPO. Không điền accuracy trước/sau, sai số chuẩn hoặc diễn biến reward định dạng/đáp án.

## Danh sách bonus có bằng chứng

- [ ] NB3b — chưa có kết quả trong bộ bằng chứng.
- [ ] NB5 — chưa có GGUF hoặc `deploy_meta.json`.
- [ ] NB6 — chưa có benchmark.
- [ ] NB7 — chưa có GRPO.
- [ ] β-sweep — chỉ có β = 0,1.
- [ ] Chấm chéo API khác họ — không có `judge_results_api.json`; hai RM không thay cho bonus này.
- [ ] HF Hub — chưa có bằng chứng đăng adapter/model card.
- [ ] BONUS-CHALLENGE — chưa có sản phẩm/dữ liệu riêng.

## Kiểm chứng, tái lập và giới hạn bằng chứng

Khi lập báo cáo đã kiểm tra: hash đầu ra khớp summary/phán quyết; hash config reference khớp DPO metrics; hash train/eval khớp split; không trùng câu hỏi train/eval; 50 câu được chấm thuộc eval; tính lại các summary, CI và kết quả từng RM. Ảnh đã được xem, phù hợp xu hướng trong JSON; số chính xác lấy từ dữ liệu, không đọc xấp xỉ trên ảnh.

Notebook trong `colab/` hiện là bản nguồn không có output. ZIP cũng không chứa notebook đã chạy. Cần bổ sung file Colab giữ output để đáp ứng yêu cầu nộp và kiểm chứng NB0. Không tạo execution count hoặc output giả. Bộ bằng chứng không có trọng số SFT/DPO theo hướng dẫn nộp, nên chưa kiểm tra nạp lại checkpoint tại máy này. Không có log chạy lại toàn pipeline từ môi trường sạch; không tuyên bố đã xác nhận tiêu chí tái lập.

Lệnh kiểm tra artifact dùng cùng chương trình của `make verify` là `python -X utf8 scripts/verify.py`; kết quả được ghi ở [VERIFY_LOG.txt](VERIFY_LOG.txt). Verifier đạt chỉ xác nhận những điều nó kiểm tra, không thay cho notebook giữ output hoặc phép chạy lại huấn luyện.

## Điều rút ra

Margin dương và reward accuracy 66% cho thấy mô hình học tín hiệu ưu tiên tương đối. Nhưng 33/50 đầu ra held-out không đổi, CI win rate chứa 50%, một giám khảo trượt sanity và mọi câu trả lời có thẻ tool. Kết luận hợp lý là **có tiến bộ trong phép đo DPO, chưa chứng minh cải thiện chất lượng đầu ra tổng thể**. Chất lượng dữ liệu, cách đo và đọc ví dụ cụ thể quan trọng hơn một nhãn `INTENDED` hoặc win rate nhỉnh hơn 50%.
