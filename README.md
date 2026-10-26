# PI–NAFC Extra Test

Thí nghiệm bổ sung trên **Phi-4-mini-instruct (3.8B)** và toán Olympic trong **NuminaMath-1.5**, nối tiếp thiết lập GRPO + PI của [DATN-FPTU](https://github.com/HoangLeminh17/DATN-FPTU). Mục tiêu là kiểm tra khả năng chống quên của NAFC trên model và dữ liệu khác, đồng thời so sánh với replay cùng số lượng và KL.

**Đã chạy xong seed 1.** Notebook được chuẩn hóa để chạy từ đầu với **3 seed: 1, 2, 3**; chưa có kết luận tổng hợp ba seed.

## Thiết kế thí nghiệm

| Thành phần | Cấu hình |
|---|---|
| Model | `microsoft/Phi-4-mini-instruct`, checkpoint cố định |
| Dữ liệu | `AI-MO/NuminaMath-1.5`, nguồn `olympiads` và `olympiads_ref` |
| Train / validation / test | 2.000 / 256 / 2.000 câu cố định, tách riêng |
| Khởi tạo | Mỗi nhánh và mỗi seed bắt đầu từ cùng model gốc |
| Huấn luyện | 300 bước/nhánh; 8 câu × 8 đáp án/bước; LoRA rank 32; learning rate 1e-5 |
| PI | Mục tiêu **400,86 token**; eta 0,05; kp 4; trần lambda 3 |
| Giới hạn sinh | Completion tối đa 1.536 token; tổng context 1.538 token, gồm cả đề |
| Anchor | 128 câu model gốc đúng ≥6/8; 64 câu sửa, 64 câu đo; audit mỗi 25 bước, 4 mẫu/câu |
| Đánh giá | Validation tại bước 50/100/150/200/250; cùng 2.000 câu test tại bước 300, 4 mẫu/câu |

Bốn nhánh: **GRPO + PI**, **GRPO + PI + NAFC**, **GRPO + PI + quota replay**, **GRPO + PI + KL**. Quota dùng đúng số nhóm replay từng bước của NAFC trong cùng seed, nhưng chọn anchor vòng tròn; đây là đối chứng chẩn đoán. KL dùng beta = 0,04. Các nhánh giữ cùng cấu hình PI; PI là mục tiêu độ dài trung bình nên độ dài thực tế có thể khác nhau.

Chuẩn bị dữ liệu, chọn anchor và đo tham chiếu được làm một lần, dùng chung cho ba seed. Không chọn seed hoặc checkpoint tốt nhất để báo cáo.

## Kết quả seed 1 — bước 300

| Nhánh | Accuracy (%) | Số câu quên | Câu mới làm tốt | Token trung bình |
|---|---:|---:|---:|---:|
| GRPO + PI | 16,0750 | 40 | 13 | 355,45 |
| **GRPO + PI + NAFC** | **17,6125** | **32** | **21** | 385,34 |
| GRPO + PI + quota replay | 14,3250 | 54 | 8 | 329,67 |
| GRPO + PI + KL | 15,8750 | 36 | 11 | 395,22 |

**Số câu quên:** tham chiếu model gốc đúng ≥6/8 mẫu, sau train chỉ đúng ≤1/4 mẫu. **Câu mới làm tốt:** tham chiếu đúng ≤2/8 mẫu, sau train đúng ≥3/4 mẫu. Accuracy là tỷ lệ đáp án đúng trung bình trên 4 mẫu/câu; số câu quên không suy ra trực tiếp từ accuracy.

Ở seed 1, NAFC đạt accuracy cao nhất và số câu quên thấp nhất trong bốn nhánh. So với GRPO + PI, NAFC tăng **1,5375 điểm phần trăm**, giảm **8 câu quên (20%)** và tăng **8 câu mới làm tốt**. So với KL, NAFC cao hơn **1,7375 điểm phần trăm** và ít hơn **4 câu quên**.

Quota replay chưa cải thiện kết quả dù dùng cùng số nhóm replay với NAFC. Kết quả này gợi ý cách chọn và điều khiển replay có vai trò, thay vì chỉ tăng lượng ôn. Tuy nhiên, độ dài thực tế giữa các nhánh khác nhau và hiện mới hoàn tất một seed; cần kết quả seed 2–3 để đánh giá độ ổn định, chưa khẳng định nguyên nhân hay hiệu quả tổng quát.

Bảng được tính lại từ [kết quả từng câu](results/seed_1/) bằng `python results/summarize.py`.

## Chạy trên Colab

Mở [01_Phi_Olympiad_3Seeds.ipynb](notebooks/01_Phi_Olympiad_3Seeds.ipynb), chọn **A100**, chọn **Run all** và cấp quyền Drive tại bước 2. Token Hugging Face là tùy chọn. Notebook tự chuẩn bị từ đầu; không cần output hoặc notebook cũ.

Kết quả và checkpoint tự lưu tại `My Drive/NAFC_Phi_Olympic_3Seeds_v2/`, riêng từng seed trong `seed_1/`, `seed_2/`, `seed_3/`. Có nút tải toàn bộ output ở cuối notebook. Khi Colab ngắt, Run all lại trên cùng Drive để dùng phần đã lưu của notebook này. Xem [hướng dẫn](docs/COLAB.md).

## Cấu trúc

```text
notebooks/01_Phi_Olympiad_3Seeds.ipynb   Notebook duy nhất để chạy
src/phi_pilot/                         Runtime, cấu hình và câu hỏi cố định
src/build_notebook.py                  Tạo lại notebook từ mã nguồn
results/seed_1/                        Kết quả seed 1 đã hoàn tất
results/summarize.py                   Tính lại bảng kết quả
tests/                                Kiểm tra CPU, checkpoint và notebook
docs/COLAB.md                         Hướng dẫn chạy và đọc output
```

Để kiểm tra cục bộ: cài `requirements-dev.txt`, chạy `python -m pytest -q`. Tạo lại notebook bằng `python src/build_notebook.py`. Các kiểm tra CPU không thay cho việc chạy thực tế trên A100.
