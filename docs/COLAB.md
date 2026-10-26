# Chạy Phi–Olympiad trên Colab

1. Upload `notebooks/01_Phi_Olympiad_3Seeds.ipynb` lên Colab.
2. Chọn A100 trong phần loại môi trường thời gian chạy.
3. Token Hugging Face ở đầu notebook là tùy chọn; không cần Secrets.
4. Chọn Run all và đồng ý quyền Drive tại bước 2.
5. Theo dõi số câu đã lưu ở bước chuẩn bị và số bước train tại từng seed/nhánh.
6. Xem báo cáo trên Drive hoặc bấm **Download output ZIP** ở cuối notebook.

Notebook nhúng đầy đủ câu hỏi, mã và cấu hình. Mọi câu trả lời tham chiếu, anchor,
calibration và model train được tạo từ đầu; không lấy model đã train của seed khác.
Danh sách train/validation/test giữ nguyên thứ tự của thí nghiệm seed 1.

## Output trên Drive

```text
NAFC_Phi_Olympic_3Seeds_v2/
  identity.json, plan.json, questions.json, context.json
  source/                         Mã và manifest dùng cho run này
  logs/                           Log cài đặt, chuẩn bị và lần gọi đầu từng seed
  samples/                        Lô đáp án tham chiếu đã lưu
  anchors.json, buckets_*.json     Anchor và calibration chung
  readiness.json                  Chẩn đoán trước train
  results/base_*.json              Lần chấm lại model gốc để đo nhiễu
  smoke/                          Lượt thử 2 bước, không dùng làm model chính
  seed_1/                         Tương tự cho seed_2/ và seed_3/
    identity.json, plan.json       Seed và hash dữ liệu chung
    runs/mon/, rep/, quota/, kl/   Checkpoint và trace riêng từng nhánh
    samples/                      Đáp án đánh giá
    results/                      Điểm từng câu, từng mốc
    logs/
    summary.csv, summary.json, REPORT.md
  aggregate.csv, aggregate.json   Chỉ tạo khi đủ cả ba seed
```

## Checkpoint và gián đoạn

Checkpoint đầy đủ được ghi mỗi 25 bước, giữ ba bản gần nhất. Các adapter đánh giá
ở bước 50/100/150/200/250/300 được giữ riêng. Chỉ checkpoint có `COMPLETE.json`
và đầy đủ các file có kích thước khớp mới được nạp lại. Không sử dụng bản thiếu;
nếu không còn bản hợp lệ, chương trình dừng để tránh vô tình train lại từ đầu.

Sau gián đoạn, kết nối lại A100 và Run all cùng notebook trên cùng Drive.
Các lô dữ liệu đã ghi và nhánh đã hoàn tất sẽ được dùng lại. Không xóa thư mục
trên Drive, không chạy hai notebook cùng lúc vào cùng thư mục output.
Đây là cấu trúc mới; không trỏ nó vào thư mục checkpoint của notebook phiên bản cũ.

Bộ giám sát dùng file log thường, không đợi EOF của worker còn sống sau khi tiến trình
chính thoát. Mỗi lệnh có timeout, heartbeat 30 giây và giới hạn 15 phút không có output;
train còn kiểm tra bước tối ưu có tăng hay không. Colab vẫn có thể ngắt hoặc thu hồi VM.
Chạy lại checkpoint không cam kết sinh đáp án giống từng bit với một phiên không ngắt.

## Cấu hình và nguồn

`src/phi_pilot/plan.json` ghi model revision, dataset revision, các hash mã nguồn
và phiên bản thư viện. Bộ điều khiển PI và NAFC kế thừa mã tại commit `786291e`
của DATN-FPTU; giữ nguyên công thức. Runtime thêm ghi trace, lưu trạng thái replay,
quota đối chiếu, chia seed và gộp adapter trên CPU trong tiến trình riêng.

Ngân sách PI **400,86 token** được giữ từ calibration validation của seed 1.
Chuẩn bị mới vẫn sinh tham chiếu và hiệu chỉnh lambda khởi tạo, không tìm ngân sách
bằng kết quả test. Ngân sách PI là mục tiêu mềm, không bảo đảm mọi nhánh có cùng độ dài.

Để tái hiện cấu hình thực tế seed 1, tổng context được khai báo rõ là **1.538 token**.
Completion cap là **1.536 token**, nhưng phần còn lại sau đề bài có thể thấp hơn.
Độ dài đề được đếm từ token IDs, không dùng số trường của đối tượng tokenizer.

Kết quả `results/seed_1/` trong repo là số đo đã có trước đợt dọn mã này.
Notebook mới không tự nhận những checkpoint cũ là kết quả của một lần chạy mới.
KL và quota là các đối chứng bổ sung; không coi chúng là những giả thuyết chưa từng xem dữ liệu.
