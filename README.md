# Dự Đoán Giới Tính Sử Dụng Thuật Toán Phân Lớp Naive Bayes

Repository lưu trữ mã nguồn, dữ liệu, kết quả thực nghiệm và báo cáo Bài tập lớn học phần **Khai phá dữ liệu** (Lớp: `Khai phá dữ liệu-1-26 (65CNTT)`) — Trường Đại học Thủy Lợi (TLU).

---

## 📌 Thông tin chung
- **Đề tài:** Dự đoán giới tính sử dụng thuật toán phân lớp Gaussian Naive Bayes
- **Giảng viên hướng dẫn:** TS. Trần Mạnh Tuấn
- **Nhóm thực hiện:** Nhóm 20
  - **Nguyễn Khắc Minh Hiếu** — `2251162000` — Lớp `64HTTT3` *(Nhóm trưởng)*
  - **Nguyễn Năng Vũ Hùng** — `2151160512` — Lớp `63HTTT1` *(Thành viên)*
  - **Ngô Thanh Huyền** — `1951060770` — Lớp `61TH3` *(Thành viên)*

---

## 📊 Dữ liệu & Quy trình xử lý
- **Bộ dữ liệu:** `Davis` (gói `carData`) gồm 200 bản ghi khảo sát với các thuộc tính `height` (chiều cao đo, cm), `weight` (cân nặng đo, kg), `sex` (nhãn `F` hoặc `M`) cùng các cột tự khai (`repht`, `repwt`) dùng để đối chiếu.
- **Tiền xử lý:**
  - Loại bỏ bản ghi bất thường số `12` (`height = 57 cm`, `weight = 166 kg` do dấu hiệu đảo ngược cột đo lường) và lưu vết tại `dong_loai.csv`.
  - Giữ lại **199 bản ghi hợp lệ** (`111 F`, `88 M`), chỉ sử dụng 2 thuộc tính đầu vào là `height` và `weight`.
  - Chia dữ liệu theo tỷ lệ **70/30** có phân tầng (`stratify` theo `sex`, `random_state = 42`): **139 mẫu huấn luyện** (`78 F`, `61 M`) và **60 mẫu kiểm thử** (`33 F`, `27 M`).

---

## 📂 Cấu trúc thư mục
```text
Data_Mining_TLU/
├── Nhom20_Source_HoanChinh/
│   ├── data/
│   │   └── Davis.csv               # Dữ liệu khảo sát Davis gốc
│   ├── ket_qua/
│   │   ├── dong_loai.csv           # Bản ghi bất thường bị loại (bản ghi số 12)
│   │   ├── du_lieu_sach.csv        # 199 bản ghi hợp lệ sau làm sạch
│   │   ├── tap_huan_luyen.csv      # 139 mẫu huấn luyện
│   │   └── du_doan_kiem_thu.csv    # Nhãn thật và nhãn dự đoán trên 60 mẫu kiểm thử
│   ├── weka/
│   │   ├── train_139.arff          # Tập huấn luyện định dạng ARFF cho Weka
│   │   ├── test_60.arff            # Tập kiểm thử định dạng ARFF cho Weka
│   │   ├── doi_chieu.csv           # Bảng đối chiếu dự đoán giữa Python và Weka
│   │   └── weka_evaluation.txt     # Kết quả xuất ra từ Weka 3.8.6
│   ├── main.py                     # Luồng chính: làm sạch, chia tập, đánh giá và dự đoán mẫu mới
│   ├── naive_bayes.py              # Tự cài đặt thuật toán Gaussian Naive Bayes (fit & predict)
│   ├── thuc_nghiem.py              # Chạy kiểm tra chéo 5-fold, thử nghiệm thuộc tính & kiểm định McNemar
│   └── requirements.txt            # Danh sách phiên bản thư viện Python cố định
└── README.md                       # Tài liệu giới thiệu dự án
