# Nhóm 20 · Khai phá dữ liệu
**Đề tài theo danh sách lớp** — Dự đoán giới tính sử dụng thuật toán phân lớp Navie Bayes. Tên thuật toán dùng trong chương trình là Naive Bayes.

**Nhóm trưởng** — Nguyễn Khắc Minh Hiếu. Các thành viên cùng nhóm trong danh sách là Nguyễn Năng Vũ Hùng và Ngô Thanh Huyền.

## Chạy lần đầu trên Windows bằng VS Code

1. Cài Python 3.12 bản 64 bit từ https://www.python.org/downloads/ và đánh dấu **Add python.exe to PATH** nếu dùng bộ cài truyền thống. Cài VS Code từ https://code.visualstudio.com/.
2. Giải nén toàn bộ ZIP. Mở VS Code, chọn **File → Open Folder** và chọn thư mục **Nhom20** chứa main.py. Trong **Extensions** (Ctrl+Shift+X), tìm **Python** do Microsoft phát hành và bấm **Install**.
3. Vào **Terminal → New Terminal**. Chạy từng dòng sau. Lần cài thư viện đầu tiên cần Internet.

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
~~~

4. Để dùng nút chạy hình tam giác, nhấn **Ctrl+Shift+P → Python: Select Interpreter** rồi chọn Python trong **.venv**. Mở **main.py** và bấm **Run Python File**.
5. Để thử dự đoán một mẫu gồm chiều cao 170 cm và cân nặng 65 kg, chạy lệnh sau. Hai số phải nhập theo đúng thứ tự.

~~~powershell
.\.venv\Scripts\python.exe main.py --predict 170 65
~~~

Nếu Windows báo không tìm thấy lệnh **py**, thử **python --version**. Khi kết quả là 3.12.x, dùng **python -m venv .venv** ở dòng đầu. Nếu Terminal báo không tìm thấy requirements.txt, mở lại đúng thư mục chứa main.py.

## Dữ liệu và cách triển khai

Dòng phân công ghi tên đề tài. Nhóm dùng bộ Davis làm lựa chọn triển khai. Dữ liệu gồm 200 người thường xuyên tập thể dục. Nhãn sex có hai giá trị F và M. Mô hình học nhãn đã ghi trong khảo sát. Chiều cao, cân nặng có phân bố chồng lấn, và mẫu khảo sát có phạm vi hẹp. Kết quả chỉ dùng minh họa cho bài thực hành này.

- **data/Davis.csv** là bản CSV kèm sẵn để chạy ngoại tuyến sau khi cài thư viện. Tài liệu bộ dữ liệu: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Davis.html . Nguồn CSV: https://vincentarelbundock.github.io/Rdatasets/csv/carData/Davis.csv .
- **main.py → load_data** đọc dữ liệu, thống kê giá trị thiếu, giữ hai biến đo trực tiếp height (cm), weight (kg) và nhãn sex. rownames dùng đối chiếu bản ghi. repht và repwt là số tự khai, có ô trống và đo lại cùng hai đại lượng. Dòng 12 có số đo 57 cm, 166 kg trong khi tự khai 163 cm, 56 kg. Nhóm loại riêng dòng này và lưu dấu vết trong dong_loai.csv. Nhiều người có thể có cùng số đo nên chương trình giữ các bản ghi có cùng chiều cao và cân nặng.
- **main.py → run** chia phân tầng theo nhãn, giữ gần cùng tỷ lệ F/M ở hai tập. Tỷ lệ 70/30 theo cách trình bày trong Bai tap lon.pdf. Hạt giống 42 giúp người chấm tái lập kết quả.
- **naive_bayes.py → fit** tính tần suất mỗi lớp, trung bình và phương sai của mỗi thuộc tính trong lớp từ tập huấn luyện. Gaussian Naive Bayes giả sử các thuộc tính độc lập khi đã biết lớp và mô hình hóa từng thuộc tính liên tục bằng phân phối chuẩn. Giả sử độc lập là một xấp xỉ vì chiều cao và cân nặng có thể liên hệ với nhau.
- **naive_bayes.py → predict** tính điểm mỗi lớp bằng log xác suất tiên nghiệm cộng hai log mật độ Gaussian. Chọn lớp có điểm lớn nhất. Chương trình so sánh từng nhãn kiểm thử với sklearn.naive_bayes.GaussianNB.
- **ket_qua/** có dữ liệu sau làm sạch, các dòng bị loại, tập huấn luyện, kết quả từng dòng kiểm thử, bảng precision/recall/F1, biểu đồ phân bố và ma trận nhầm lẫn. Chạy lại sẽ tạo lại các tệp này.

## Đọc kết quả chạy mẫu

Trong 199 bản ghi hợp lệ, 139 dùng để học và 60 để kiểm thử. Mô hình dự đoán đúng 54/60, đạt accuracy 90% và macro-F1 0,8982. Nếu luôn đoán lớp phổ biến nhất của tập huấn luyện (F), kết quả là 33/60, tức 55%. Trong 33 mẫu F, mô hình đoán đúng 31. Trong 27 mẫu M, mô hình đoán đúng 23. Mỗi hàng trong hình ma trận ứng với nhãn thật, mỗi cột ứng với nhãn dự đoán. Đây là kết quả trên một lần chia cố định.

## Đối chiếu tài liệu môn học

Bài 2 Tiền xử lý dữ liệu và giáo trình Khai phá dữ liệu là cơ sở cho bước mô tả, chọn thuộc tính và làm sạch. Phần phân lớp Bayesian trong Bài 5 là cơ sở cho thuật toán. Hai tài liệu Python hỗ trợ phần cài đặt. File Bai tap lon.pdf minh họa quy trình từ dữ liệu thô đến đánh giá, còn 3. Thuc hanh BTL.pdf hướng dẫn xác định bài toán và diễn giải kết quả. Bài mẫu dùng ID3 cho dữ liệu ô tô. Source của nhóm 20 triển khai Gaussian Naive Bayes cho hai biến số liên tục.
