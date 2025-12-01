# Hệ Thống Phân Loại Biển Báo Giao Thông

Hệ thống nhận dạng biển báo giao thông sử dụng học sâu với mạng CNN (Convolutional Neural Network). Dự án này phân loại 4 loại biển báo: Đoạn đường hay xảy ra tai nạn, Điểm dừng xe buýt, Hạn chế chiều cao, và Đi chậm.

## 📋 Mục Lục

- [Tính Năng](#tính-năng)
- [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
- [Yêu Cầu](#yêu-cầu)
- [Cài Đặt](#cài-đặt)
- [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)
- [Thông Tin Mô Hình](#thông-tin-mô-hình)
- [Dataset](#dataset)
- [Giấy Phép](#giấy-phép)

## ✨ Tính Năng

- **Huấn Luyện Mô Hình CNN**: Huấn luyện mô hình CNN tùy chỉnh cho phân loại biển báo
- **Tiền Xử Lý Dữ Liệu**: Tự động tiền xử lý và kiểm tra ảnh
- **Phân Chia Dataset**: Tự động chia train/test với stratification
- **Giao Diện Web**: Giao diện web tương tác dựa trên Gradio để dự đoán thời gian thực
- **Độ Chính Xác Cao**: Đạt độ chính xác 94%+ trên tập test
- **Di Động**: Sử dụng đường dẫn tương đối để dễ dàng triển khai

## 📁 Cấu Trúc Dự Án

```
Traffic_Sign_Classification_System/
├── code/                          # Thư mục mã nguồn
│   ├── train.py                   # Script huấn luyện mô hình
│   ├── gra.py                     # Giao diện web Gradio
│   ├── check.py                   # Kiểm tra dataset
│   ├── split.py                   # Chia train/test
│   ├── normalize.py               # Chuẩn hóa dữ liệu
│   └── pre_process.py             # Tiền xử lý ảnh
├── dataset_split/                 # Dataset đã chia (train/test)
│   ├── train/                     # Ảnh huấn luyện
│   └── test/                      # Ảnh kiểm tra
├── data_ok/                       # Dataset đã validate
├── dataset_raw/                   # Dữ liệu thô
├── cnn_traffic_sign.keras         # File mô hình đã huấn luyện
└── README_VI.md                   # File này
```

## 🔧 Yêu Cầu

- Python 3.8+
- TensorFlow 2.x
- OpenCV (cv2)
- Pillow (PIL)
- NumPy
- scikit-learn
- Gradio
- Matplotlib

### Cài Đặt Thư Viện

```bash
pip install tensorflow opencv-python pillow numpy scikit-learn gradio matplotlib
```

## 🚀 Cài Đặt

1. **Clone hoặc tải dự án này**
   ```bash
   cd Traffic_Sign_Classification_System
   ```

2. **Cài đặt các package cần thiết**
   ```bash
   pip install -r requirements.txt
   ```

3. **Kiểm tra cấu trúc thư mục**
   Đảm bảo bạn có thư mục `code/` với tất cả các script Python

## 💻 Hướng Dẫn Sử Dụng

### 1. Tiền Xử Lý Dữ Liệu

Xử lý ảnh raw về kích thước chuẩn 64x64:

```bash
cd code
python pre_process.py
```

### 2. Kiểm Tra Dataset

Kiểm tra định dạng và kích thước ảnh:

```bash
python check.py
```

### 3. Phân Chia Dataset

Chia dữ liệu thành train (80%) và test (20%):

```bash
python split.py
```

### 4. Huấn Luyện Mô Hình

Huấn luyện mô hình CNN:

```bash
python train.py
```

**Kết quả huấn luyện:**
- Mô hình sẽ huấn luyện trong 10 epochs
- Hiển thị tiến trình cho mỗi epoch
- Mô hình được lưu tại `../cnn_traffic_sign.keras`

### 5. Chạy Giao Diện Web

Khởi chạy giao diện web Gradio để dự đoán:

```bash
python gra.py
```

**Truy cập giao diện:**
- Local: `http://127.0.0.1:7860`
- Link công khai sẽ được hiển thị nếu `share=True`

## 🧠 Thông Tin Mô Hình

### Kiến Trúc

- **Loại**: Sequential CNN
- **Input Shape**: (64, 64, 3)
- **Các Lớp**:
  - Conv2D (32 filters) + MaxPooling
  - Conv2D (64 filters) + MaxPooling
  - Conv2D (128 filters) + MaxPooling
  - Flatten
  - Dense (128 units) + Dropout (0.5)
  - Dense (4 units, softmax)

### Hiệu Suất

- **Độ Chính Xác Training**: ~98%
- **Độ Chính Xác Test**: ~94%
- **Hàm Loss**: Sparse Categorical Crossentropy
- **Optimizer**: Adam

### Các Lớp

Mô hình phân loại 4 loại biển báo:

| Class ID | Nhãn (Label) | Tên Đầy Đủ                   |
|----------|--------------|------------------------------|
| 0        | accident     | Đoạn đường hay xảy ra tai nạn|
| 1        | bus          | Điểm dừng xe buýt            |
| 2        | high         | Hạn chế chiều cao            |
| 3        | slow         | Đi chậm                      |

## 📊 Dataset

### Yêu Cầu Ảnh

- **Kích thước**: 64x64 pixels
- **Định dạng**: RGB hoặc RGBA
- **Loại file**: PNG, JPG, JPEG

### Thống Kê Dataset

- **Tổng số ảnh**: ~385 ảnh
- **Tập Training**: 80% (~308 ảnh)
- **Tập Test**: 20% (~77 ảnh)
- **Số lớp**: 4 (phân bố cân bằng)

## 🛠️ Mô Tả Các Script

| Script | Mục Đích |
|--------|----------|
| `train.py` | Huấn luyện mô hình CNN và lưu lại |
| `gra.py` | Khởi chạy giao diện web Gradio để dự đoán |
| `check.py` | Kiểm tra tính hợp lệ của ảnh dataset |
| `split.py` | Chia dataset thành tập train/test |
| `normalize.py` | Chuẩn hóa và tiền xử lý dữ liệu |
| `pre_process.py` | Resize và lọc ảnh |

## 📝 Lưu Ý

- Tất cả đường dẫn sử dụng tham chiếu tương đối để dễ di chuyển
- Định dạng file mô hình: `.keras` (được khuyến nghị bởi TensorFlow 2.x)
- Chạy tất cả scripts từ thư mục `code/`
- Đảm bảo các thư mục dataset tồn tại trước khi chạy

## 🐛 Xử Lý Lỗi

### Lỗi Không Tìm Thấy Mô Hình

Nếu gặp lỗi "Model not found":
- Đảm bảo file `cnn_traffic_sign.keras` tồn tại ở thư mục gốc dự án
- Chạy `train.py` để tạo mô hình

### Lỗi Đường Dẫn

Nếu gặp lỗi đường dẫn:
- Đảm bảo bạn đang chạy scripts từ thư mục `code/`
- Kiểm tra các thư mục dữ liệu tồn tại ở cấp cha

### Lỗi Import

Nếu thiếu packages:
```bash
pip install --upgrade tensorflow opencv-python pillow numpy scikit-learn gradio matplotlib
```

## 📄 Giấy Phép

Dự án này phục vụ mục đích giáo dục.

## 👤 Tác Giả

Được tạo để demo phân loại biển báo giao thông.

---

**Phiên Bản Tiếng Anh**: Xem [README.md](README.md) để đọc tài liệu tiếng Anh.


