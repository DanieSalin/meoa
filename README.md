# Phục hồi Bề mặt Implied Volatility từ Dữ liệu Thiếu

Dự án này sử dụng các mô hình Diffusion để phục hồi và nội suy bề mặt Implied Volatility (IV) từ dữ liệu thị trường bị thiếu.

## Giới thiệu

Bề mặt Implied Volatility (IVS) là một công cụ quan trọng trong định giá và quản lý rủi ro tài chính. Tuy nhiên, thường xuyên có tình trạng thiếu dữ liệu ở một số mức giá thực hiện (strike) hoặc thời gian đáo hạn (maturity), đặc biệt là đối với các tùy chọn xa giá thị trường (deep OTM/ITM).

Dự án này sử dụng Denoising Diffusion Probabilistic Models (DDPMs) để tạo lại hoặc nội suy bề mặt IVS hoàn chỉnh từ dữ liệu bị thiếu, đồng thời đảm bảo các ràng buộc arbitrage-free.

## Cấu trúc dự án

### File gốc:
- `read_pdfs.py`: Công cụ đọc và trích xuất thông tin từ các bài báo PDF
- `create_diffusion_model.py`: Mô hình Diffusion ban đầu cho việc phục hồi bề mặt IV
- `LSTM_model.py`: Mô hình LSTM gốc cho dự đoán implied volatility
- `Data_engineering.py`: Xử lý và chuẩn bị dữ liệu
- `Grid_Search_Optimization.py`: Tối ưu hóa hyperparameter cho mô hình
- `Number_of_Steps_Optimization.py`: Tối ưu hóa số bước thời gian cho mô hình LSTM

### File mới:
- `real_data_diffusion.py`: Mô hình Diffusion cải tiến cho dữ liệu CSV thực tế
- `fixed_diffusion.py`: Lớp EnhancedDiffusionModel với nhiều cải tiến
- `run.py`: Script chạy mô hình với các tùy chọn từ dòng lệnh
- `fix_model.py`: Mô hình đã sửa với InstanceNorm thay vì BatchNorm
- `debug_model.py`: Công cụ debug khi gặp vấn đề với kích thước tensor

## Cài đặt

1. Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

2. Chuẩn bị dữ liệu CSV của bạn. Các file CSV phải tuân theo một trong các quy tắc sau:
   - Các file có tên bắt đầu bằng "pivot_df_" hoặc "pivot_csv.csv" với các cột bắt đầu bằng "IV_"
   - Các file CSV khác với các cột chứa "volatility" hoặc "iv" trong tên
   - Các cột số khác nếu không tìm thấy cột IV cụ thể

## Sử dụng

### Khôi phục bề mặt IV từ dữ liệu CSV thực tế

```bash
python run.py --train
```

Các tùy chọn:
- `--missing_ratio X`: Tỷ lệ dữ liệu bị thiếu (mặc định: 0.3)
- `--train`: Huấn luyện mô hình mới
- `--resume`: Tiếp tục huấn luyện từ mô hình đã lưu
- `--model_path PATH`: Đường dẫn đến mô hình đã lưu (mặc định: models/diffusion_model_final.pth)
- `--epochs N`: Số epoch huấn luyện (mặc định: 50)
- `--batch_size N`: Kích thước batch (mặc định: 8)
- `--lr X`: Learning rate (mặc định: 1e-4)

### Hoặc sử dụng file trực tiếp:

```bash
python real_data_diffusion.py
```

### Đọc và tóm tắt bài báo PDF

```bash
python read_pdfs.py
```

### Chạy mô hình Diffusion gốc

```bash
python create_diffusion_model.py
```

## Phương pháp

Dự án này sử dụng Denoising Diffusion Probabilistic Models (DDPMs), một loại mô hình sinh xác suất dựa trên quá trình Markov thêm nhiễu dần vào dữ liệu và sau đó học cách loại bỏ nhiễu.

Các bước chính:
1. Dữ liệu bị thiếu được biểu diễn dưới dạng ma trận IV với một số vị trí bị thiếu (masked)
2. Mô hình U-Net được sử dụng để học quá trình loại bỏ nhiễu
3. Quá trình lấy mẫu ngược để phục hồi dữ liệu bị thiếu

### Cải tiến trong phiên bản mới:
- Sử dụng InstanceNorm thay vì BatchNorm để xử lý tốt hơn với batch size nhỏ
- Cải thiện quản lý kích thước tensor để tránh lỗi không khớp
- Hỗ trợ nhiều định dạng dữ liệu CSV khác nhau
- Khả năng tiếp tục huấn luyện từ mô hình đã lưu
- Đánh giá kết quả trực quan với so sánh trực tiếp

## Kết quả

Kết quả sinh ra được lưu trong thư mục `results/`:
- Biểu đồ bề mặt IV gốc (original_surface.png)
- Biểu đồ bề mặt IV bị thiếu (masked_surface.png)
- Biểu đồ bề mặt IV được phục hồi (reconstructed_surface.png)
- So sánh trực quan giữa 3 bề mặt (comparison.png)
- Đánh giá chất lượng phục hồi (metrics.txt): MSE, RMSE, MAE, MAPE
