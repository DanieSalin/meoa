#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from matplotlib.ticker import MaxNLocator
from fix_model import FixedDiffusionUNet
from fixed_diffusion import EnhancedDiffusionModel

# Thiết lập seed cho quá trình tái tạo
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Kiểm tra thiết bị có sẵn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {device}")

class RealIVSurfaceDataset(Dataset):
    def __init__(self, csv_files=None, transform=None, missing_ratio=0.2):
        """
        Dataset cho bề mặt Implied Volatility từ dữ liệu thực
        
        Tham số:
            csv_files (list): Danh sách các file CSV chứa dữ liệu
            transform (callable, optional): Biến đổi dữ liệu
            missing_ratio (float): Tỷ lệ dữ liệu bị thiếu (0.0 - 1.0)
        """
        self.transform = transform
        self.missing_ratio = missing_ratio
        
        # Đọc và xử lý dữ liệu từ các file CSV
        if csv_files is None:
            # Sử dụng tất cả file pivot_df_*.csv nếu không có danh sách cụ thể
            self.csv_files = [f for f in os.listdir() if f.startswith('pivot_df_') and f.endswith('.csv')]
            if not self.csv_files:
                # Sử dụng pivot_csv.csv nếu không có file pivot_df_*.csv
                self.csv_files = ['pivot_csv.csv']
        else:
            self.csv_files = csv_files
            
        self.data = self._load_data()
        
    def _load_data(self):
        """Đọc dữ liệu từ nhiều file CSV và tổ chức thành ma trận 2D"""
        iv_data = []
        
        # Đọc từng file CSV
        for csv_file in self.csv_files:
            try:
                print(f"Đọc dữ liệu từ file {csv_file}...")
                df = pd.read_csv(csv_file)
                
                # Xử lý file pivot_df_*.csv hoặc pivot_csv.csv
                if csv_file.startswith('pivot_df_') or csv_file == 'pivot_csv.csv':
                    # Lấy các cột IV
                    iv_cols = [col for col in df.columns if col.startswith('IV_')]
                    
                    if not iv_cols:
                        print(f"Không tìm thấy cột IV trong file {csv_file}")
                        continue
                    
                    # Lấy dữ liệu IV
                    iv_values = df[iv_cols].values
                    
                    # Lọc bỏ các hàng có chứa NaN
                    valid_rows = ~np.isnan(iv_values).any(axis=1)
                    iv_values = iv_values[valid_rows]
                    
                # Xử lý các file CSV khác (*.csv)
                else:
                    # Kiểm tra xem file có phải là file OHLC không
                    if any(col in df.columns for col in ['open', 'high', 'low', 'close', 'Open', 'High', 'Low', 'Close']):
                        print(f"File {csv_file} có vẻ là dữ liệu OHLC, bỏ qua.")
                        continue
                    
                    # Giả sử đây là file dữ liệu IV
                    # Nếu có cột "implied_volatility" hoặc tương tự
                    iv_cols = [col for col in df.columns if 'volatility' in col.lower() or 'iv' in col.lower()]
                    
                    if iv_cols:
                        iv_values = df[iv_cols].values
                    else:
                        # Nếu không tìm thấy cột IV, giả sử tất cả các cột số đều là dữ liệu IV
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        iv_values = df[numeric_cols].values
                
                if len(iv_values) == 0:
                    print(f"Không có dữ liệu hợp lệ trong file {csv_file}")
                    continue
                
                # Lọc bỏ các hàng có chứa NaN
                valid_rows = ~np.isnan(iv_values).any(axis=1)
                iv_values = iv_values[valid_rows]
                
                # Thêm vào danh sách dữ liệu
                print(f"Đã đọc {len(iv_values)} hàng từ file {csv_file}")
                
                # Nếu dữ liệu chỉ là một cột, thêm shape để tạo 2D
                if len(iv_values.shape) == 1 or iv_values.shape[1] == 1:
                    if len(iv_values.shape) == 1:
                        iv_values = iv_values.reshape(-1, 1)
                    
                    # Tạo ma trận 2D từ mảng 1D
                    n_rows = min(20, len(iv_values))  # Tối đa 20 hàng
                    n_cols = max(11, 1)  # Tối thiểu 11 cột
                    
                    # Tạo ma trận với các giá trị lặp lại
                    temp = np.zeros((n_rows, n_cols))
                    for i in range(n_rows):
                        temp[i, :] = iv_values[i % len(iv_values), 0]
                    
                    iv_values = temp
                
                iv_data.append(iv_values)
                
            except Exception as e:
                print(f"Lỗi khi đọc file {csv_file}: {e}")
        
        # Ghép dữ liệu từ tất cả các file
        if iv_data:
            all_data = np.vstack([data[:min(20, len(data))] for data in iv_data if len(data) > 0])
            
            # Đảm bảo tất cả các mẫu có cùng kích thước
            if len(set([data.shape[1] for data in iv_data])) > 1:
                print("Cảnh báo: Các file có kích thước khác nhau. Điều chỉnh để có cùng kích thước.")
                
                # Tìm số cột phổ biến nhất
                max_cols = max([data.shape[1] for data in iv_data])
                
                # Điều chỉnh kích thước tất cả mẫu
                resized_data = []
                for data in iv_data:
                    if data.shape[1] < max_cols:
                        # Mở rộng bằng cách lặp lại cột cuối
                        padded = np.zeros((data.shape[0], max_cols))
                        padded[:, :data.shape[1]] = data
                        for j in range(data.shape[1], max_cols):
                            padded[:, j] = data[:, -1]
                        resized_data.append(padded)
                    else:
                        resized_data.append(data)
                
                all_data = np.vstack([data[:min(20, len(data))] for data in resized_data])
            
            return all_data
        else:
            # Tạo dữ liệu giả nếu không thể đọc từ file
            print("Không thể đọc dữ liệu từ các file CSV. Tạo dữ liệu giả.")
            return np.random.rand(20, 11) * 0.2 + 0.1
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        iv_surface = self.data[idx].astype(np.float32)
        
        # Tạo mặt nạ (mask) cho các giá trị bị thiếu
        mask = np.random.choice([0, 1], size=iv_surface.shape, p=[self.missing_ratio, 1-self.missing_ratio])
        
        # Nhân với mặt nạ để tạo dữ liệu bị thiếu (0 = thiếu, giá trị khác = có dữ liệu)
        masked_iv = iv_surface * mask
        
        # Chuyển sang tensor và thêm chiều kênh
        iv_surface = torch.tensor(iv_surface, dtype=torch.float32).unsqueeze(0)
        masked_iv = torch.tensor(masked_iv, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        
        if self.transform:
            iv_surface = self.transform(iv_surface)
            masked_iv = self.transform(masked_iv)
        
        return {'original': iv_surface, 'masked': masked_iv, 'mask': mask}

def train_diffusion_model(dataset, epochs=50, batch_size=8, lr=1e-4, save_path="models", resume=False, model_path=None):
    """
    Huấn luyện mô hình Diffusion với dữ liệu thực
    
    Tham số:
        dataset: Dataset chứa dữ liệu IV
        epochs: Số epoch huấn luyện
        batch_size: Kích thước batch
        lr: Tốc độ học
        save_path: Đường dẫn để lưu mô hình
        resume: Tiếp tục huấn luyện từ mô hình đã lưu
        model_path: Đường dẫn đến mô hình đã lưu
    """
    # Tạo DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Lấy mẫu đầu tiên để xác định kích thước dữ liệu
    sample = next(iter(dataloader))
    input_shape = sample['original'].shape[-1]
    print(f"Kích thước dữ liệu đầu vào: {input_shape}")
    
    # Khởi tạo mô hình hoặc tải mô hình đã lưu
    if resume and model_path and os.path.exists(model_path):
        print(f"Tiếp tục huấn luyện từ mô hình {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model = FixedDiffusionUNet(in_channels=2, out_channels=1).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        
        # Khởi tạo Diffusion model với mô hình đã tải
        diffusion = EnhancedDiffusionModel(model, num_diffusion_steps=100)
        
        # Khởi tạo optimizer
        optimizer = Adam(model.parameters(), lr=lr)
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Đã tải trạng thái optimizer")
            except Exception as e:
                print(f"Không thể tải trạng thái optimizer: {e}")
    else:
        print("Khởi tạo mô hình mới")
        model = FixedDiffusionUNet(in_channels=2, out_channels=1).to(device)
        diffusion = EnhancedDiffusionModel(model, num_diffusion_steps=100)
        optimizer = Adam(model.parameters(), lr=lr)
        start_epoch = 0
    
    # Tạo thư mục lưu trữ mô hình
    os.makedirs(save_path, exist_ok=True)
    
    # Ghi lại tổng số epoch để huấn luyện
    total_epochs = start_epoch + epochs
    print(f"Huấn luyện từ epoch {start_epoch + 1} đến {total_epochs}")
    
    # Huấn luyện
    diffusion.fit(dataloader, epochs, optimizer, save_path=save_path, start_epoch=start_epoch)
    
    return diffusion

@torch.no_grad()
def generate_iv_surface(diffusion_model, masked_iv, mask):
    """
    Sử dụng mô hình Diffusion để phục hồi bề mặt IV
    
    Tham số:
        diffusion_model: Mô hình Diffusion đã huấn luyện
        masked_iv: Bề mặt IV bị thiếu
        mask: Mặt nạ cho biết vị trí dữ liệu bị thiếu
    """
    diffusion_model.model.eval()  # Đặt mô hình vào chế độ đánh giá
    
    # Chuyển đổi sang tensor và đưa lên thiết bị
    if not torch.is_tensor(masked_iv):
        masked_iv = torch.tensor(masked_iv, dtype=torch.float32)
    if not torch.is_tensor(mask):
        mask = torch.tensor(mask, dtype=torch.float32)
    
    # Thêm chiều batch nếu cần
    if len(masked_iv.shape) == 2:
        masked_iv = masked_iv.unsqueeze(0).unsqueeze(0)
    elif len(masked_iv.shape) == 3:
        masked_iv = masked_iv.unsqueeze(0)
    
    if len(mask.shape) == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif len(mask.shape) == 3:
        mask = mask.unsqueeze(0)
    
    # Đưa lên thiết bị
    masked_iv = masked_iv.to(device)
    mask = mask.to(device)
    
    # Sinh bề mặt IV
    reconstructed = diffusion_model.sample(masked_iv, mask)
    
    # Kết hợp dữ liệu gốc và dữ liệu phục hồi
    result = torch.where(mask > 0, masked_iv, reconstructed)
    
    return result.cpu().squeeze().numpy()

def plot_iv_surface(iv_data, title="Implied Volatility Surface", save_path=None):
    """
    Vẽ biểu đồ bề mặt Implied Volatility
    
    Tham số:
        iv_data: Dữ liệu IV
        title: Tiêu đề biểu đồ
        save_path: Đường dẫn để lưu biểu đồ
    """
    # Chuyển sang numpy nếu là tensor
    if torch.is_tensor(iv_data):
        iv_data = iv_data.cpu().numpy()
    
    # Xử lý trường hợp có thêm chiều kênh
    if len(iv_data.shape) == 3:
        iv_data = iv_data.squeeze(0)
    
    # Tạo lưới maturities và strikes
    n_rows, n_cols = iv_data.shape
    maturities = np.arange(n_rows)
    strikes = np.arange(n_cols)
    
    X, Y = np.meshgrid(strikes, maturities)
    
    # Vẽ bề mặt 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, iv_data, cmap='viridis', edgecolor='none', alpha=0.8)
    
    # Thêm colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Implied Volatility')
    
    # Đặt nhãn
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Strike Index')
    ax.set_ylabel('Maturity Index')
    ax.set_zlabel('Implied Volatility')
    
    # Điều chỉnh số ticks trên trục để hiển thị số nguyên
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Xoay biểu đồ để có góc nhìn tốt hơn
    ax.view_init(elev=30, azim=45)
    
    # Lưu hoặc hiển thị
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()
    
    plt.close()

def evaluate_reconstruction(original, reconstructed):
    """
    Đánh giá chất lượng phục hồi bề mặt IV
    
    Tham số:
        original: Bề mặt IV gốc
        reconstructed: Bề mặt IV được phục hồi
    
    Trả về:
        metrics: Dict chứa các thông số đánh giá (MSE, RMSE, MAE, MAPE)
    """
    # Chuyển sang numpy nếu là tensor
    if torch.is_tensor(original):
        original = original.cpu().numpy()
    if torch.is_tensor(reconstructed):
        reconstructed = reconstructed.cpu().numpy()
    
    # Loại bỏ chiều thừa
    if len(original.shape) > 2:
        original = original.squeeze()
    if len(reconstructed.shape) > 2:
        reconstructed = reconstructed.squeeze()
    
    # Tính toán các metrics
    mse = mean_squared_error(original, reconstructed)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(original, reconstructed)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((original - reconstructed) / (original + 1e-8))) * 100
    
    # In kết quả
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"MAPE: {mape:.2f}%")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }

def main():
    """
    Hàm chính để chạy mô hình
    
    Bạn có thể điều chỉnh các tham số sau:
    - missing_ratio: Tỷ lệ dữ liệu bị thiếu (0.0 - 1.0)
    - train_model: True để huấn luyện mô hình mới, False để tải mô hình đã lưu
    - resume_training: True để tiếp tục huấn luyện từ mô hình đã lưu
    - model_path: Đường dẫn đến mô hình đã lưu
    - csv_files: Danh sách các file CSV chứa dữ liệu
    """
    # Tham số có thể điều chỉnh
    missing_ratio = 0.3  # 30% dữ liệu bị thiếu
    train_model = True   # Huấn luyện mô hình mới
    resume_training = False  # Tiếp tục huấn luyện từ mô hình đã lưu
    model_path = "models/diffusion_model_final.pth"  # Đường dẫn đến mô hình đã lưu
    
    # Tạo thư mục kết quả
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Tìm tất cả các file CSV
    all_csv_files = [f for f in os.listdir() if f.endswith('.csv')]
    
    # Ưu tiên các file pivot_df_*.csv
    csv_files = [f for f in all_csv_files if f.startswith('pivot_df_')]
    
    # Nếu không có file pivot_df_*.csv, sử dụng pivot_csv.csv
    if not csv_files and 'pivot_csv.csv' in all_csv_files:
        csv_files = ['pivot_csv.csv']
    
    # Nếu không có file nào, sử dụng tất cả các file CSV
    if not csv_files:
        csv_files = all_csv_files
    
    print(f"Sử dụng các file CSV: {csv_files}")
    
    # Tạo dataset từ dữ liệu thực
    dataset = RealIVSurfaceDataset(csv_files=csv_files, missing_ratio=missing_ratio)
    
    print(f"Dataset có {len(dataset)} mẫu")
    
    # Huấn luyện hoặc tải mô hình
    if train_model:
        if resume_training and os.path.exists(model_path):
            print(f"Tiếp tục huấn luyện từ mô hình {model_path}...")
            diffusion = train_diffusion_model(dataset, epochs=50, batch_size=8, lr=1e-4, resume=True, model_path=model_path)
        else:
            print("Bắt đầu huấn luyện mô hình mới...")
            diffusion = train_diffusion_model(dataset, epochs=50, batch_size=8, lr=1e-4)
    else:
        print(f"Tải mô hình đã lưu từ {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Khởi tạo mô hình
        model = FixedDiffusionUNet(in_channels=2, out_channels=1).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        diffusion = EnhancedDiffusionModel(model, num_diffusion_steps=100)
    
    # Lấy một mẫu để kiểm tra
    sample_idx = np.random.randint(0, len(dataset))
    sample = dataset[sample_idx]
    original_iv = sample['original']
    masked_iv = sample['masked']
    mask = sample['mask']
    
    print(f"Đang xử lý mẫu #{sample_idx}...")
    
    # Lưu biểu đồ bề mặt IV gốc
    plot_iv_surface(original_iv, title="Bề mặt IV gốc", save_path="results/original_surface.png")
    
    # Lưu biểu đồ bề mặt IV bị thiếu
    plot_iv_surface(masked_iv, title="Bề mặt IV bị thiếu", save_path="results/masked_surface.png")
    
    # Phục hồi bề mặt IV
    print("Phục hồi bề mặt IV...")
    reconstructed_iv = generate_iv_surface(diffusion, masked_iv, mask)
    
    # Lưu biểu đồ bề mặt IV được phục hồi
    plot_iv_surface(reconstructed_iv, title="Bề mặt IV được phục hồi", save_path="results/reconstructed_surface.png")
    
    # Đánh giá chất lượng phục hồi
    print("Đánh giá chất lượng phục hồi:")
    metrics = evaluate_reconstruction(original_iv, reconstructed_iv)
    
    # Lưu metrics
    with open("results/metrics.txt", "w") as f:
        f.write(f"Sample index: {sample_idx}\n")
        f.write(f"Missing ratio: {missing_ratio}\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    print("Hoàn thành! Kết quả đã được lưu trong thư mục 'results/'")
    
    # Tạo so sánh trực quan
    plt.figure(figsize=(18, 6))
    
    # Hiển thị dữ liệu gốc
    plt.subplot(131)
    plt.imshow(original_iv.squeeze(), cmap='viridis')
    plt.colorbar(label='IV')
    plt.title('Dữ liệu gốc')
    
    # Hiển thị dữ liệu bị thiếu
    plt.subplot(132)
    plt.imshow(masked_iv.squeeze(), cmap='viridis')
    plt.colorbar(label='IV')
    plt.title('Dữ liệu bị thiếu')
    
    # Hiển thị dữ liệu phục hồi
    plt.subplot(133)
    plt.imshow(reconstructed_iv, cmap='viridis')
    plt.colorbar(label='IV')
    plt.title('Dữ liệu phục hồi')
    
    plt.tight_layout()
    plt.savefig("results/comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main() 