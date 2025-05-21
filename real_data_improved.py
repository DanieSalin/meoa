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
from fix_model_improved import ImprovedDiffusionUNet, ImprovedDiffusionModel
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
    def __init__(self, csv_files=None, transform=None, missing_ratio=0.2, normalize=True):
        """
        Dataset cho bề mặt Implied Volatility từ dữ liệu thực
        
        Tham số:
            csv_files (list): Danh sách các file CSV chứa dữ liệu
            transform (callable, optional): Biến đổi dữ liệu
            missing_ratio (float): Tỷ lệ dữ liệu bị thiếu (0.0 - 1.0)
            normalize (bool): Chuẩn hóa dữ liệu IV
        """
        self.transform = transform
        self.missing_ratio = missing_ratio
        self.normalize = normalize
        
        # Đọc và xử lý dữ liệu từ các file CSV
        if csv_files is None:
            # Sử dụng tất cả file pivot_df_*.csv
            self.csv_files = [f for f in os.listdir() if f.startswith('pivot_df_') and f.endswith('.csv')]
            if not self.csv_files:
                # Sử dụng pivot_csv.csv nếu không có file pivot_df_*.csv
                self.csv_files = ['pivot_csv.csv']
        else:
            self.csv_files = csv_files
        
        print(f"Danh sách file CSV: {self.csv_files}")
        self.data, self.dtm, self.moneyness = self._load_data()
        
    def _load_data(self):
        """Đọc dữ liệu từ nhiều file CSV và tổ chức thành ma trận 2D với thông tin DTM và moneyness"""
        # Lưu trữ dữ liệu theo kích thước ma trận
        iv_data_groups = {}  # Nhóm theo kích thước ma trận (số cột)
        dtm_data = []
        moneyness_data = []
        
        # Đọc từng file CSV
        for csv_file in self.csv_files:
            try:
                print(f"Đọc dữ liệu từ file {csv_file}...")
                df = pd.read_csv(csv_file)
                
                # Xử lý file pivot_df_*.csv
                if csv_file.startswith('pivot_df_'):
                    # Lấy các cột IV và thông tin khác
                    iv_cols = [col for col in df.columns if col.startswith('IV_')]
                    
                    if not iv_cols:
                        print(f"Không tìm thấy cột IV trong file {csv_file}")
                        continue
                    
                    # Lấy thông tin DTM và các thông số khác
                    dtm = df['Days to Maturity'].values
                    
                    # Lấy Log Forward Moneyness nếu có
                    if 'Log Forward Moneyness' in df.columns:
                        moneyness = df['Log Forward Moneyness'].values
                    else:
                        # Tính moneyness từ Strike Price và Close_Index
                        strike = int(csv_file.split('_')[-1].replace('.csv', ''))
                        moneyness = np.log(strike / df['Close_Index'])
                    
                    # Lấy dữ liệu IV
                    iv_values = df[iv_cols].values
                    
                    # Lọc bỏ các hàng có chứa NaN
                    valid_rows = ~np.isnan(iv_values).any(axis=1)
                    iv_values = iv_values[valid_rows]
                    dtm_filtered = dtm[valid_rows]
                    moneyness_filtered = moneyness[valid_rows]
                    
                    # Chuẩn hóa dữ liệu nếu cần
                    if self.normalize and len(iv_values) > 0:
                        # Tránh lỗi với mảng có 1 phần tử
                        if len(iv_values) == 1:
                            # Chuẩn hóa đặc biệt cho mảng có 1 phần tử
                            iv_values = np.ones_like(iv_values) * 0.5
                        else:
                            # Tránh division by zero với std=0
                            std = np.std(iv_values)
                            if std != 0:
                                iv_values = (iv_values - np.mean(iv_values)) / std
                            else:
                                iv_values = iv_values - np.mean(iv_values)
                    
                    # Nhóm dữ liệu theo số cột (width)
                    width = iv_values.shape[1] if iv_values.size > 0 else 0
                    if width > 0:
                        if width not in iv_data_groups:
                            iv_data_groups[width] = []
                        iv_data_groups[width].append((iv_values, dtm_filtered, moneyness_filtered))
                    
                    print(f"Đã đọc {len(iv_values)} hàng từ file {csv_file}, kích thước: {iv_values.shape if iv_values.size > 0 else 'trống'}")
                    
                # Xử lý pivot_csv.csv
                elif csv_file == 'pivot_csv.csv':
                    # Lấy các cột IV
                    iv_cols = [col for col in df.columns if col.startswith('IV_')]
                    
                    if not iv_cols:
                        print(f"Không tìm thấy cột IV trong file {csv_file}")
                        continue
                    
                    # Lấy thông tin DTM
                    dtm = df['Days to Maturity'].values
                    
                    # Tạo moneyness cho các mức giá khác nhau
                    strikes = [int(col.split('_')[1]) for col in iv_cols]
                    moneyness_matrix = np.zeros((len(df), len(strikes)))
                    
                    for i, row in enumerate(df.itertuples()):
                        close_index = row.Close_Index
                        for j, strike in enumerate(strikes):
                            moneyness_matrix[i, j] = np.log(strike / close_index)
                    
                    # Lấy dữ liệu IV
                    iv_values = df[iv_cols].values
                    
                    # Lọc bỏ các hàng có chứa NaN
                    valid_rows = ~np.isnan(iv_values).any(axis=1)
                    iv_values = iv_values[valid_rows]
                    dtm_filtered = dtm[valid_rows]
                    moneyness_filtered = moneyness_matrix[valid_rows]
                    
                    # Chuẩn hóa dữ liệu nếu cần
                    if self.normalize and len(iv_values) > 0:
                        # Tránh lỗi với mảng có 1 phần tử
                        if len(iv_values) == 1:
                            # Chuẩn hóa đặc biệt cho mảng có 1 phần tử
                            iv_values = np.ones_like(iv_values) * 0.5
                        else:
                            # Tránh division by zero với std=0
                            std = np.std(iv_values)
                            if std != 0:
                                iv_values = (iv_values - np.mean(iv_values)) / std
                            else:
                                iv_values = iv_values - np.mean(iv_values)
                    
                    # Nhóm dữ liệu theo số cột (width)
                    width = iv_values.shape[1] if iv_values.size > 0 else 0
                    if width > 0:
                        if width not in iv_data_groups:
                            iv_data_groups[width] = []
                        iv_data_groups[width].append((iv_values, dtm_filtered, moneyness_filtered))
                    
                    print(f"Đã đọc {len(iv_values)} hàng từ file {csv_file}, kích thước: {iv_values.shape if iv_values.size > 0 else 'trống'}")
                
            except Exception as e:
                print(f"Lỗi khi đọc file {csv_file}: {e}")
        
        # Chọn nhóm lớn nhất để sử dụng
        if iv_data_groups:
            # Tìm nhóm có nhiều mẫu nhất
            max_samples_count = 0
            max_width = None
            
            for width, data_group in iv_data_groups.items():
                total_samples = sum(len(item[0]) for item in data_group)
                print(f"Nhóm kích thước {width}: {len(data_group)} file, {total_samples} mẫu")
                
                if total_samples > max_samples_count:
                    max_samples_count = total_samples
                    max_width = width
            
            if max_width is not None:
                print(f"Sử dụng nhóm kích thước {max_width} với {max_samples_count} mẫu")
                selected_group = iv_data_groups[max_width]
                
                # Ghép dữ liệu từ nhóm đã chọn
                all_iv_data = np.vstack([item[0] for item in selected_group])
                all_dtm_data = np.concatenate([item[1] for item in selected_group])
                
                # Xử lý moneyness data
                if selected_group and isinstance(selected_group[0][2], np.ndarray):
                    if len(selected_group[0][2].shape) == 1:
                        # Trường hợp moneyness là vector 1D
                        all_moneyness_data = np.concatenate([item[2] for item in selected_group])
                    else:
                        # Trường hợp moneyness là ma trận 2D
                        all_moneyness_data = np.vstack([item[2] for item in selected_group])
                else:
                    all_moneyness_data = np.zeros_like(all_dtm_data)
                
                print(f"Kích thước dữ liệu cuối cùng: {all_iv_data.shape}")
                return all_iv_data, all_dtm_data, all_moneyness_data
            
        # Nếu không có dữ liệu hợp lệ, tạo dữ liệu giả
        print("Không thể đọc dữ liệu từ các file CSV hoặc không có dữ liệu hợp lệ. Tạo dữ liệu giả.")
        fake_iv = np.random.rand(20, 11) * 0.2 + 0.1
        fake_dtm = np.ones(20) * 30
        fake_moneyness = np.zeros(20)
        return fake_iv, fake_dtm, fake_moneyness
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        iv_surface = self.data[idx].astype(np.float32)
        dtm = self.dtm[idx].astype(np.float32)
        
        # Nếu moneyness là ma trận 2D, lấy hàng tương ứng
        if len(self.moneyness.shape) > 1 and self.moneyness.shape[0] == len(self.data):
            moneyness = self.moneyness[idx].astype(np.float32)
        else:
            # Nếu moneyness là vector 1D, lấy phần tử tương ứng
            moneyness = self.moneyness[idx].astype(np.float32) if idx < len(self.moneyness) else 0.0
        
        # Tạo mặt nạ (mask) cho các giá trị bị thiếu
        mask = np.random.choice([0, 1], size=iv_surface.shape, p=[self.missing_ratio, 1-self.missing_ratio])
        
        # Nhân với mặt nạ để tạo dữ liệu bị thiếu (0 = thiếu, giá trị khác = có dữ liệu)
        masked_iv = iv_surface * mask
        
        # Chuyển sang tensor và thêm chiều kênh
        iv_surface = torch.tensor(iv_surface, dtype=torch.float32).unsqueeze(0)
        masked_iv = torch.tensor(masked_iv, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        
        # Thêm thông tin DTM và moneyness
        dtm_tensor = torch.tensor(dtm, dtype=torch.float32)
        moneyness_tensor = torch.tensor(moneyness, dtype=torch.float32)
        
        if self.transform:
            iv_surface = self.transform(iv_surface)
            masked_iv = self.transform(masked_iv)
        
        return {
            'original': iv_surface, 
            'masked': masked_iv, 
            'mask': mask,
            'dtm': dtm_tensor,
            'moneyness': moneyness_tensor
        }

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
    # Tạo thư mục lưu trữ mô hình
    os.makedirs(save_path, exist_ok=True)
    
    # Tách dataset thành training và validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Tạo DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Lấy mẫu đầu tiên để xác định kích thước dữ liệu
    sample = next(iter(train_loader))
    input_shape = sample['original'].shape[-1]
    print(f"Kích thước dữ liệu đầu vào: {input_shape}")
    
    # Khởi tạo mô hình hoặc tải mô hình đã lưu
    if resume and model_path and os.path.exists(model_path):
        print(f"Tiếp tục huấn luyện từ mô hình {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model = ImprovedDiffusionUNet(in_channels=2, out_channels=1).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        
        # Kiểm tra nếu dữ liệu quá nhỏ, bỏ qua lớp downsample cuối
        if input_shape <= 4:
            model.skip_last_downsample = True
            print("Dữ liệu nhỏ, đã bỏ qua lớp downsample cuối cùng")
        
        # Khởi tạo Diffusion model với mô hình đã tải
        # Tăng số bước khuếch tán từ 100 lên 200
        diffusion = ImprovedDiffusionModel(model, num_diffusion_steps=200)
        
        # Khởi tạo optimizer
        optimizer = Adam(model.parameters(), lr=lr)
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Đã tải trạng thái optimizer")
            except Exception as e:
                print(f"Không thể tải trạng thái optimizer: {e}")
        
        # Tải best metrics nếu có
        best_loss = checkpoint.get('best_loss', float('inf'))
    else:
        print("Khởi tạo mô hình mới")
        model = ImprovedDiffusionUNet(in_channels=2, out_channels=1).to(device)
        
        # Kiểm tra nếu dữ liệu quá nhỏ, bỏ qua lớp downsample cuối
        if input_shape <= 4:
            model.skip_last_downsample = True
            print("Dữ liệu nhỏ, đã bỏ qua lớp downsample cuối cùng")
        
        # Tăng số bước khuếch tán để cải thiện chất lượng
        diffusion = ImprovedDiffusionModel(model, num_diffusion_steps=200)
        optimizer = Adam(model.parameters(), lr=lr)
        start_epoch = 0
        best_loss = float('inf')
    
    # Thêm learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # Thiết lập early stopping
    patience = 12
    counter = 0
    early_stop = False
    
    # Lưu loss để theo dõi
    losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(start_epoch, start_epoch + epochs):
        if early_stop:
            print("Early stopping triggered!")
            break
            
        model.train()
        epoch_loss = 0.0
        
        # Huấn luyện một epoch
        for batch in train_loader:
            optimizer.zero_grad()
            
            x_0 = batch['original'].to(device)
            mask = batch['mask'].to(device)
            
            loss = diffusion.p_losses(x_0, mask)
            loss.backward()
            
            # Gradient clipping để tránh exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        # Tính loss trung bình
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_0 = batch['original'].to(device)
                mask = batch['mask'].to(device)
                
                loss = diffusion.p_losses(x_0, mask)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Cập nhật learning rate scheduler
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{start_epoch+epochs}, Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Lưu mô hình tốt nhất dựa trên validation loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            counter = 0
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'val_loss': avg_val_loss,
                'best_loss': best_loss
            }
            
            torch.save(checkpoint, os.path.join(save_path, 'best_model.pth'))
            print(f"Đã lưu mô hình tốt nhất với val_loss = {best_loss:.6f}")
        else:
            counter += 1
            if counter >= patience:
                early_stop = True
                print(f"Early stopping sau {patience} epochs không cải thiện")
        
        # Lưu checkpoint
        if (epoch + 1) % 5 == 0 or (epoch + 1) == start_epoch + epochs:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'val_loss': avg_val_loss,
                'best_loss': best_loss
            }
            
            torch.save(checkpoint, os.path.join(save_path, f'model_epoch_{epoch+1}.pth'))
            print(f"Đã lưu checkpoint tại epoch {epoch+1}")
    
    # Vẽ biểu đồ loss
    plt.figure(figsize=(12, 6))
    plt.plot(losses, label='Training Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    
    # Lưu biểu đồ
    plt.savefig(os.path.join(save_path, 'loss_curve.png'))
    plt.close()
    
    print(f"Đã hoàn thành huấn luyện! Mô hình được lưu tại {save_path}")
    
    # Tải mô hình tốt nhất cho việc trả về
    best_model_path = os.path.join(save_path, 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Đã tải mô hình tốt nhất với val_loss = {checkpoint.get('best_loss', 'N/A')}")
    
    return model, diffusion

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
    
    # In kích thước ban đầu để debug
    print(f"generate_iv_surface - Đầu vào: masked_iv.shape={masked_iv.shape}, mask.shape={mask.shape}")
    
    # Chuyển đổi sang tensor và đưa lên thiết bị
    if not torch.is_tensor(masked_iv):
        masked_iv = torch.tensor(masked_iv, dtype=torch.float32)
    if not torch.is_tensor(mask):
        mask = torch.tensor(mask, dtype=torch.float32)
    
    # Chuẩn hóa tensor về [B, C, L] (batch, channel, length)
    if len(masked_iv.shape) == 1:  # [L]
        masked_iv = masked_iv.unsqueeze(0).unsqueeze(0)  # [1, 1, L]
    elif len(masked_iv.shape) == 2:  # [B, L] or [1, L]
        masked_iv = masked_iv.unsqueeze(1)  # [B, 1, L]
        
    if len(mask.shape) == 1:  # [L]
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L]
    elif len(mask.shape) == 2:  # [B, L] or [1, L]
        mask = mask.unsqueeze(1)  # [B, 1, L]
    
    # Đưa lên thiết bị
    masked_iv = masked_iv.to(device)
    mask = mask.to(device)
    
    print(f"generate_iv_surface - Sau chuẩn hóa: masked_iv.shape={masked_iv.shape}, mask.shape={mask.shape}")
    
    # Sinh bề mặt IV với ensemble (tạo nhiều mẫu và lấy trung bình)
    try:
        # Số mẫu ensemble
        num_samples = 3
        all_samples = []
        
        for i in range(num_samples):
            print(f"Tạo mẫu {i+1}/{num_samples}...")
            sample = diffusion_model.sample(masked_iv, mask)
            all_samples.append(sample)
        
        # Tính trung bình các mẫu để giảm nhiễu
        if num_samples > 1:
            reconstructed = torch.mean(torch.stack(all_samples), dim=0)
            print(f"Đã tính trung bình {num_samples} mẫu")
        else:
            reconstructed = all_samples[0]
            
        print(f"generate_iv_surface - Kết quả: reconstructed.shape={reconstructed.shape}")
    
        # Kết hợp dữ liệu gốc và dữ liệu phục hồi
        result = torch.where(mask > 0, masked_iv, reconstructed)
        
        # Áp dụng smooth để cải thiện chất lượng phục hồi
        # Chuyển về numpy để áp dụng smoothing
        result_np = result.cpu().squeeze().numpy()
        
        # Nếu result_np là 1D, reshape thành 2D
        if len(result_np.shape) == 1:
            result_np = result_np.reshape(1, -1)
            
        # Áp dụng gaussian filter để smooth kết quả (tùy chọn)
        try:
            from scipy.ndimage import gaussian_filter
            result_smooth = gaussian_filter(result_np, sigma=0.5)
            print("Đã áp dụng gaussian smoothing")
            return result_smooth
        except:
            print("Không thể áp dụng smoothing, trả về kết quả gốc")
            return result_np
        
    except Exception as e:
        print(f"Lỗi trong generate_iv_surface: {e}")
        import traceback
        traceback.print_exc()
        
        # Trả về dữ liệu gốc nếu có lỗi
        print("Trả về dữ liệu gốc do có lỗi trong quá trình phục hồi")
        return masked_iv.cpu().squeeze().numpy()

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
    
    # Xử lý trường hợp có thêm chiều
    print(f"plot_iv_surface - shape đầu vào: {iv_data.shape}")
    
    # Xử lý các trường hợp khác nhau về shape
    if len(iv_data.shape) == 1:
        # Nếu chỉ có 1 chiều, reshape thành 2D (1 x L)
        iv_data = iv_data.reshape(1, -1)
        print(f"Đã reshape 1D -> 2D: {iv_data.shape}")
    elif len(iv_data.shape) == 3:
        # Nếu có 3 chiều (C x H x W), squeeze chiều channel
        iv_data = iv_data.squeeze(0)
        print(f"Đã squeeze chiều channel: {iv_data.shape}")
    elif len(iv_data.shape) > 3:
        # Nếu nhiều hơn 3 chiều, cố gắng squeeze và reshape
        iv_data = iv_data.squeeze()
        if len(iv_data.shape) > 2:
            # Nếu vẫn > 2 chiều, lấy 2 chiều cuối
            shape = iv_data.shape
            iv_data = iv_data.reshape(shape[-2], shape[-1])
        print(f"Đã xử lý tensor nhiều chiều: {iv_data.shape}")
    
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
    
    # In kích thước để debug
    print(f"evaluate_reconstruction - original.shape: {original.shape}, reconstructed.shape: {reconstructed.shape}")
    
    # Loại bỏ chiều thừa
    if len(original.shape) > 2:
        # Sử dụng squeeze() để loại bỏ các chiều có kích thước 1
        original = original.squeeze()
        # Nếu còn hơn 2 chiều, dùng reshape
        if len(original.shape) > 2:
            # Lấy 2 chiều cuối
            shape = original.shape
            original = original.reshape(-1, shape[-1])
            
    if len(reconstructed.shape) > 2:
        # Sử dụng squeeze() để loại bỏ các chiều có kích thước 1
        reconstructed = reconstructed.squeeze()
        # Nếu còn hơn 2 chiều, dùng reshape
        if len(reconstructed.shape) > 2:
            # Lấy 2 chiều cuối
            shape = reconstructed.shape
            reconstructed = reconstructed.reshape(-1, shape[-1])
    
    # Nếu original hoặc reconstructed chỉ còn 1 chiều, reshape thành 2D
    if len(original.shape) == 1:
        original = original.reshape(1, -1)
    if len(reconstructed.shape) == 1:
        reconstructed = reconstructed.reshape(1, -1)
        
    print(f"evaluate_reconstruction - Sau khi xử lý: original.shape: {original.shape}, reconstructed.shape: {reconstructed.shape}")
    
    # Đảm bảo cả hai mảng có cùng kích thước
    if original.shape != reconstructed.shape:
        # Cắt giảm về kích thước nhỏ hơn nếu khác nhau
        min_rows = min(original.shape[0], reconstructed.shape[0])
        min_cols = min(original.shape[1], reconstructed.shape[1])
        original = original[:min_rows, :min_cols]
        reconstructed = reconstructed[:min_rows, :min_cols]
        print(f"Đã điều chỉnh để cùng kích thước: {original.shape}")
    
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
    - num_samples: Số lượng mẫu cho ensemble averaging
    """
    # Tham số có thể điều chỉnh
    missing_ratio = 0.3  # 30% dữ liệu bị thiếu
    train_model = True   # Huấn luyện mô hình mới
    resume_training = False  # Tiếp tục huấn luyện từ mô hình đã lưu
    model_path = "models/best_model.pth"  # Đường dẫn đến mô hình đã lưu
    num_samples = 5  # Số lượng mẫu cho ensemble averaging
    
    # Tạo thư mục kết quả
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Tìm kiếm tất cả các file CSV từ dữ liệu cũ và mới
    all_csv_files = []
    
    # 1. Tìm các file pivot_df_*.csv từ dữ liệu cũ
    old_csv_files = [f for f in os.listdir() if f.startswith('pivot_df_') and f.endswith('.csv')]
    all_csv_files.extend(old_csv_files)
    
    # 2. Tìm các file CSV mới từ dữ liệu SPY
    spy_csv_files = []
    
    # Tìm file pivot_SPY_*.csv hoặc SPY_*.csv
    spy_patterns = ['pivot_SPY_', 'SPY_', 'pivot_spy_', 'spy_']
    for pattern in spy_patterns:
        found_files = [f for f in os.listdir() if f.startswith(pattern) and f.endswith('.csv')]
        spy_csv_files.extend(found_files)
    
    # 3. Tìm các file CSV mới từ dữ liệu AAPL
    aapl_csv_files = []
    
    # Tìm file pivot_AAPL_*.csv hoặc AAPL_*.csv
    aapl_patterns = ['pivot_AAPL_', 'AAPL_', 'pivot_aapl_', 'aapl_']
    for pattern in aapl_patterns:
        found_files = [f for f in os.listdir() if f.startswith(pattern) and f.endswith('.csv')]
        aapl_csv_files.extend(found_files)
    
    # Gộp tất cả các file CSV mới
    new_csv_files = spy_csv_files + aapl_csv_files
    
    # Chia dữ liệu mới thành tập train và test
    if new_csv_files:
        # Chỉ sử dụng một phần các mức giá để huấn luyện (mỗi file là một mức giá)
        new_train_files = sorted(new_csv_files)[::2]  # Lấy một nửa các file (các vị trí chẵn)
        new_test_files = sorted(new_csv_files)[1::2]  # Lấy một nửa các file (các vị trí lẻ)
    else:
        new_train_files = []
        new_test_files = []
    
    print(f"Tìm thấy {len(old_csv_files)} file CSV từ dữ liệu cũ")
    print(f"Tìm thấy {len(spy_csv_files)} file CSV từ dữ liệu SPY mới")
    print(f"Tìm thấy {len(aapl_csv_files)} file CSV từ dữ liệu AAPL mới")
    print(f"Dùng {len(new_train_files)} file mới cho huấn luyện và {len(new_test_files)} file mới cho kiểm thử")
    
    # Kết hợp dữ liệu cũ và dữ liệu mới cho việc huấn luyện
    training_files = old_csv_files + new_train_files
    print(f"Tổng số file CSV sử dụng cho huấn luyện: {len(training_files)}")
    
    # Tạo dataset với dữ liệu kết hợp
    dataset = RealIVSurfaceDataset(csv_files=training_files, missing_ratio=missing_ratio, normalize=True)
    print(f"Đã tạo dataset với {len(dataset)} mẫu")
    
    # Tạo thư mục lưu mô hình
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Tùy chọn 1: Huấn luyện mô hình mới hoặc tiếp tục huấn luyện
    if train_model:
        if resume_training and os.path.exists(model_path):
            print(f"Tiếp tục huấn luyện từ mô hình {model_path}...")
            model, diffusion = train_diffusion_model(dataset, epochs=50, batch_size=8, lr=1e-4, resume=True, model_path=model_path)
        else:
            print("Bắt đầu huấn luyện mô hình mới...")
            model, diffusion = train_diffusion_model(dataset, epochs=50, batch_size=8, lr=1e-4)
    else:
        print(f"Tải mô hình đã lưu từ {model_path}...")
        # Tải mô hình đã lưu
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model = ImprovedDiffusionUNet(in_channels=2, out_channels=1).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            diffusion = ImprovedDiffusionModel(model, num_diffusion_steps=200)
            print("Đã tải mô hình thành công!")
        else:
            print(f"Không tìm thấy mô hình tại {model_path}. Huấn luyện mô hình mới...")
            model, diffusion = train_diffusion_model(dataset, epochs=50, batch_size=8, lr=1e-4)
    
    # Đánh giá mô hình với dữ liệu huấn luyện
    print("Đánh giá mô hình với dữ liệu huấn luyện...")
    evaluate_model(model, diffusion, dataset, results_dir, "train", num_samples, 5)
    
    # Đánh giá mô hình với dữ liệu mới (không dùng để huấn luyện)
    if new_test_files:
        print("Đánh giá mô hình với dữ liệu mới...")
        test_dataset = RealIVSurfaceDataset(csv_files=new_test_files, missing_ratio=missing_ratio, normalize=True)
        evaluate_model(model, diffusion, test_dataset, results_dir, "test_new", num_samples, 5)
    
    # Kiểm tra mô hình với tỷ lệ dữ liệu bị thiếu cao hơn
    high_missing_evaluate(model, diffusion, dataset, results_dir, num_samples)
    
    print("\nĐã hoàn thành đánh giá mô hình và lưu kết quả vào thư mục results/")
    print(f"Xem kết quả tại: {os.path.abspath(results_dir)}")

def evaluate_model(model, diffusion, dataset, results_dir, prefix, num_samples, test_size=5):
    """
    Đánh giá mô hình với tập dữ liệu cho trước
    
    Tham số:
        model: Mô hình UNet
        diffusion: Mô hình Diffusion
        dataset: Tập dữ liệu để đánh giá
        results_dir: Thư mục lưu kết quả
        prefix: Tiền tố cho tên file kết quả
        num_samples: Số lượng mẫu cho ensemble
        test_size: Số lượng mẫu kiểm thử
    """
    # Lấy số mẫu ngẫu nhiên để đánh giá
    test_size = min(test_size, len(dataset))
    test_indices = np.random.choice(len(dataset), test_size, replace=False)
    
    # Kết quả đánh giá
    metrics_all = {
        'mse': [],
        'rmse': [],
        'mae': [],
        'mape': []
    }
    
    for i, idx in enumerate(test_indices):
        print(f"\nĐánh giá mẫu {i+1}/{test_size} (index={idx}):")
        sample = dataset[idx]
        
        # Chuẩn bị dữ liệu
        original = sample['original'].unsqueeze(0).to(device)  # Thêm batch dimension
        masked = sample['masked'].unsqueeze(0).to(device)
        mask = sample['mask'].unsqueeze(0).to(device)
        
        # Phục hồi bề mặt IV với ensemble averaging
        print(f"Phục hồi bề mặt IV với {num_samples} mẫu ensemble...")
        model.eval()
        with torch.no_grad():
            reconstructed = diffusion.sample(masked, mask, num_samples=num_samples)
        
        # Chuyển về numpy
        original_np = original.cpu().squeeze().numpy()
        masked_np = masked.cpu().squeeze().numpy()
        reconstructed_np = reconstructed.cpu().squeeze().numpy()
        
        # Vẽ bề mặt gốc, bị thiếu và phục hồi
        # Bề mặt gốc
        plot_iv_surface(
            original_np, 
            title=f"Original IV Surface - {prefix} Sample {i+1}", 
            save_path=os.path.join(results_dir, f"{prefix}_sample_{i+1}_original.png")
        )
        
        # Bề mặt bị thiếu
        plot_iv_surface(
            masked_np, 
            title=f"Masked IV Surface - {prefix} Sample {i+1}", 
            save_path=os.path.join(results_dir, f"{prefix}_sample_{i+1}_masked.png")
        )
        
        # Bề mặt phục hồi
        plot_iv_surface(
            reconstructed_np, 
            title=f"Reconstructed IV Surface - {prefix} Sample {i+1}", 
            save_path=os.path.join(results_dir, f"{prefix}_sample_{i+1}_reconstructed.png")
        )
        
        # Đánh giá kết quả phục hồi
        print(f"Đánh giá kết quả phục hồi mẫu {i+1}:")
        metrics = evaluate_reconstruction(original_np, reconstructed_np)
        
        # Thêm vào kết quả tổng hợp
        for key in metrics:
            metrics_all[key].append(metrics[key])
    
    # Tính trung bình các metrics
    print(f"\n==== Kết quả đánh giá tổng hợp cho {prefix} ====")
    for key in metrics_all:
        avg_value = np.mean(metrics_all[key])
        print(f"Trung bình {key.upper()}: {avg_value:.6f}")
    
    # Lưu metrics vào file
    metrics_df = pd.DataFrame(metrics_all)
    metrics_df.to_csv(os.path.join(results_dir, f"{prefix}_metrics.csv"), index=False)
    
    return metrics_all

def high_missing_evaluate(model, diffusion, dataset, results_dir, num_samples):
    """
    Đánh giá mô hình với tỷ lệ dữ liệu bị thiếu cao
    
    Tham số:
        model: Mô hình UNet
        diffusion: Mô hình Diffusion
        dataset: Tập dữ liệu để đánh giá
        results_dir: Thư mục lưu kết quả
        num_samples: Số lượng mẫu cho ensemble
    """
    # Tạo bộ dữ liệu mới với tỷ lệ dữ liệu bị thiếu lớn hơn
    high_missing_ratio = 0.5  # 50% dữ liệu bị thiếu
    print(f"\nKiểm tra mô hình với tỷ lệ dữ liệu bị thiếu cao hơn ({high_missing_ratio*100}%)...")
    
    # Sử dụng cùng các file CSV như dataset gốc, nhưng với tỷ lệ missing cao hơn
    if hasattr(dataset, 'csv_files'):
        test_dataset = RealIVSurfaceDataset(csv_files=dataset.csv_files, missing_ratio=high_missing_ratio, normalize=True)
    else:
        test_dataset = RealIVSurfaceDataset(missing_ratio=high_missing_ratio, normalize=True)
    
    # Lấy một mẫu ngẫu nhiên
    idx = np.random.randint(0, len(test_dataset))
    sample = test_dataset[idx]
    
    # Chuẩn bị dữ liệu
    original = sample['original'].unsqueeze(0).to(device)
    masked = sample['masked'].unsqueeze(0).to(device)
    mask = sample['mask'].unsqueeze(0).to(device)
    
    # Phục hồi bề mặt IV với ensemble averaging
    print(f"Phục hồi bề mặt IV với tỷ lệ dữ liệu bị thiếu cao ({num_samples} mẫu ensemble)...")
    model.eval()
    with torch.no_grad():
        reconstructed = diffusion.sample(masked, mask, num_samples=num_samples)
    
    # Chuyển về numpy
    original_np = original.cpu().squeeze().numpy()
    masked_np = masked.cpu().squeeze().numpy()
    reconstructed_np = reconstructed.cpu().squeeze().numpy()
    
    # Vẽ bề mặt gốc, bị thiếu và phục hồi
    # Bề mặt gốc
    plot_iv_surface(
        original_np, 
        title=f"Original IV Surface - High Missing Ratio", 
        save_path=os.path.join(results_dir, "high_missing_original.png")
    )
    
    # Bề mặt bị thiếu
    plot_iv_surface(
        masked_np, 
        title=f"Masked IV Surface - High Missing Ratio ({high_missing_ratio*100}%)", 
        save_path=os.path.join(results_dir, "high_missing_masked.png")
    )
    
    # Bề mặt phục hồi
    plot_iv_surface(
        reconstructed_np, 
        title=f"Reconstructed IV Surface - High Missing Ratio", 
        save_path=os.path.join(results_dir, "high_missing_reconstructed.png")
    )
    
    # Đánh giá kết quả phục hồi
    print("Đánh giá kết quả phục hồi với tỷ lệ dữ liệu bị thiếu cao:")
    high_missing_metrics = evaluate_reconstruction(original_np, reconstructed_np)
    
    # Lưu metrics vào file
    metrics_df = pd.DataFrame({k: [v] for k, v in high_missing_metrics.items()})
    metrics_df.to_csv(os.path.join(results_dir, "high_missing_metrics.csv"), index=False)

if __name__ == "__main__":
    main() 