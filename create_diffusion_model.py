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

# Thiết lập seed cho quá trình tái tạo
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Kiểm tra thiết bị có sẵn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {device}")

# Định nghĩa lớp dataset cho bề mặt Implied Volatility
class IVSurfaceDataset(Dataset):
    def __init__(self, csv_files, transform=None, missing_ratio=0.2):
        """
        Dataset cho bề mặt Implied Volatility
        
        Tham số:
            csv_files (list): Danh sách các file CSV chứa dữ liệu
            transform (callable, optional): Biến đổi dữ liệu
            missing_ratio (float): Tỷ lệ dữ liệu bị thiếu (0.0 - 1.0)
        """
        self.transform = transform
        self.missing_ratio = missing_ratio
        
        # Đọc và xử lý dữ liệu từ các file CSV
        self.data = self._load_data(csv_files)
        
    def _load_data(self, csv_files):
        """Đọc dữ liệu từ nhiều file CSV và tổ chức thành ma trận 2D"""
        # Đọc pivot_csv.csv để lấy cấu trúc dữ liệu
        pivot_df = pd.read_csv('pivot_csv.csv')
        
        # Lấy các cột IV
        iv_cols = [col for col in pivot_df.columns if col.startswith('IV_')]
        
        # Chuyển dữ liệu thành tensor
        data = []
        
        for _, row in pivot_df.iterrows():
            # Lấy giá trị IV từ các cột
            iv_values = row[iv_cols].values
            
            # Thêm vào danh sách dữ liệu
            data.append(iv_values)
        
        return np.array(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        iv_surface = self.data[idx].astype(np.float32)
        
        # Tạo mặt nạ (mask) cho các giá trị bị thiếu
        mask = np.random.choice([0, 1], size=iv_surface.shape, p=[self.missing_ratio, 1-self.missing_ratio])
        
        # Nhân với mặt nạ để tạo dữ liệu bị thiếu (0 = thiếu, giá trị khác = có dữ liệu)
        masked_iv = iv_surface * mask
        
        # Chuyển sang tensor
        iv_surface = torch.tensor(iv_surface, dtype=torch.float32)
        masked_iv = torch.tensor(masked_iv, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        
        if self.transform:
            iv_surface = self.transform(iv_surface)
            masked_iv = self.transform(masked_iv)
        
        return {'original': iv_surface, 'masked': masked_iv, 'mask': mask}

# Mô hình U-Net cho Diffusion model
class DiffusionUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, time_emb_dim=32):
        super(DiffusionUNet, self).__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Lưu lại số chiều của time_emb_dim
        self.time_emb_dim = time_emb_dim
        
        # Encoder - input là concat của x và time embedding
        total_channels = in_channels + time_emb_dim  # Tổng số kênh sau khi ghép
        
        self.enc1 = self._make_conv_block(total_channels, 32)
        self.enc2 = self._make_conv_block(32, 64)
        self.enc3 = self._make_conv_block(64, 128)
        
        # Bottleneck
        self.bottleneck = self._make_conv_block(128, 256)
        
        # Decoder
        self.dec3 = self._make_conv_block(256 + 128, 128)
        self.dec2 = self._make_conv_block(128 + 64, 64)
        self.dec1 = self._make_conv_block(64 + 32, 32)
        
        # Output layer
        self.final = nn.Conv1d(32, out_channels, kernel_size=1)
        
        # Downsample and upsample
        self.down = nn.MaxPool1d(2)
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
    
    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, t, mask):
        """
        x: Tensor [B, C, L] - Thông tin đầu vào (nhiễu và mặt nạ)
        t: Tensor [B, 1] - Bước thời gian (time step)
        mask: Tensor [B, 1, L] - Mặt nạ cho biết giá trị nào bị thiếu
        """
        # Thời gian embedding
        t_emb = self.time_mlp(t.unsqueeze(-1))
        t_emb = t_emb.unsqueeze(-1).repeat(1, 1, x.shape[-1])
        
        # Debug: In kích thước để kiểm tra
        # print(f"x shape: {x.shape}, t_emb shape: {t_emb.shape}")
        
        # Kết hợp t_emb vào x
        x = torch.cat([x, t_emb], dim=1)
        # print(f"After concatenation, x shape: {x.shape}")
        
        # Encoder
        e1 = self.enc1(x)
        e1_down = self.down(e1)
        
        e2 = self.enc2(e1_down)
        e2_down = self.down(e2)
        
        e3 = self.enc3(e2_down)
        e3_down = self.down(e3)
        
        # Bottleneck
        b = self.bottleneck(e3_down)
        
        # Decoder
        b_up = self.up(b)
        d3 = self.dec3(torch.cat([b_up, e3], dim=1))
        
        d3_up = self.up(d3)
        d2 = self.dec2(torch.cat([d3_up, e2], dim=1))
        
        d2_up = self.up(d2)
        d1 = self.dec1(torch.cat([d2_up, e1], dim=1))
        
        # Đầu ra
        output = self.final(d1)
        
        return output

# Lớp Diffusion Model
class DiffusionModel:
    def __init__(self, model, beta_start=1e-4, beta_end=0.02, num_diffusion_steps=1000):
        """
        Diffusion Model cho phục hồi bề mặt Implied Volatility
        
        Tham số:
            model: Mô hình U-Net
            beta_start: Beta ban đầu cho lịch trình nhiễu
            beta_end: Beta cuối cho lịch trình nhiễu
            num_diffusion_steps: Số bước khuếch tán
        """
        self.model = model
        self.num_diffusion_steps = num_diffusion_steps
        
        # Lịch trình beta
        self.betas = torch.linspace(beta_start, beta_end, num_diffusion_steps).to(device)
        
        # Tính toán các hệ số
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        
        # Tính các hệ số để lấy mẫu
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
    
    def q_sample(self, x_0, t, noise=None):
        """
        Thêm nhiễu Gaussian vào dữ liệu x_0 tại bước t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_0, mask, noise=None, loss_type="l2"):
        """
        Tính loss trong quá trình huấn luyện
        """
        batch_size = x_0.shape[0]
        
        # Chọn bước thời gian ngẫu nhiên
        t = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=device)
        
        if noise is None:
            noise = torch.randn_like(x_0)
            
        # Thêm nhiễu
        x_t = self.q_sample(x_0, t, noise)
        
        # Kết hợp nhiễu và mặt nạ làm đầu vào - chỉ truyền 2 kênh
        model_input = torch.cat([x_t, mask], dim=1)
        
        # Debug: In kích thước để kiểm tra
        # print(f"In p_losses: model_input shape: {model_input.shape}, t shape: {t.shape}")
        
        # Dự đoán nhiễu - forward sẽ xử lý time embedding bên trong
        noise_pred = self.model(model_input, t/self.num_diffusion_steps, mask)
        
        # Tính loss
        if loss_type == "l1":
            loss = F.l1_loss(noise_pred, noise)
        else:
            loss = F.mse_loss(noise_pred, noise)
            
        return loss
    
    @torch.no_grad()
    def p_sample(self, x_t, t, mask):
        """
        Lấy mẫu từ quá trình ngược
        """
        betas_t = self.betas[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas[t])
        
        # Kết hợp nhiễu và mặt nạ làm đầu vào
        model_input = torch.cat([x_t, mask], dim=1)
        
        # Dự đoán nhiễu
        model_output = self.model(model_input, t/self.num_diffusion_steps, mask)
        
        # Quá trình ngược cho phương trình khuếch tán
        model_mean = sqrt_recip_alphas * (
            x_t - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t
        )
        
        # Không thêm nhiễu ở bước cuối (t=0)
        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x_t)
            variance = torch.sqrt(self.posterior_variance[t])
            return model_mean + variance * noise
    
    @torch.no_grad()
    def sample(self, masked_iv, mask, num_samples=1):
        """
        Tạo mẫu hoàn chỉnh từ dữ liệu bị thiếu
        """
        batch_size = masked_iv.shape[0]
        shape = (batch_size, 1, masked_iv.shape[2])
        
        samples = []
        for _ in range(num_samples):
            # Khởi tạo từ nhiễu ngẫu nhiên
            img = torch.randn(shape, device=device)
            
            # Lặp qua các bước thời gian từ T về 0
            for t in reversed(range(self.num_diffusion_steps)):
                t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
                img = self.p_sample(img, t_batch, mask)
            
            # Kết hợp dữ liệu gốc (đã biết) và dữ liệu đã được tạo ra (bị thiếu)
            combined = mask * masked_iv + (1 - mask) * img
            samples.append(combined)
            
        return samples[0] if num_samples == 1 else torch.stack(samples)
    
    def fit(self, dataloader, epochs, optimizer, scheduler=None, save_path="models"):
        """
        Huấn luyện mô hình
        """
        self.model.train()
        
        os.makedirs(save_path, exist_ok=True)
        
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for batch in dataloader:
                optimizer.zero_grad()
                
                x_0 = batch['original'].to(device).unsqueeze(1)  # [B, 1, L]
                mask = batch['mask'].to(device).unsqueeze(1)     # [B, 1, L]
                
                loss = self.p_losses(x_0, mask)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if scheduler is not None:
                scheduler.step()
                
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            
            # Lưu mô hình sau mỗi 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save(
                    {
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'loss': avg_loss,
                    },
                    os.path.join(save_path, f"diffusion_model_epoch_{epoch+1}.pt")
                )
        
        # Vẽ biểu đồ loss
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title('Loss qua các epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(save_path, 'loss_curve.png'))
        plt.close()
        
        return losses

# Hàm kiểm tra để đảm bảo mô hình không tạo arbitrage
def check_arbitrage_free(iv_surface):
    """
    Kiểm tra các điều kiện để đảm bảo bề mặt IV là arbitrage-free
    
    Tham số:
        iv_surface: Bề mặt Implied Volatility (numpy array)
        
    Trả về:
        is_valid: Boolean, True nếu bề mặt IV thỏa mãn điều kiện
        violations: Dict, các vi phạm điều kiện nếu có
    """
    violations = {}
    
    # 1. IV không âm
    if np.any(iv_surface < 0):
        violations['negative_iv'] = np.sum(iv_surface < 0)
    
    # 2. Kiểm tra Butterfly arbitrage (hình dạng lồi theo strike)
    # Với mỗi hàng (cùng một maturity)
    butterfly_violations = 0
    for row in iv_surface:
        for i in range(1, len(row) - 1):
            if 2 * row[i] > row[i-1] + row[i+1]:
                butterfly_violations += 1
    
    if butterfly_violations > 0:
        violations['butterfly'] = butterfly_violations
    
    # 3. Kiểm tra Calendar arbitrage (tăng theo maturity)
    # Nếu có nhiều ngày đáo hạn khác nhau
    calendar_violations = 0
    for col in range(iv_surface.shape[1]):
        for i in range(1, iv_surface.shape[0]):
            if iv_surface[i, col] < iv_surface[i-1, col]:
                calendar_violations += 1
                
    if calendar_violations > 0:
        violations['calendar'] = calendar_violations
    
    is_valid = len(violations) == 0
    
    return is_valid, violations

# Hàm huấn luyện mô hình
def train_diffusion_model(csv_files, epochs=100, batch_size=32, lr=1e-4, missing_ratio=0.2):
    """
    Huấn luyện mô hình Diffusion
    
    Tham số:
        csv_files: Danh sách các file CSV chứa dữ liệu
        epochs: Số epoch huấn luyện
        batch_size: Kích thước batch
        lr: Learning rate
        missing_ratio: Tỷ lệ dữ liệu bị thiếu
    """
    # Tạo dataset và dataloader
    dataset = IVSurfaceDataset(csv_files, missing_ratio=missing_ratio)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Khởi tạo mô hình
    model = DiffusionUNet(in_channels=2, out_channels=1).to(device)
    
    # Khởi tạo diffusion model
    diffusion = DiffusionModel(model)
    
    # Khởi tạo optimizer
    optimizer = Adam(model.parameters(), lr=lr)
    
    # Huấn luyện
    losses = diffusion.fit(dataloader, epochs, optimizer)
    
    return diffusion, losses

# Hàm tạo bề mặt IV từ dữ liệu bị thiếu
def generate_iv_surface(diffusion_model, masked_iv, mask):
    """
    Tạo bề mặt IV hoàn chỉnh từ dữ liệu bị thiếu
    
    Tham số:
        diffusion_model: Mô hình Diffusion đã huấn luyện
        masked_iv: Tensor bề mặt IV bị thiếu
        mask: Tensor mặt nạ (1 = có dữ liệu, 0 = thiếu)
    """
    # Chuyển sang thiết bị tính toán
    masked_iv = masked_iv.to(device)
    mask = mask.to(device)
    
    # Lấy mẫu từ mô hình
    generated_iv = diffusion_model.sample(masked_iv, mask)
    
    return generated_iv

# Hàm đánh giá chất lượng phục hồi
def evaluate_reconstruction(original, reconstructed):
    """
    Đánh giá chất lượng phục hồi bề mặt IV
    
    Tham số:
        original: Bề mặt IV gốc
        reconstructed: Bề mặt IV được phục hồi
    """
    # Chuyển sang CPU và NumPy nếu cần
    if torch.is_tensor(original):
        original = original.cpu().numpy()
    if torch.is_tensor(reconstructed):
        reconstructed = reconstructed.cpu().numpy()
    
    # Tính MSE
    mse = np.mean((original - reconstructed) ** 2)
    
    # Tính RMSE
    rmse = np.sqrt(mse)
    
    # Tính MAE
    mae = np.mean(np.abs(original - reconstructed))
    
    # Tính MAPE
    epsilon = 1e-10  # Để tránh chia cho 0
    mape = np.mean(np.abs((original - reconstructed) / (original + epsilon))) * 100
    
    # Kiểm tra arbitrage-free
    is_valid, violations = check_arbitrage_free(reconstructed)
    
    results = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'Arbitrage-free': is_valid,
        'Violations': violations
    }
    
    return results

# Hàm vẽ bề mặt Implied Volatility
def plot_iv_surface(iv_surface, title="Implied Volatility Surface", save_path=None):
    """
    Vẽ bề mặt Implied Volatility
    
    Tham số:
        iv_surface: Ma trận bề mặt IV có kích thước [M, N]
            M: Số ngày đáo hạn (maturities)
            N: Số giá strike khác nhau
        title: Tiêu đề biểu đồ
        save_path: Đường dẫn để lưu biểu đồ (None = không lưu)
    """
    # Chuyển sang CPU và NumPy nếu cần
    if torch.is_tensor(iv_surface):
        iv_surface = iv_surface.cpu().numpy()
    
    # Nếu tensor có 3 chiều, lấy chiều cuối cùng
    if len(iv_surface.shape) == 3:
        iv_surface = iv_surface.squeeze(1)
    
    # Tạo lưới giá trị strike và maturity
    strikes = np.arange(iv_surface.shape[1])
    maturities = np.arange(iv_surface.shape[0])
    
    # Tạo lưới 2D
    X, Y = np.meshgrid(strikes, maturities)
    
    # Vẽ biểu đồ 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Vẽ bề mặt
    surf = ax.plot_surface(X, Y, iv_surface, cmap='viridis', edgecolor='none')
    
    # Thêm thanh màu
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Đặt nhãn và tiêu đề
    ax.set_xlabel('Strike Index')
    ax.set_ylabel('Maturity Index')
    ax.set_zlabel('Implied Volatility')
    ax.set_title(title)
    
    # Lưu biểu đồ nếu được chỉ định
    if save_path:
        plt.savefig(save_path)
    
    plt.close()

# Hàm chính để chạy mô hình
def main():
    """Hàm chính để chạy và đánh giá mô hình"""
    
    # Danh sách file CSV chứa dữ liệu IV
    csv_files = [f for f in os.listdir('.') if f.startswith('pivot_df_')]
    
    print(f"Tìm thấy {len(csv_files)} file dữ liệu")
    
    # Huấn luyện mô hình
    print("Bắt đầu huấn luyện mô hình...")
    diffusion_model, losses = train_diffusion_model(
        csv_files, 
        epochs=50,  # Giảm số epoch để demo
        batch_size=16, 
        lr=1e-4, 
        missing_ratio=0.3
    )
    
    # Đánh giá mô hình trên tập kiểm thử
    print("Đánh giá mô hình...")
    
    # Tạo dataset kiểm thử
    test_dataset = IVSurfaceDataset(csv_files, missing_ratio=0.3)
    
    # Lấy một mẫu để kiểm tra
    sample = test_dataset[0]
    original = sample['original'].unsqueeze(0)
    masked = sample['masked'].unsqueeze(0)
    mask = sample['mask'].unsqueeze(0)
    
    # Phục hồi bề mặt IV
    reconstructed = generate_iv_surface(diffusion_model, masked, mask)
    
    # Đánh giá kết quả
    results = evaluate_reconstruction(original, reconstructed)
    print("Kết quả đánh giá:")
    for metric, value in results.items():
        if metric != 'Violations':
            print(f"{metric}: {value}")
    
    # Vẽ các biểu đồ so sánh
    os.makedirs("results", exist_ok=True)
    
    # Vẽ bề mặt gốc
    plot_iv_surface(
        original.squeeze(1),
        "Bề mặt IV gốc",
        "results/original_iv.png"
    )
    
    # Vẽ bề mặt bị thiếu
    plot_iv_surface(
        masked.squeeze(1),
        "Bề mặt IV bị thiếu",
        "results/masked_iv.png"
    )
    
    # Vẽ bề mặt được phục hồi
    plot_iv_surface(
        reconstructed.squeeze(1),
        "Bề mặt IV được phục hồi",
        "results/reconstructed_iv.png"
    )
    
    print("Đã lưu các biểu đồ vào thư mục 'results'")

if __name__ == "__main__":
    main() 