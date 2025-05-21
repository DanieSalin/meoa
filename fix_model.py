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

# Định nghĩa lớp dataset cho bề mặt Implied Volatility (đơn giản hóa)
class SimpleIVSurfaceDataset(Dataset):
    def __init__(self, missing_ratio=0.2):
        """
        Dataset đơn giản cho bề mặt Implied Volatility
        
        Tham số:
            missing_ratio (float): Tỷ lệ dữ liệu bị thiếu (0.0 - 1.0)
        """
        self.missing_ratio = missing_ratio
        
        # Tạo dữ liệu giả - 20 mẫu, mỗi mẫu có 11 điểm (tương ứng với số strike price)
        self.data = np.random.rand(20, 11) * 0.2 + 0.1  # IV thường nằm trong khoảng 0.1-0.3
        
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
        
        return {'original': iv_surface, 'masked': masked_iv, 'mask': mask}

# Mô hình U-Net đã sửa lỗi cho Diffusion model
class FixedDiffusionUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, time_emb_dim=32):
        super(FixedDiffusionUNet, self).__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Lưu lại số chiều
        self.time_emb_dim = time_emb_dim
        self.in_channels = in_channels
        
        # Encoder - input là concat của x và time embedding
        total_channels = in_channels + time_emb_dim
        
        # Encoder (kết hợp các lớp convolution và InstanceNorm thay vì BatchNorm)
        # InstanceNorm không bị ảnh hưởng bởi batch size = 1
        self.enc1 = nn.Sequential(
            nn.Conv1d(total_channels, 32, kernel_size=3, padding=1),
            nn.InstanceNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.InstanceNorm1d(32),
            nn.ReLU()
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.InstanceNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm1d(64),
            nn.ReLU()
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.InstanceNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.InstanceNorm1d(128),
            nn.ReLU()
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.InstanceNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.InstanceNorm1d(256),
            nn.ReLU()
        )
        
        # Decoder
        self.dec3 = nn.Sequential(
            nn.Conv1d(256 + 128, 128, kernel_size=3, padding=1),
            nn.InstanceNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.InstanceNorm1d(128),
            nn.ReLU()
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv1d(128 + 64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm1d(64),
            nn.ReLU()
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv1d(64 + 32, 32, kernel_size=3, padding=1),
            nn.InstanceNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.InstanceNorm1d(32),
            nn.ReLU()
        )
        
        # Output layer
        self.final = nn.Conv1d(32, out_channels, kernel_size=1)
        
        # Downsample and upsample
        self.down = nn.MaxPool1d(2)
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
    
    def forward(self, x, t, mask=None):
        """
        x: Tensor [B, C, L] - Thông tin đầu vào (nhiễu và mặt nạ)
        t: Tensor [B, 1] - Bước thời gian (time step)
        mask: Tensor [B, 1, L] - Mặt nạ cho biết giá trị nào bị thiếu
        """
        # Thời gian embedding
        t_emb = self.time_mlp(t.unsqueeze(-1))
        t_emb = t_emb.unsqueeze(-1).repeat(1, 1, x.shape[-1])
        
        # Kết hợp t_emb vào x
        x = torch.cat([x, t_emb], dim=1)
        
        # Debug: In kích thước để giúp việc phát hiện lỗi
        # print(f"Input x shape: {x.shape}")
        
        # Encoder
        e1 = self.enc1(x)
        # print(f"e1 shape: {e1.shape}")
        e1_down = self.down(e1)
        # print(f"e1_down shape: {e1_down.shape}")
        
        e2 = self.enc2(e1_down)
        # print(f"e2 shape: {e2.shape}")
        e2_down = self.down(e2)
        # print(f"e2_down shape: {e2_down.shape}")
        
        e3 = self.enc3(e2_down)
        # print(f"e3 shape: {e3.shape}")
        e3_down = self.down(e3)
        # print(f"e3_down shape: {e3_down.shape}")
        
        # Bottleneck
        b = self.bottleneck(e3_down)
        # print(f"b shape: {b.shape}")
        
        # Decoder
        b_up = self.up(b)
        # Xử lý trường hợp kích thước không khớp
        if b_up.shape[-1] != e3.shape[-1]:
            b_up = F.pad(b_up, (0, e3.shape[-1] - b_up.shape[-1]))
        d3 = self.dec3(torch.cat([b_up, e3], dim=1))
        
        d3_up = self.up(d3)
        # Xử lý trường hợp kích thước không khớp
        if d3_up.shape[-1] != e2.shape[-1]:
            d3_up = F.pad(d3_up, (0, e2.shape[-1] - d3_up.shape[-1]))
        d2 = self.dec2(torch.cat([d3_up, e2], dim=1))
        
        d2_up = self.up(d2)
        # Xử lý trường hợp kích thước không khớp
        if d2_up.shape[-1] != e1.shape[-1]:
            d2_up = F.pad(d2_up, (0, e1.shape[-1] - d2_up.shape[-1]))
        d1 = self.dec1(torch.cat([d2_up, e1], dim=1))
        
        # Đầu ra
        output = self.final(d1)
        
        return output

# Lớp Diffusion Model đã sửa
class FixedDiffusionModel:
    def __init__(self, model, beta_start=1e-4, beta_end=0.02, num_diffusion_steps=100):
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
        
        # Kết hợp nhiễu và mặt nạ làm đầu vào
        model_input = torch.cat([x_t, mask], dim=1)
        
        # Dự đoán nhiễu
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
        # Đặt mô hình ở chế độ đánh giá để tránh BatchNorm lỗi với batch_size = 1
        self.model.eval()
        
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
        
        # Đặt lại mô hình về chế độ huấn luyện sau khi lấy mẫu
        self.model.train()
            
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

# Kiểm tra và sửa lỗi mô hình với dữ liệu giả
def test_model():
    """Kiểm tra mô hình với dữ liệu giả"""
    print("Kiểm tra mô hình với dữ liệu giả...")
    
    # Tạo dataset giả với số chiều lớn hơn để tránh kích thước bị giảm xuống 1
    # Tạo dữ liệu giả có 16 điểm thay vì 11 để sau khi pooling 3 lần vẫn còn > 1
    class ModifiedDataset(Dataset):
        def __init__(self, missing_ratio=0.3, seq_length=16):
            self.missing_ratio = missing_ratio
            # 20 mẫu, mỗi mẫu có seq_length điểm (ví dụ 16 điểm)
            self.data = np.random.rand(20, seq_length) * 0.2 + 0.1
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            iv_surface = self.data[idx].astype(np.float32)
            mask = np.random.choice([0, 1], size=iv_surface.shape, p=[self.missing_ratio, 1-self.missing_ratio])
            masked_iv = iv_surface * mask
            
            # Chuyển sang tensor
            iv_surface = torch.tensor(iv_surface, dtype=torch.float32)
            masked_iv = torch.tensor(masked_iv, dtype=torch.float32)
            mask = torch.tensor(mask, dtype=torch.float32)
            
            return {'original': iv_surface, 'masked': masked_iv, 'mask': mask}
    
    # Sử dụng dataset có kích thước lớn hơn
    dataset = ModifiedDataset(missing_ratio=0.3, seq_length=16)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print(f"Kích thước mẫu dữ liệu: {dataset[0]['original'].shape}")
    
    # Khởi tạo mô hình
    model = FixedDiffusionUNet(in_channels=2, out_channels=1).to(device)
    
    # Khởi tạo diffusion model
    diffusion = FixedDiffusionModel(model, num_diffusion_steps=10)  # Giảm số bước để test nhanh hơn
    
    # Khởi tạo optimizer
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    # Huấn luyện
    print("Bắt đầu huấn luyện mô hình...")
    losses = diffusion.fit(dataloader, epochs=5, optimizer=optimizer)
    
    # Kiểm tra mô hình với một mẫu
    sample = dataset[0]
    original = sample['original'].unsqueeze(0).unsqueeze(1).to(device)
    masked = sample['masked'].unsqueeze(0).unsqueeze(1).to(device)
    mask = sample['mask'].unsqueeze(0).unsqueeze(1).to(device)
    
    # Phục hồi bề mặt IV
    print("Tạo dự đoán...")
    reconstructed = diffusion.sample(masked, mask)
    
    print("Hoàn thành kiểm tra!")
    print(f"Kích thước đầu ra: {reconstructed.shape}")
    
    return diffusion

if __name__ == "__main__":
    test_model() 