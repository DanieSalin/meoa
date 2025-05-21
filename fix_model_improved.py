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

# Layer cho positional embedding dạng sin/cos
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        """
        Time embedding dạng Sinusoidal cho Diffusion Model
        
        Tham số:
            dim (int): Kích thước embedding
        """
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Tính toán time embedding
        
        Tham số:
            time (Tensor): Bước thời gian, shape [B]
            
        Kết quả:
            Tensor: Time embedding, shape [B, dim]
        """
        # Đảm bảo time là 1D tensor
        if len(time.shape) > 1:
            time = time.squeeze(-1)
            
        # Đảm bảo kiểu dữ liệu phù hợp
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        
        # Nhân time với embeddings và tính sin/cos
        embeddings = time.unsqueeze(1) * embeddings.unsqueeze(0)
        embeddings = torch.cat((torch.sin(embeddings), torch.cos(embeddings)), dim=-1)
        
        # Nếu kích thước là lẻ, thêm 1 chiều
        if self.dim % 2 == 1:
            embeddings = torch.nn.functional.pad(embeddings, (0, 1, 0, 0))
            
        return embeddings

# Mô hình U-Net cải tiến cho Diffusion model 
# Thay thế InstanceNorm bằng LayerNorm ở bottleneck và các lớp decoder cần thiết.
class ImprovedDiffusionUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, time_dim=128, skip_last_downsample=False):
        """
        Mô hình UNet cải tiến cho Diffusion
        
        Tham số:
            in_channels: Số kênh đầu vào (2 = dữ liệu + mặt nạ)
            out_channels: Số kênh đầu ra (1 = dữ liệu phục hồi)
            time_dim: Kích thước embedding của thời gian
            skip_last_downsample: Bỏ qua lớp downsample cuối cùng nếu dữ liệu quá nhỏ
        """
        super().__init__()
        self.skip_last_downsample = skip_last_downsample
        
        # Lớp đầu vào
        self.inc = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            # Sử dụng GroupNorm thay cho InstanceNorm1d
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.GELU()
        )
        
        # Lớp Encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            # Sử dụng GroupNorm thay cho InstanceNorm1d
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.GELU()
        )
        self.down1 = nn.MaxPool1d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            # Sử dụng GroupNorm thay cho InstanceNorm1d
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.GELU()
        )
        self.down2 = nn.MaxPool1d(2)
        
        # Bottleneck với GroupNorm
        self.bottleneck = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            # GroupNorm với num_groups=1 hoạt động như LayerNorm cho tensor nhỏ
            nn.GroupNorm(num_groups=1, num_channels=256),
            nn.GELU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=256),
            nn.GELU()
        )
        
        # Lớp Decoder với Upsampling
        self.up1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.dec1 = nn.Sequential(
            nn.Conv1d(256 + 128, 128, kernel_size=3, padding=1),
            # Sử dụng GroupNorm thay cho InstanceNorm1d
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.GELU()
        )
        
        self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.dec2 = nn.Sequential(
            nn.Conv1d(128 + 64, 64, kernel_size=3, padding=1),
            # Sử dụng GroupNorm thay cho InstanceNorm1d
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.GELU()
        )
        
        # Lớp đầu ra
        self.outc = nn.Sequential(
            nn.Conv1d(64 + 32, 32, kernel_size=3, padding=1),
            # Sử dụng GroupNorm thay cho InstanceNorm1d
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.GELU(),
            nn.Conv1d(32, out_channels, kernel_size=1)
        )
        
        # Sinusoidal time embedding
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim)
        )
        
        # Lớp projection cho time embedding
        self.time_inc = nn.Linear(time_dim, 32)
        self.time_enc1 = nn.Linear(time_dim, 64)
        self.time_enc2 = nn.Linear(time_dim, 128)
        self.time_bottleneck = nn.Linear(time_dim, 256)
        self.time_dec1 = nn.Linear(time_dim, 128)
        self.time_dec2 = nn.Linear(time_dim, 64)
    
    def forward(self, x, t, mask=None):
        """
        x: Tensor [B, C, L] hoặc [B, C, H, L] - Thông tin đầu vào (nhiễu và mặt nạ)
        t: Tensor [B, 1] - Bước thời gian (time step)
        mask: Tensor [B, 1, L] hoặc [B, 1, H, L] - Mặt nạ cho biết giá trị nào bị thiếu
        """
        # Kiểm tra và xử lý nếu x có 4 chiều (batch, channel, height, length)
        has_4_dim = len(x.shape) == 4
        original_shape = x.shape
        
        if has_4_dim:
            print(f"Forward - Phát hiện đầu vào 4 chiều: {x.shape}")
            batch, channels, height, length = x.shape
            # Reshape để xử lý như tensor 3D
            x = x.reshape(batch, channels, -1)
            if mask is not None:
                mask = mask.reshape(batch, 1, -1)
        
        # Chuyển t sang kiểu float cho time embedding
        t = t.float()
        
        # Nếu mask không được cung cấp, tạo mask toàn 1 (không có giá trị nào bị thiếu)
        if mask is None:
            mask = torch.ones_like(x[:, :1, :])
        
        # Tạo embedding cho thời gian và reshape để broadcast
        t_emb = self.time_mlp(t)  # [B, time_dim]
        
        # Forward path với skip connections và time embeddings
        # Input block
        e0 = self.inc(x)
        e0 = e0 + self.time_inc(t_emb).unsqueeze(-1)
        
        # Encoder 1
        e1 = self.enc1(e0)
        e1 = e1 + self.time_enc1(t_emb).unsqueeze(-1)
        
        # Downsample 1
        e1_down = self.down1(e1)
        
        # Encoder 2
        e2 = self.enc2(e1_down)
        e2 = e2 + self.time_enc2(t_emb).unsqueeze(-1)
        
        # Kiểm tra kích thước spatial và quyết định có downsampling tiếp hay không
        if self.skip_last_downsample or e2.shape[2] <= 2:
            print(f"Bỏ qua downsampling cuối do kích thước không gian nhỏ: {e2.shape}")
            # Bottleneck không downsampling
            bottle = self.bottleneck(e2)
        else:
            # Downsample 2
            e2_down = self.down2(e2)
            
            # Bottleneck
            bottle = self.bottleneck(e2_down)
        
        bottle = bottle + self.time_bottleneck(t_emb).unsqueeze(-1)
        
        # Decoder 1 (Upsampling)
        d1_up = self.up1(bottle)
        
        # Đảm bảo kích thước đúng cho skip connection
        if d1_up.shape[2] != e2.shape[2]:
            d1_up = F.interpolate(d1_up, size=e2.shape[2], mode='linear', align_corners=True)
        
        # Nối với skip connection
        d1_cat = torch.cat([d1_up, e2], dim=1)
        d1 = self.dec1(d1_cat)
        d1 = d1 + self.time_dec1(t_emb).unsqueeze(-1)
        
        # Decoder 2 (Upsampling)
        d2_up = self.up2(d1)
        
        # Đảm bảo kích thước đúng cho skip connection
        if d2_up.shape[2] != e1.shape[2]:
            d2_up = F.interpolate(d2_up, size=e1.shape[2], mode='linear', align_corners=True)
        
        # Nối với skip connection
        d2_cat = torch.cat([d2_up, e1], dim=1)
        d2 = self.dec2(d2_cat)
        d2 = d2 + self.time_dec2(t_emb).unsqueeze(-1)
        
        # Output
        out_cat = torch.cat([d2, e0], dim=1)
        out = self.outc(out_cat)
        
        # Nếu đầu vào là tensor 4 chiều, reshape đầu ra về 4 chiều
        if has_4_dim:
            out = out.reshape(batch, out.shape[1], height, length)
        
        return out

# Lớp Diffusion Model đã sửa
class ImprovedDiffusionModel:
    def __init__(self, model, beta_start=1e-4, beta_end=0.02, num_diffusion_steps=150):
        """
        Diffusion Model cho phục hồi bề mặt Implied Volatility
                
        Tham số:
            model: Mô hình UNet
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
        
        # Thiết bị đang sử dụng
        self.device = device
        
        # Hệ số cho Gaussian smoothing trong quá trình lấy mẫu
        self.smoothing_sigma = 0.01
    
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
        
        # In các shape để debug
        print(f"p_losses - Đầu vào: x_0.shape={x_0.shape}, mask.shape={mask.shape}")
        
        # Đảm bảo mask có cùng shape với x_0
        if mask.shape != x_0.shape:
            print(f"p_losses - Điều chỉnh mask.shape={mask.shape} cho phù hợp với x_0.shape={x_0.shape}")
            if len(mask.shape) < len(x_0.shape):
                # Thêm chiều cho mask
                mask = mask.unsqueeze(1)
            elif mask.shape[1] != x_0.shape[1]:
                # Nếu số kênh khác nhau, broadcast hoặc lặp lại kênh
                if mask.shape[1] == 1:
                    # Broadcast từ 1 kênh sang multiple kênh
                    mask = mask.repeat(1, x_0.shape[1], 1)
        
        # Chọn bước thời gian ngẫu nhiên
        t = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=device)
        
        if noise is None:
            noise = torch.randn_like(x_0)
            
        # Thêm nhiễu
        x_t = self.q_sample(x_0, t, noise)
        
        # Kết hợp nhiễu và mặt nạ làm đầu vào
        model_input = torch.cat([x_t, mask], dim=1)
        
        # Dự đoán nhiễu
        self.model.train()  # Đảm bảo mô hình ở chế độ huấn luyện
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
        Lấy một mẫu từ p(x_{t-1}|x_t) sử dụng mô hình đã huấn luyện
        
        Tham số:
            x_t: Mẫu tại bước thời gian t
            t: Bước thời gian hiện tại
            mask: Mặt nạ cho biết giá trị nào bị thiếu
            
        Kết quả:
            Mẫu mới tại bước thời gian t-1
        """
        # Lưu bản sao của tensor t dạng long để sử dụng làm chỉ số
        t_index = t.clone()
        
        # Chuyển đổi t sang float trước khi sử dụng cho time embedding
        t_float = t.float()
        
        # Lấy hệ số cho bước thời gian hiện tại sử dụng t_index dạng long
        betas_t = self.betas[t_index]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_index]
        sqrt_recip_alphas_t = torch.rsqrt(self.alphas[t_index])
        
        # Reshape hệ số dựa trên kích thước của x_t
        if len(x_t.shape) == 4:  # [B, C, H, W]
            betas_t = betas_t.reshape(-1, 1, 1, 1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.reshape(-1, 1, 1, 1)
            sqrt_recip_alphas_t = sqrt_recip_alphas_t.reshape(-1, 1, 1, 1)
        else:  # [B, C, L]
            betas_t = betas_t.reshape(-1, 1, 1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.reshape(-1, 1, 1)
            sqrt_recip_alphas_t = sqrt_recip_alphas_t.reshape(-1, 1, 1)
        
        # Đầu vào cho mô hình: x_t và mặt nạ
        model_input = torch.cat([x_t, mask], dim=1)
        
        # Dự đoán nhiễu - sử dụng t_float cho time embedding
        pred_noise = self.model(model_input, t_float)
        
        # Tính toán trung bình của phân phối
        mean = sqrt_recip_alphas_t * (x_t - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t)
        
        # Gaussian smoothing để giảm nhiễu nếu cần
        if t_index[0] < 20 and self.smoothing_sigma > 0:
            kernel_size = 3
            padding = kernel_size // 2
            if len(x_t.shape) == 4:  # [B, C, H, W]
                smoothing = nn.Conv2d(
                    x_t.shape[1], x_t.shape[1], kernel_size=kernel_size, padding=padding, 
                    groups=x_t.shape[1], bias=False
                ).to(self.device)
                # Tạo Gaussian kernel
                kernel = torch.ones((kernel_size, kernel_size)) * (1.0 / (kernel_size * kernel_size))
                for i in range(x_t.shape[1]):
                    smoothing.weight.data[i, 0] = kernel
                # Áp dụng smoothing
                mean = smoothing(mean)
            else:  # [B, C, L]
                smoothing = nn.Conv1d(
                    x_t.shape[1], x_t.shape[1], kernel_size=kernel_size, padding=padding, 
                    groups=x_t.shape[1], bias=False
                ).to(self.device)
                # Tạo Gaussian kernel
                kernel = torch.ones(kernel_size) * (1.0 / kernel_size)
                for i in range(x_t.shape[1]):
                    smoothing.weight.data[i, 0] = kernel
                # Áp dụng smoothing
                mean = smoothing(mean)
        
        # Không thêm nhiễu nếu t=0
        if t_index[0] > 0:
            # Phương sai
            if len(x_t.shape) == 4:
                variance = torch.sqrt(self.posterior_variance[t_index].reshape(-1, 1, 1, 1))
                noise = torch.randn_like(x_t)
            else:
                variance = torch.sqrt(self.posterior_variance[t_index].reshape(-1, 1, 1))
                noise = torch.randn_like(x_t)
            
            # Thêm nhiễu
            x_t_prev = mean + variance * noise
        else:
            x_t_prev = mean
        
        # Giữ nguyên các giá trị đã biết từ mặt nạ
        known_values = (mask == 1.0).float()
        if len(x_t.shape) == 4:
            original_values = model_input[:, 0:1]  # Lấy kênh đầu tiên của model_input
        else:
            original_values = model_input[:, 0:1]  # Lấy kênh đầu tiên của model_input
        
        # Kết hợp giá trị đã biết và giá trị dự đoán
        x_t_prev = (1 - known_values) * x_t_prev + known_values * original_values
        
        return x_t_prev

    @torch.no_grad()
    def sample(self, masked_iv, mask, num_samples=1):
        """
        Lấy mẫu từ Diffusion Model
        
        Tham số:
            masked_iv: Dữ liệu IV bị thiếu
            mask: Mặt nạ cho biết giá trị nào bị thiếu
            num_samples: Số lượng mẫu cần lấy
            
        Kết quả:
            Các mẫu được lấy
        """
        # Lưu kích thước ban đầu
        original_shape = masked_iv.shape
        
        # Kiểm tra nếu đầu vào là 4D hoặc 3D
        has_4_dim = len(original_shape) == 4
        
        # Chuẩn bị đầu vào
        if not has_4_dim and len(original_shape) == 3:
            batch_size, channels, length = original_shape
        elif has_4_dim:
            batch_size, channels, height, width = original_shape
        else:
            raise ValueError(f"Kích thước đầu vào không hợp lệ: {original_shape}")
        
        samples = []
        
        for _ in range(num_samples):
            # Bắt đầu từ nhiễu Gaussian
            if has_4_dim:
                x_t = torch.randn((batch_size, 1, height, width), device=self.device)
            else:
                x_t = torch.randn((batch_size, 1, length), device=self.device)
            
            # Lấy mẫu dần dần từ x_T về x_0
            for t in reversed(range(self.num_diffusion_steps)):
                t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
                # Chuyển đổi t sang float tại điểm gọi p_sample
                x_t = self.p_sample(x_t, t_tensor, mask)
            
            # Thêm vào danh sách mẫu
            samples.append(x_t)
        
        # Tính trung bình của các mẫu (ensemble)
        if num_samples > 1:
            print(f"Tính trung bình của {num_samples} mẫu")
            samples = torch.cat(samples, dim=0)
            samples = samples.reshape(num_samples, batch_size, *samples.shape[1:])
            samples = samples.mean(dim=0)
        else:
            samples = samples[0]
        
        # Kết hợp giữa giá trị đã biết và giá trị dự đoán
        known_values = (mask == 1.0).float()
        result = (1 - known_values) * samples + known_values * masked_iv
        
        # Áp dụng Gaussian smoothing để làm mịn kết quả cuối cùng
        if has_4_dim:  # [B, C, H, W]
            kernel_size = 3
            padding = kernel_size // 2
            smoothing = nn.Conv2d(
                result.shape[1], result.shape[1], kernel_size=kernel_size, padding=padding, 
                groups=result.shape[1], bias=False
            ).to(self.device)
            # Tạo Gaussian kernel
            kernel = torch.ones((kernel_size, kernel_size)) * (1.0 / (kernel_size * kernel_size))
            for i in range(result.shape[1]):
                smoothing.weight.data[i, 0] = kernel
            # Áp dụng smoothing
            result = smoothing(result)
        else:  # [B, C, L]
            kernel_size = 3
            padding = kernel_size // 2
            smoothing = nn.Conv1d(
                result.shape[1], result.shape[1], kernel_size=kernel_size, padding=padding, 
                groups=result.shape[1], bias=False
            ).to(self.device)
            # Tạo Gaussian kernel
            kernel = torch.ones(kernel_size) * (1.0 / kernel_size)
            for i in range(result.shape[1]):
                smoothing.weight.data[i, 0] = kernel
            # Áp dụng smoothing
            result = smoothing(result)
        
        return result

# Test hàm
def test_model():
    """
    Test mô hình với dữ liệu nhỏ để kiểm tra lỗi
    """
    # Tạo dữ liệu nhỏ
    class ModifiedDataset(Dataset):
        def __init__(self, missing_ratio=0.3, seq_length=16):
            self.missing_ratio = missing_ratio
            self.seq_length = seq_length
            self.data = np.random.rand(10, seq_length) * 0.2 + 0.1
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            iv = self.data[idx].astype(np.float32)
            mask = np.random.choice([0, 1], size=iv.shape, p=[self.missing_ratio, 1-self.missing_ratio])
            masked_iv = iv * mask
            
            # Không thêm chiều kênh tại đây, để xử lý thống nhất trong hàm sample
            iv = torch.tensor(iv, dtype=torch.float32)
            masked_iv = torch.tensor(masked_iv, dtype=torch.float32)
            mask = torch.tensor(mask, dtype=torch.float32)
            
            return {'original': iv, 'masked': masked_iv, 'mask': mask}
    
    # Kiểm tra với dữ liệu nhỏ (seq_length = 11 như trong dữ liệu thực)
    dataset = ModifiedDataset(seq_length=11)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Tạo mô hình và kiểm tra forward pass
    model = ImprovedDiffusionUNet(in_channels=2, out_channels=1).to(device)
    diffusion = ImprovedDiffusionModel(model)
    
    # Lấy batch đầu tiên
    batch = next(iter(dataloader))
    
    # Chuẩn bị tensor đầu vào cho forward pass - thêm channel dimension
    original = batch['original'].unsqueeze(1).to(device)
    mask = batch['mask'].unsqueeze(1).to(device)
    
    try:
        # Thử forward pass
        loss = diffusion.p_losses(original, mask)
        print(f"Forward pass thành công với loss: {loss.item()}")
        
        # Chuẩn bị dữ liệu cho inference
        # Lấy một mẫu để test riêng
        masked_sample = batch['masked'][0].to(device)  # Lấy mẫu đầu tiên trong batch
        mask_sample = batch['mask'][0].to(device)
        
        print(f"Debug - Trước khi gọi sample: masked_sample.shape: {masked_sample.shape}, mask_sample.shape: {mask_sample.shape}")
        
        # Thêm channel dimension nếu cần
        if len(masked_sample.shape) == 1:
            masked_sample = masked_sample.unsqueeze(0)  # [L] -> [1, L]
        if len(mask_sample.shape) == 1:
            mask_sample = mask_sample.unsqueeze(0)      # [L] -> [1, L]
        
        # Thử inference với một mẫu duy nhất
        result = diffusion.sample(masked_sample, mask_sample)
        print(f"Inference thành công, shape kết quả: {result.shape}")
        
        print("Mô hình hoạt động tốt!")
        return True
    except Exception as e:
        print(f"Lỗi: {e}")
        import traceback
        traceback.print_exc()  # In chi tiết lỗi
        return False

if __name__ == "__main__":
    test_model() 