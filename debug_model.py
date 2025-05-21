#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# Thiết lập seed cho quá trình tái tạo
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Kiểm tra thiết bị có sẵn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {device}")

# Dataset giả
class DummyIVSurfaceDataset(Dataset):
    def __init__(self, missing_ratio=0.2):
        self.missing_ratio = missing_ratio
        # Tạo dữ liệu giả - 20 mẫu, mỗi mẫu có 11 điểm (tương ứng với số strike price)
        self.data = np.random.rand(20, 11)
        
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

# Mô hình U-Net đơn giản hoá cho Diffusion model
class SimpleDiffusionUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, time_emb_dim=8):
        super(SimpleDiffusionUNet, self).__init__()
        
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
        
        # Đơn giản hoá mô hình
        self.encoder = nn.Sequential(
            nn.Conv1d(total_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, out_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x, t, mask=None):
        """
        x: Tensor [B, C, L] - Thông tin đầu vào (nhiễu và mặt nạ)
        t: Tensor [B, 1] - Bước thời gian (time step)
        """
        # Time embedding
        t_emb = self.time_mlp(t.unsqueeze(-1))
        
        # Chuyển shape để phù hợp với x
        t_emb = t_emb.unsqueeze(-1).repeat(1, 1, x.shape[-1])
        
        # Debug: In kích thước để kiểm tra
        print(f"Input x shape: {x.shape}, t_emb shape: {t_emb.shape}")
        
        # Kết hợp t_emb vào x
        x = torch.cat([x, t_emb], dim=1)
        print(f"After concatenation, x shape: {x.shape}")
        
        # Encoder
        x = self.encoder(x)
        print(f"After encoder, x shape: {x.shape}")
        
        # Decoder
        x = self.decoder(x)
        print(f"After decoder, x shape: {x.shape}")
        
        return x

def main():
    """Hàm chính để debug mô hình"""
    print("Debug mô hình U-Net cho Diffusion Model...")
    
    # Tạo dataset giả
    dataset = DummyIVSurfaceDataset(missing_ratio=0.3)
    print(f"Số mẫu: {len(dataset)}")
    
    # Lấy một mẫu để kiểm tra
    sample = dataset[0]
    print(f"Original shape: {sample['original'].shape}")
    print(f"Masked shape: {sample['masked'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    
    # Tạo tensor batch
    x_0 = sample['original'].unsqueeze(0).unsqueeze(1)  # [B, 1, L]
    mask = sample['mask'].unsqueeze(0).unsqueeze(1)     # [B, 1, L]
    
    print("\nShape sau khi chuyển thành batch:")
    print(f"x_0 shape: {x_0.shape}")
    print(f"mask shape: {mask.shape}")
    
    # Tạo model input
    model_input = torch.cat([x_0, mask], dim=1)  # Kết hợp thành 2 kênh
    print(f"model_input shape: {model_input.shape}")
    
    # Tạo mô hình
    model = SimpleDiffusionUNet(in_channels=2, out_channels=1, time_emb_dim=8).to(device)
    
    # In thông tin mô hình
    print("\nCấu trúc mô hình:")
    print(model)
    
    # Đưa dữ liệu vào mô hình
    t = torch.tensor([0.5], device=device)  # Giả sử t = 0.5
    
    # Truyền dữ liệu qua mô hình
    print("\nBắt đầu forward pass...")
    output = model(model_input.to(device), t)
    
    print(f"\nOutput shape: {output.shape}")
    print("Debug kết thúc!")

if __name__ == "__main__":
    main() 