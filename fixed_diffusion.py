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
from tqdm import tqdm

# Từ fix_model.py
from fix_model import FixedDiffusionUNet

# Thiết lập seed cho quá trình tái tạo
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Kiểm tra thiết bị có sẵn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lớp Diffusion Model đã sửa đổi
class EnhancedDiffusionModel:
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
        Mẫu từ phân phối p(x_{t-1} | x_t)
        """
        self.model.eval()  # Đảm bảo mô hình ở chế độ đánh giá
        
        # Chuẩn bị đầu vào cho mô hình
        model_input = torch.cat([x_t, mask], dim=1)
        
        # Dự đoán nhiễu
        noise_pred = self.model(model_input, t/self.num_diffusion_steps, mask)
        
        # Các hệ số cần thiết
        betas_t = self.betas[t].reshape(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t]).reshape(-1, 1, 1)
        
        # Tính toán trung bình của phân phối posterior
        mean = sqrt_recip_alphas_t * (x_t - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)
        
        # Chỉ thêm nhiễu nếu t > 0
        noise = torch.zeros_like(x_t)
        if t > 0:
            variance_t = self.posterior_variance[t].reshape(-1, 1, 1)
            noise = torch.randn_like(x_t) * torch.sqrt(variance_t)
        
        return mean + noise
    
    @torch.no_grad()
    def sample(self, masked_iv, mask, num_samples=1):
        """
        Sinh bề mặt IV từ dữ liệu bị thiếu
        
        Tham số:
            masked_iv: Bề mặt IV bị thiếu
            mask: Mặt nạ cho biết vị trí dữ liệu bị thiếu
            num_samples: Số lượng mẫu cần sinh
        """
        self.model.eval()  # Đảm bảo mô hình ở chế độ đánh giá
        
        # Thêm batch dimension nếu cần thiết
        if len(masked_iv.shape) == 2:
            masked_iv = masked_iv.unsqueeze(0).unsqueeze(0)
        elif len(masked_iv.shape) == 3:
            masked_iv = masked_iv.unsqueeze(0)
        
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif len(mask.shape) == 3:
            mask = mask.unsqueeze(0)
        
        batch_size = masked_iv.shape[0]
        
        # Khởi tạo nhiễu Gaussian
        x_t = torch.randn_like(masked_iv)
        
        # Dữ liệu đã biết không cần khôi phục
        x_t = x_t * (1 - mask) + masked_iv * mask
        
        # Lấy mẫu từ quá trình khử nhiễu
        for t in range(self.num_diffusion_steps - 1, -1, -1):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x_t = self.p_sample(x_t, t_batch, mask)
            
            # Đặt lại các giá trị đã biết
            x_t = x_t * (1 - mask) + masked_iv * mask
        
        return x_t
    
    def fit(self, dataloader, epochs, optimizer, scheduler=None, save_path="models", start_epoch=0):
        """
        Huấn luyện mô hình
        
        Tham số:
            dataloader: DataLoader chứa dữ liệu huấn luyện
            epochs: Số epoch huấn luyện
            optimizer: Optimizer để cập nhật trọng số
            scheduler: Learning rate scheduler (tuỳ chọn)
            save_path: Đường dẫn để lưu mô hình
            start_epoch: Epoch bắt đầu (dùng khi tiếp tục huấn luyện)
        """
        # Tạo thư mục lưu trữ mô hình
        os.makedirs(save_path, exist_ok=True)
        
        # Đường dẫn đến file checkpoint
        checkpoint_path = os.path.join(save_path, "diffusion_model_checkpoint.pth")
        best_model_path = os.path.join(save_path, "diffusion_model_best.pth")
        final_model_path = os.path.join(save_path, "diffusion_model_final.pth")
        
        # Theo dõi loss tốt nhất
        best_loss = float('inf')
        
        # Huấn luyện
        for epoch in range(start_epoch, start_epoch + epochs):
            # Theo dõi loss
            epoch_loss = 0.0
            num_batches = 0
            
            # Duyệt qua dữ liệu
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{start_epoch+epochs}")
            for batch in progress_bar:
                # Lấy dữ liệu từ batch
                original = batch['original'].to(device)
                mask = batch['mask'].to(device)
                
                # Tính loss
                optimizer.zero_grad()
                loss = self.p_losses(original, mask)
                loss.backward()
                optimizer.step()
                
                # Cập nhật learning rate nếu có scheduler
                if scheduler is not None:
                    scheduler.step()
                
                # Cập nhật loss
                epoch_loss += loss.item()
                num_batches += 1
                
                # Hiển thị loss
                progress_bar.set_postfix(loss=loss.item())
            
            # Tính loss trung bình
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1}/{start_epoch+epochs}, Loss: {avg_loss:.6f}")
            
            # Lưu checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Lưu mô hình tốt nhất
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(checkpoint, best_model_path)
                print(f"Đã lưu mô hình tốt nhất với loss: {best_loss:.6f}")
        
        # Lưu mô hình cuối cùng
        torch.save(checkpoint, final_model_path)
        print(f"Đã lưu mô hình cuối cùng sau {start_epoch+epochs} epochs")

# Sử dụng lớp Diffusion Model đã sửa đổi
def test_enhanced_model():
    """
    Kiểm tra mô hình Diffusion đã sửa đổi
    """
    # Tạo mô hình
    model = FixedDiffusionUNet(in_channels=2, out_channels=1).to(device)
    diffusion = EnhancedDiffusionModel(model)
    
    # Tạo dữ liệu giả
    data = torch.randn(4, 1, 16).to(device)  # [batch_size, channels, sequence_length]
    mask = torch.ones(4, 1, 16).to(device)   # [batch_size, 1, sequence_length]
    
    # Mask một số vị trí
    mask[:, :, 5:10] = 0
    
    # Tính loss
    loss = diffusion.p_losses(data, mask)
    print(f"Loss: {loss.item()}")
    
    # Sinh mẫu
    sample = diffusion.sample(data * mask, mask)
    print(f"Sample shape: {sample.shape}")
    
    return diffusion

if __name__ == "__main__":
    test_enhanced_model() 