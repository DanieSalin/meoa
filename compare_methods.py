#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, RectBivariateSpline
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
from torch.utils.data import Dataset

# Từ create_diffusion_model.py
from create_diffusion_model import (
    IVSurfaceDataset, 
    DiffusionUNet, 
    DiffusionModel, 
    generate_iv_surface, 
    evaluate_reconstruction,
    plot_iv_surface,
    device
)

# Thiết lập seed cho quá trình tái tạo
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Phương pháp nội suy cubic spline
def cubic_spline_interpolation(iv_masked, mask, grid_shape=None):
    """
    Nội suy bề mặt IV sử dụng cubic spline
    
    Tham số:
        iv_masked: Bề mặt IV bị thiếu
        mask: Mặt nạ (1 = có dữ liệu, 0 = thiếu)
        grid_shape: Hình dạng lưới của bề mặt IV
    
    Trả về:
        iv_interpolated: Bề mặt IV sau khi nội suy
    """
    # Chuyển sang numpy nếu là tensor
    if torch.is_tensor(iv_masked):
        iv_masked = iv_masked.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    
    if len(iv_masked.shape) == 3:
        iv_masked = iv_masked.squeeze(1)
    if len(mask.shape) == 3:
        mask = mask.squeeze(1)
    
    # Tạo lưới strikes và maturities
    if grid_shape is None:
        n_rows, n_cols = iv_masked.shape
    else:
        n_rows, n_cols = grid_shape
    
    strikes = np.arange(n_cols)
    maturities = np.arange(n_rows)
    
    # Nội suy với mỗi hàng (cùng một maturity)
    iv_interpolated = np.zeros_like(iv_masked)
    
    # Nội suy theo hàng (theo maturity)
    for i in range(n_rows):
        # Lấy chỉ số có dữ liệu
        valid_indices = np.where(mask[i, :] > 0)[0]
        
        if len(valid_indices) < 4:  # Cần ít nhất 4 điểm cho cubic spline
            # Sao chép trực tiếp nếu không đủ dữ liệu
            iv_interpolated[i, :] = iv_masked[i, :]
            continue
        
        # Tạo cubic spline
        cs = CubicSpline(strikes[valid_indices], iv_masked[i, valid_indices])
        
        # Nội suy cho tất cả các điểm
        iv_interpolated[i, :] = cs(strikes)
    
    # Nội suy theo cột (theo strike)
    for j in range(n_cols):
        # Lấy chỉ số có dữ liệu
        valid_indices = np.where(mask[:, j] > 0)[0]
        
        if len(valid_indices) < 4:  # Cần ít nhất 4 điểm cho cubic spline
            continue
        
        # Tạo cubic spline
        cs = CubicSpline(maturities[valid_indices], iv_masked[valid_indices, j])
        
        # Nội suy cho các điểm còn thiếu
        missing_indices = np.where(mask[:, j] == 0)[0]
        iv_interpolated[missing_indices, j] = cs(maturities[missing_indices])
    
    # Kết hợp dữ liệu gốc và dữ liệu nội suy
    result = np.where(mask > 0, iv_masked, iv_interpolated)
    
    return result

# Phương pháp nội suy 2D với RectBivariateSpline
def bivariate_spline_interpolation(iv_masked, mask, grid_shape=None):
    """
    Nội suy bề mặt IV sử dụng RectBivariateSpline
    
    Tham số:
        iv_masked: Bề mặt IV bị thiếu
        mask: Mặt nạ (1 = có dữ liệu, 0 = thiếu)
        grid_shape: Hình dạng lưới của bề mặt IV
    
    Trả về:
        iv_interpolated: Bề mặt IV sau khi nội suy
    """
    # Chuyển sang numpy nếu là tensor
    if torch.is_tensor(iv_masked):
        iv_masked = iv_masked.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    
    if len(iv_masked.shape) == 3:
        iv_masked = iv_masked.squeeze(1)
    if len(mask.shape) == 3:
        mask = mask.squeeze(1)
    
    # Tạo lưới strikes và maturities
    if grid_shape is None:
        n_rows, n_cols = iv_masked.shape
    else:
        n_rows, n_cols = grid_shape
    
    strikes = np.arange(n_cols)
    maturities = np.arange(n_rows)
    
    # Lấy chỉ số có dữ liệu
    valid_indices = np.where(mask > 0)
    valid_rows, valid_cols = valid_indices
    
    # Không thể nội suy nếu quá ít điểm
    if len(valid_rows) < 4 or len(np.unique(valid_rows)) < 4 or len(np.unique(valid_cols)) < 4:
        return np.where(mask > 0, iv_masked, 0)
    
    try:
        # Tạo RectBivariateSpline với các điểm có dữ liệu
        spline = RectBivariateSpline(
            np.unique(valid_rows), 
            np.unique(valid_cols), 
            iv_masked[np.ix_(np.unique(valid_rows), np.unique(valid_cols))],
            kx=3, ky=3
        )
        
        # Tạo lưới đầy đủ
        X, Y = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing='ij')
        
        # Nội suy cho tất cả các điểm
        iv_interpolated = spline(X.flatten(), Y.flatten(), grid=False).reshape(n_rows, n_cols)
        
        # Kết hợp dữ liệu gốc và dữ liệu nội suy
        result = np.where(mask > 0, iv_masked, iv_interpolated)
        
        return result
    except Exception as e:
        print(f"Lỗi khi dùng RectBivariateSpline: {e}")
        # Fallback về phương pháp cubic spline 1D nếu có lỗi
        return cubic_spline_interpolation(iv_masked, mask, grid_shape)

# Hàm so sánh các phương pháp khác nhau
def compare_methods(csv_files, missing_ratio=0.3, num_samples=5, diffusion_model_path=None):
    """
    So sánh các phương pháp phục hồi bề mặt IV khác nhau
    
    Tham số:
        csv_files: Danh sách các file CSV chứa dữ liệu
        missing_ratio: Tỷ lệ dữ liệu bị thiếu
        num_samples: Số lượng mẫu để so sánh
        diffusion_model_path: Đường dẫn đến mô hình Diffusion đã huấn luyện (nếu có)
    """
    # Tạo thư mục kết quả
    os.makedirs("comparison_results", exist_ok=True)
    
    # Tạo dataset
    dataset = IVSurfaceDataset(csv_files, missing_ratio=missing_ratio)
    
    # Kết quả đánh giá
    results = {
        'sample_id': [],
        'method': [],
        'mse': [],
        'rmse': [],
        'mae': [],
        'mape': [],
        'arbitrage_free': []
    }
    
    # Tạo hoặc tải mô hình Diffusion
    if diffusion_model_path is not None and os.path.exists(diffusion_model_path):
        print(f"Tải mô hình từ {diffusion_model_path}")
        checkpoint = torch.load(diffusion_model_path, map_location=device)
        
        model = DiffusionUNet(in_channels=2, out_channels=1).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        diffusion = DiffusionModel(model)
    else:
        print("Không tìm thấy mô hình Diffusion đã huấn luyện")
        diffusion = None
    
    # So sánh các phương pháp với các mẫu
    for i in range(min(num_samples, len(dataset))):
        print(f"\nSo sánh mẫu {i+1}/{num_samples}:")
        
        sample = dataset[i]
        original = sample['original'].unsqueeze(0)
        masked = sample['masked'].unsqueeze(0)
        mask = sample['mask'].unsqueeze(0)
        
        # 1. Phương pháp Cubic Spline 1D
        print("  Đang dùng Cubic Spline 1D...")
        cubic_spline_result = cubic_spline_interpolation(masked, mask)
        cubic_spline_metrics = evaluate_reconstruction(original.numpy(), cubic_spline_result)
        
        # Thêm kết quả vào bảng
        results['sample_id'].append(i)
        results['method'].append('Cubic Spline 1D')
        results['mse'].append(cubic_spline_metrics['MSE'])
        results['rmse'].append(cubic_spline_metrics['RMSE'])
        results['mae'].append(cubic_spline_metrics['MAE'])
        results['mape'].append(cubic_spline_metrics['MAPE'])
        results['arbitrage_free'].append(cubic_spline_metrics['Arbitrage-free'])
        
        # Vẽ kết quả
        plot_iv_surface(
            cubic_spline_result,
            f"Bề mặt IV phục hồi bằng Cubic Spline 1D - Mẫu {i+1}",
            f"comparison_results/cubic_spline_1d_sample_{i+1}.png"
        )
        
        # 2. Phương pháp Bivariate Spline 2D
        print("  Đang dùng Bivariate Spline 2D...")
        bivariate_spline_result = bivariate_spline_interpolation(masked, mask)
        bivariate_spline_metrics = evaluate_reconstruction(original.numpy(), bivariate_spline_result)
        
        # Thêm kết quả vào bảng
        results['sample_id'].append(i)
        results['method'].append('Bivariate Spline 2D')
        results['mse'].append(bivariate_spline_metrics['MSE'])
        results['rmse'].append(bivariate_spline_metrics['RMSE'])
        results['mae'].append(bivariate_spline_metrics['MAE'])
        results['mape'].append(bivariate_spline_metrics['MAPE'])
        results['arbitrage_free'].append(bivariate_spline_metrics['Arbitrage-free'])
        
        # Vẽ kết quả
        plot_iv_surface(
            bivariate_spline_result,
            f"Bề mặt IV phục hồi bằng Bivariate Spline 2D - Mẫu {i+1}",
            f"comparison_results/bivariate_spline_2d_sample_{i+1}.png"
        )
        
        # 3. Phương pháp Diffusion Model (nếu có)
        if diffusion is not None:
            print("  Đang dùng Diffusion Model...")
            diffusion_result = generate_iv_surface(diffusion, masked, mask)
            diffusion_metrics = evaluate_reconstruction(original, diffusion_result)
            
            # Thêm kết quả vào bảng
            results['sample_id'].append(i)
            results['method'].append('Diffusion Model')
            results['mse'].append(diffusion_metrics['MSE'])
            results['rmse'].append(diffusion_metrics['RMSE'])
            results['mae'].append(diffusion_metrics['MAE'])
            results['mape'].append(diffusion_metrics['MAPE'])
            results['arbitrage_free'].append(diffusion_metrics['Arbitrage-free'])
            
            # Vẽ kết quả
            plot_iv_surface(
                diffusion_result,
                f"Bề mặt IV phục hồi bằng Diffusion Model - Mẫu {i+1}",
                f"comparison_results/diffusion_model_sample_{i+1}.png"
            )
        
        # Vẽ bề mặt gốc và bị thiếu
        plot_iv_surface(
            original.squeeze(1),
            f"Bề mặt IV gốc - Mẫu {i+1}",
            f"comparison_results/original_sample_{i+1}.png"
        )
        
        plot_iv_surface(
            masked.squeeze(1),
            f"Bề mặt IV bị thiếu - Mẫu {i+1}",
            f"comparison_results/masked_sample_{i+1}.png"
        )
    
    # Tạo DataFrame từ kết quả
    results_df = pd.DataFrame(results)
    
    # Tính trung bình theo phương pháp
    avg_results = results_df.groupby('method').mean().reset_index()
    
    # Lưu kết quả vào file CSV
    results_df.to_csv("comparison_results/detailed_results.csv", index=False)
    avg_results.to_csv("comparison_results/average_results.csv", index=False)
    
    # Vẽ biểu đồ so sánh
    metrics = ['mse', 'rmse', 'mae', 'mape']
    methods = avg_results['method'].values
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        axes[i].bar(methods, avg_results[metric].values)
        axes[i].set_title(f'So sánh {metric.upper()} theo phương pháp')
        axes[i].set_ylabel(metric.upper())
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("comparison_results/metrics_comparison.png")
    plt.close()
    
    # Vẽ biểu đồ tỷ lệ arbitrage-free
    arbitrage_free_rates = avg_results.groupby('method')['arbitrage_free'].mean()
    
    plt.figure(figsize=(10, 6))
    plt.bar(arbitrage_free_rates.index, arbitrage_free_rates.values)
    plt.title('Tỷ lệ Arbitrage-free theo phương pháp')
    plt.ylabel('Tỷ lệ Arbitrage-free')
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig("comparison_results/arbitrage_free_comparison.png")
    plt.close()
    
    print("\nKết quả trung bình:")
    print(avg_results)
    
    print("\nĐã lưu kết quả chi tiết vào comparison_results/")
    
    return results_df, avg_results

# Hàm chính
def main():
    """Hàm chính để so sánh các phương pháp"""
    # Danh sách file CSV chứa dữ liệu IV
    csv_files = [f for f in os.listdir('.') if f.startswith('pivot_df_')]
    
    print(f"Tìm thấy {len(csv_files)} file dữ liệu")
    
    # Đường dẫn đến mô hình Diffusion đã huấn luyện (nếu có)
    diffusion_model_path = "models/diffusion_model_epoch_50.pt"
    
    # So sánh các phương pháp
    results_df, avg_results = compare_methods(
        csv_files,
        missing_ratio=0.3,
        num_samples=3,
        diffusion_model_path=diffusion_model_path if os.path.exists(diffusion_model_path) else None
    )

if __name__ == "__main__":
    main() 