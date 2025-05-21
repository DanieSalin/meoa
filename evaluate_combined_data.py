#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from fix_model_improved import ImprovedDiffusionUNet, ImprovedDiffusionModel
from real_data_improved import RealIVSurfaceDataset, plot_iv_surface, evaluate_reconstruction

# Thiết lập seed cho khả năng tái tạo
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Kiểm tra thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {device}")

def collect_dataset_stats(dataset_files):
    """
    Thu thập thống kê về bộ dữ liệu
    
    Tham số:
        dataset_files: Danh sách các file CSV
    
    Kết quả:
        dict: Thống kê về bộ dữ liệu
    """
    stats = {
        'total_files': len(dataset_files),
        'strike_prices': [],
        'dtm_ranges': [],
        'iv_ranges': [],
        'file_sizes': []
    }
    
    # Thu thập thông tin từ các file
    for file in dataset_files:
        try:
            df = pd.read_csv(file)
            
            # Tìm cột IV
            iv_col = None
            if 'IV' in df.columns:
                iv_col = 'IV'
            elif 'IV_Call' in df.columns:
                iv_col = 'IV_Call'
            elif 'IV_Put' in df.columns:
                iv_col = 'IV_Put'
            
            if iv_col:
                # Thống kê IV
                iv_values = df[iv_col].dropna()
                if not iv_values.empty:
                    stats['iv_ranges'].append((iv_values.min(), iv_values.mean(), iv_values.max()))
            
            # Thống kê DTM
            if 'Days to Maturity' in df.columns:
                dtm = df['Days to Maturity'].dropna()
                if not dtm.empty:
                    stats['dtm_ranges'].append((dtm.min(), dtm.mean(), dtm.max()))
            
            # Trích Strike từ tên file
            try:
                if 'pivot_df_' in file:
                    strike = int(file.split('pivot_df_')[1].split('.')[0])
                    stats['strike_prices'].append(strike)
            except:
                pass
            
            # Kích thước file
            stats['file_sizes'].append(len(df))
            
        except Exception as e:
            print(f"Lỗi khi xử lý file {file}: {e}")
    
    # Tính toán các thống kê tổng hợp
    if stats['strike_prices']:
        stats['strike_min'] = min(stats['strike_prices'])
        stats['strike_max'] = max(stats['strike_prices'])
        stats['strike_count'] = len(stats['strike_prices'])
    
    if stats['dtm_ranges']:
        stats['dtm_min'] = min([r[0] for r in stats['dtm_ranges']])
        stats['dtm_max'] = max([r[2] for r in stats['dtm_ranges']])
    
    if stats['iv_ranges']:
        stats['iv_min'] = min([r[0] for r in stats['iv_ranges']])
        stats['iv_max'] = max([r[2] for r in stats['iv_ranges']])
        stats['iv_mean'] = np.mean([r[1] for r in stats['iv_ranges']])
    
    if stats['file_sizes']:
        stats['total_samples'] = sum(stats['file_sizes'])
        stats['avg_samples_per_file'] = np.mean(stats['file_sizes'])
    
    return stats

def evaluate_model_performance(model_path, test_files, results_dir, missing_ratios=[0.3, 0.5, 0.7], num_samples=5):
    """
    Đánh giá hiệu suất mô hình với nhiều tỷ lệ dữ liệu bị thiếu
    
    Tham số:
        model_path: Đường dẫn đến mô hình đã lưu
        test_files: Danh sách các file CSV dùng để kiểm thử
        results_dir: Thư mục lưu kết quả
        missing_ratios: Danh sách các tỷ lệ dữ liệu bị thiếu để kiểm thử
        num_samples: Số lượng mẫu cho ensemble averaging
    """
    # Tạo thư mục kết quả nếu chưa tồn tại
    os.makedirs(results_dir, exist_ok=True)
    
    # Tải mô hình
    if not os.path.exists(model_path):
        print(f"Không tìm thấy mô hình tại {model_path}")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    model = ImprovedDiffusionUNet(in_channels=2, out_channels=1).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    diffusion = ImprovedDiffusionModel(model, num_diffusion_steps=200)
    print(f"Đã tải mô hình từ {model_path}")
    
    # Đánh giá với các tỷ lệ dữ liệu bị thiếu khác nhau
    results = []
    
    for missing_ratio in missing_ratios:
        print(f"\n==== Đánh giá với tỷ lệ dữ liệu bị thiếu: {missing_ratio*100}% ====")
        
        # Tạo dataset với tỷ lệ dữ liệu bị thiếu hiện tại
        test_dataset = RealIVSurfaceDataset(csv_files=test_files, missing_ratio=missing_ratio, normalize=True)
        print(f"Tạo dataset với {len(test_dataset)} mẫu")
        
        # Lấy một số mẫu ngẫu nhiên để đánh giá
        test_size = min(5, len(test_dataset))
        test_indices = np.random.choice(len(test_dataset), test_size, replace=False)
        
        # Kết quả đánh giá
        metrics_all = {
            'mse': [],
            'rmse': [],
            'mae': [],
            'mape': []
        }
        
        for i, idx in enumerate(test_indices):
            print(f"Đánh giá mẫu {i+1}/{test_size} (index={idx}):")
            sample = test_dataset[idx]
            
            # Chuẩn bị dữ liệu
            original = sample['original'].unsqueeze(0).to(device)
            masked = sample['masked'].unsqueeze(0).to(device)
            mask = sample['mask'].unsqueeze(0).to(device)
            
            # Phục hồi bề mặt IV
            print(f"Phục hồi bề mặt IV với {num_samples} mẫu ensemble...")
            model.eval()
            with torch.no_grad():
                reconstructed = diffusion.sample(masked, mask, num_samples=num_samples)
            
            # Chuyển về numpy
            original_np = original.cpu().squeeze().numpy()
            masked_np = masked.cpu().squeeze().numpy()
            reconstructed_np = reconstructed.cpu().squeeze().numpy()
            
            # Lưu kết quả cho mẫu đầu tiên mỗi missing ratio
            if i == 0:
                # Vẽ bề mặt gốc, bị thiếu và phục hồi
                plot_iv_surface(
                    original_np, 
                    title=f"Original IV Surface - Missing {missing_ratio*100}%", 
                    save_path=os.path.join(results_dir, f"missing_{int(missing_ratio*100)}_original.png")
                )
                
                plot_iv_surface(
                    masked_np, 
                    title=f"Masked IV Surface - Missing {missing_ratio*100}%", 
                    save_path=os.path.join(results_dir, f"missing_{int(missing_ratio*100)}_masked.png")
                )
                
                plot_iv_surface(
                    reconstructed_np, 
                    title=f"Reconstructed IV Surface - Missing {missing_ratio*100}%", 
                    save_path=os.path.join(results_dir, f"missing_{int(missing_ratio*100)}_reconstructed.png")
                )
            
            # Đánh giá kết quả phục hồi
            metrics = evaluate_reconstruction(original_np, reconstructed_np)
            
            # Thêm vào kết quả tổng hợp
            for key in metrics:
                metrics_all[key].append(metrics[key])
        
        # Tính trung bình các metrics
        avg_metrics = {}
        for key in metrics_all:
            avg_metrics[key] = np.mean(metrics_all[key])
            print(f"Trung bình {key.upper()}: {avg_metrics[key]:.6f}")
        
        # Thêm tỷ lệ missing vào kết quả
        avg_metrics['missing_ratio'] = missing_ratio
        results.append(avg_metrics)
    
    # Tạo DataFrame từ kết quả và lưu
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, "missing_ratio_comparison.csv"), index=False)
    
    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(12, 8))
    
    metrics_to_plot = ['mse', 'rmse', 'mae']
    for metric in metrics_to_plot:
        plt.plot(results_df['missing_ratio'] * 100, results_df[metric], marker='o', label=metric.upper())
    
    plt.xlabel('Tỷ lệ dữ liệu bị thiếu (%)')
    plt.ylabel('Giá trị')
    plt.title('So sánh các metrics theo tỷ lệ dữ liệu bị thiếu')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "missing_ratio_comparison.png"))
    plt.close()
    
    # Vẽ biểu đồ MAPE riêng (thường có giá trị lớn hơn)
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['missing_ratio'] * 100, results_df['mape'], marker='o', color='red')
    plt.xlabel('Tỷ lệ dữ liệu bị thiếu (%)')
    plt.ylabel('MAPE (%)')
    plt.title('MAPE theo tỷ lệ dữ liệu bị thiếu')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "mape_comparison.png"))
    plt.close()
    
    return results_df

def compare_strike_ranges(model_path, spy_files, results_dir):
    """
    So sánh hiệu suất mô hình theo khoảng Strike giá khác nhau
    
    Tham số:
        model_path: Đường dẫn đến mô hình đã lưu
        spy_files: Danh sách các file CSV từ SPY
        results_dir: Thư mục lưu kết quả
    """
    # Tải mô hình
    if not os.path.exists(model_path):
        print(f"Không tìm thấy mô hình tại {model_path}")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    model = ImprovedDiffusionUNet(in_channels=2, out_channels=1).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    diffusion = ImprovedDiffusionModel(model, num_diffusion_steps=200)
    print(f"Đã tải mô hình từ {model_path}")
    
    # Phân loại các file theo khoảng Strike
    strikes = []
    for file in spy_files:
        try:
            strike = int(file.split('pivot_df_')[1].split('.')[0])
            strikes.append(strike)
        except:
            pass
    
    if not strikes:
        print("Không thể trích xuất giá Strike từ tên file")
        return
    
    min_strike = min(strikes)
    max_strike = max(strikes)
    print(f"Khoảng Strike: {min_strike} - {max_strike}")
    
    # Chia thành 3 khoảng: thấp, trung bình, cao
    range_size = (max_strike - min_strike) / 3
    low_range = (min_strike, min_strike + range_size)
    mid_range = (min_strike + range_size, min_strike + 2*range_size)
    high_range = (min_strike + 2*range_size, max_strike)
    
    # Phân loại files
    low_files = []
    mid_files = []
    high_files = []
    
    for file in spy_files:
        try:
            strike = int(file.split('pivot_df_')[1].split('.')[0])
            if low_range[0] <= strike <= low_range[1]:
                low_files.append(file)
            elif mid_range[0] < strike <= mid_range[1]:
                mid_files.append(file)
            elif high_range[0] < strike <= high_range[1]:
                high_files.append(file)
        except:
            pass
    
    print(f"Khoảng Strike thấp ({low_range[0]:.0f}-{low_range[1]:.0f}): {len(low_files)} files")
    print(f"Khoảng Strike trung bình ({mid_range[0]:.0f}-{mid_range[1]:.0f}): {len(mid_files)} files")
    print(f"Khoảng Strike cao ({high_range[0]:.0f}-{high_range[1]:.0f}): {len(high_files)} files")
    
    # Đánh giá hiệu suất mô hình trên từng khoảng
    ranges = [
        ("low", low_files, low_range),
        ("mid", mid_files, mid_range),
        ("high", high_files, high_range)
    ]
    
    results = []
    for name, files, range_values in ranges:
        if not files:
            continue
            
        print(f"\n==== Đánh giá khoảng {name} ({range_values[0]:.0f}-{range_values[1]:.0f}) ====")
        
        # Tạo dataset
        missing_ratio = 0.3
        dataset = RealIVSurfaceDataset(csv_files=files, missing_ratio=missing_ratio, normalize=True)
        
        if len(dataset) == 0:
            print(f"Không có dữ liệu cho khoảng {name}")
            continue
        
        # Lấy một vài mẫu ngẫu nhiên
        test_size = min(3, len(dataset))
        test_indices = np.random.choice(len(dataset), test_size, replace=False)
        
        # Kết quả đánh giá
        metrics_all = {
            'mse': [],
            'rmse': [],
            'mae': [],
            'mape': []
        }
        
        for i, idx in enumerate(test_indices):
            print(f"Đánh giá mẫu {i+1}/{test_size} (index={idx}):")
            sample = dataset[idx]
            
            # Chuẩn bị dữ liệu
            original = sample['original'].unsqueeze(0).to(device)
            masked = sample['masked'].unsqueeze(0).to(device)
            mask = sample['mask'].unsqueeze(0).to(device)
            
            # Phục hồi bề mặt IV
            model.eval()
            with torch.no_grad():
                reconstructed = diffusion.sample(masked, mask, num_samples=5)
            
            # Chuyển về numpy
            original_np = original.cpu().squeeze().numpy()
            masked_np = masked.cpu().squeeze().numpy()
            reconstructed_np = reconstructed.cpu().squeeze().numpy()
            
            # Lưu kết quả cho mẫu đầu tiên của mỗi khoảng
            if i == 0:
                plot_iv_surface(
                    original_np, 
                    title=f"Original IV Surface - {name.capitalize()} Strike Range", 
                    save_path=os.path.join(results_dir, f"strike_{name}_original.png")
                )
                
                plot_iv_surface(
                    masked_np, 
                    title=f"Masked IV Surface - {name.capitalize()} Strike Range", 
                    save_path=os.path.join(results_dir, f"strike_{name}_masked.png")
                )
                
                plot_iv_surface(
                    reconstructed_np, 
                    title=f"Reconstructed IV Surface - {name.capitalize()} Strike Range", 
                    save_path=os.path.join(results_dir, f"strike_{name}_reconstructed.png")
                )
            
            # Đánh giá kết quả phục hồi
            metrics = evaluate_reconstruction(original_np, reconstructed_np)
            
            # Thêm vào kết quả tổng hợp
            for key in metrics:
                metrics_all[key].append(metrics[key])
        
        # Tính trung bình các metrics
        avg_metrics = {}
        for key in metrics_all:
            avg_metrics[key] = np.mean(metrics_all[key])
            print(f"Trung bình {key.upper()}: {avg_metrics[key]:.6f}")
        
        # Thêm thông tin khoảng
        avg_metrics['range'] = name
        avg_metrics['min_strike'] = range_values[0]
        avg_metrics['max_strike'] = range_values[1]
        results.append(avg_metrics)
    
    # Tạo DataFrame từ kết quả và lưu
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(results_dir, "strike_range_comparison.csv"), index=False)
        
        # Vẽ biểu đồ so sánh
        plt.figure(figsize=(10, 6))
        
        # Tạo nhãn cho trục x
        x_labels = [f"{r['range']}\n({r['min_strike']:.0f}-{r['max_strike']:.0f})" for r in results]
        x_pos = np.arange(len(results))
        
        # Vẽ bar chart cho các metrics
        metrics_to_plot = ['mse', 'rmse', 'mae']
        width = 0.25
        offsets = np.linspace(-width, width, len(metrics_to_plot))
        
        for i, metric in enumerate(metrics_to_plot):
            values = [r[metric] for r in results]
            plt.bar(x_pos + offsets[i], values, width=width, label=metric.upper())
        
        plt.xlabel('Khoảng Strike')
        plt.ylabel('Giá trị')
        plt.title('So sánh các metrics theo khoảng Strike')
        plt.xticks(x_pos, x_labels)
        plt.legend()
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(results_dir, "strike_range_comparison.png"))
        plt.close()
        
        # Vẽ biểu đồ MAPE riêng
        plt.figure(figsize=(10, 6))
        mape_values = [r['mape'] for r in results]
        plt.bar(x_pos, mape_values, color='red')
        plt.xlabel('Khoảng Strike')
        plt.ylabel('MAPE (%)')
        plt.title('MAPE theo khoảng Strike')
        plt.xticks(x_pos, x_labels)
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(results_dir, "strike_range_mape.png"))
        plt.close()
        
        return results_df

def main():
    # Thư mục kết quả
    results_dir = "evaluation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Đường dẫn đến mô hình đã huấn luyện
    model_path = "models/best_model.pth"
    
    # Tìm các file dữ liệu
    old_csv_files = [f for f in os.listdir() if f.startswith('pivot_df_') and f.endswith('.csv')]
    spy_csv_files = [f for f in os.listdir() if f.startswith('pivot_df_') and f.endswith('.csv') and not f in old_csv_files]
    
    # Chia dữ liệu SPY thành tập kiểm thử
    spy_test_files = sorted(spy_csv_files)[1::2]  # Lấy một nửa các file (mức giá lẻ)
    
    # Thu thập thống kê về dữ liệu
    print("Thu thập thống kê về dữ liệu...")
    old_stats = collect_dataset_stats(old_csv_files)
    spy_stats = collect_dataset_stats(spy_csv_files)
    
    # In thống kê
    print("\n==== Thống kê dữ liệu cũ ====")
    print(f"Tổng số file: {old_stats['total_files']}")
    print(f"Tổng số mẫu: {old_stats.get('total_samples', 'N/A')}")
    print(f"Khoảng Strike: {old_stats.get('strike_min', 'N/A')} - {old_stats.get('strike_max', 'N/A')}")
    print(f"Khoảng DTM: {old_stats.get('dtm_min', 'N/A')} - {old_stats.get('dtm_max', 'N/A')}")
    print(f"Khoảng IV: {old_stats.get('iv_min', 'N/A'):.4f} - {old_stats.get('iv_max', 'N/A'):.4f}")
    
    print("\n==== Thống kê dữ liệu SPY mới ====")
    print(f"Tổng số file: {spy_stats['total_files']}")
    print(f"Tổng số mẫu: {spy_stats.get('total_samples', 'N/A')}")
    print(f"Khoảng Strike: {spy_stats.get('strike_min', 'N/A')} - {spy_stats.get('strike_max', 'N/A')}")
    print(f"Khoảng DTM: {spy_stats.get('dtm_min', 'N/A')} - {spy_stats.get('dtm_max', 'N/A')}")
    print(f"Khoảng IV: {spy_stats.get('iv_min', 'N/A'):.4f} - {spy_stats.get('iv_max', 'N/A'):.4f}")
    
    # Lưu thống kê
    pd.DataFrame([old_stats, spy_stats], index=['old_data', 'spy_data']).to_csv(
        os.path.join(results_dir, "dataset_statistics.csv")
    )
    
    # Đánh giá hiệu suất mô hình với các tỷ lệ dữ liệu bị thiếu khác nhau
    print("\n\n==== Đánh giá hiệu suất mô hình với các tỷ lệ dữ liệu bị thiếu khác nhau ====")
    results = evaluate_model_performance(
        model_path=model_path,
        test_files=spy_test_files,
        results_dir=results_dir,
        missing_ratios=[0.2, 0.3, 0.4, 0.5, 0.6],
        num_samples=5
    )
    
    # So sánh hiệu suất theo khoảng Strike
    print("\n\n==== So sánh hiệu suất theo khoảng Strike ====")
    strike_results = compare_strike_ranges(
        model_path=model_path,
        spy_files=spy_test_files,
        results_dir=results_dir
    )
    
    print(f"\nĐã hoàn thành đánh giá! Kết quả được lưu trong thư mục {results_dir}/")

if __name__ == "__main__":
    main() 