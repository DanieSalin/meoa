#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from real_data_diffusion import main as run_diffusion

if __name__ == "__main__":
    # Tạo parser cho các tham số dòng lệnh
    parser = argparse.ArgumentParser(description='Chạy mô hình Diffusion để phục hồi bề mặt Implied Volatility')
    
    # Thêm các tham số
    parser.add_argument('--missing_ratio', type=float, default=0.3,
                        help='Tỷ lệ dữ liệu bị thiếu (0.0 - 1.0)')
    parser.add_argument('--train', action='store_true',
                        help='Huấn luyện mô hình mới')
    parser.add_argument('--resume', action='store_true',
                        help='Tiếp tục huấn luyện từ mô hình đã lưu')
    parser.add_argument('--model_path', type=str, default='models/diffusion_model_final.pth',
                        help='Đường dẫn đến mô hình đã lưu')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Số epoch huấn luyện')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Kích thước batch')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    
    # Parse tham số
    args = parser.parse_args()
    
    # Override các tham số trong real_data_diffusion.py
    import real_data_diffusion
    real_data_diffusion.main.__globals__['missing_ratio'] = args.missing_ratio
    real_data_diffusion.main.__globals__['train_model'] = args.train
    real_data_diffusion.main.__globals__['resume_training'] = args.resume
    real_data_diffusion.main.__globals__['model_path'] = args.model_path
    
    # Override các tham số trong train_diffusion_model
    real_data_diffusion.train_diffusion_model.__defaults__ = (
        args.epochs,
        args.batch_size,
        args.lr,
        "models",
        False,
        None
    )
    
    # Chạy hàm main từ real_data_diffusion
    run_diffusion()
    
    print("\nCám ơn bạn đã sử dụng mô hình Diffusion cho phục hồi bề mặt Implied Volatility!")