#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import re

def parse_dat_file(file_path):
    """
    Đọc và phân tích cú pháp của file .dat để trích xuất dữ liệu Implied Volatility
    
    Tham số:
        file_path (str): Đường dẫn đến file .dat
        
    Kết quả:
        tuple: (stock_name, quote_date, records)
            - stock_name: Tên cổ phiếu/ETF
            - quote_date: Ngày trích xuất
            - records: DataFrame chứa dữ liệu option
    """
    # Đọc file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 3:
        print(f"File {file_path} không có đủ dữ liệu")
        return None, None, None
    
    # Lấy tên cổ phiếu từ dòng đầu tiên
    stock_name = lines[0].split(',')[0].strip()
    
    # Lấy ngày báo giá từ dòng thứ hai
    quote_date_str = lines[1].split(',')[0].strip()
    # Xử lý định dạng ngày tháng
    date_match = re.search(r'(\w+\s+\d+\s+\d+)', quote_date_str)
    if date_match:
        quote_date_str = date_match.group(1)
        try:
            quote_date = datetime.strptime(quote_date_str, '%b %d %Y')
        except ValueError:
            quote_date = None
    else:
        quote_date = None
    
    # Lấy headers từ dòng thứ 3
    headers = lines[2].strip().split(',')
    
    # Xử lý dữ liệu từ dòng thứ 4 trở đi
    records = []
    for line in lines[3:]:
        if not line.strip():
            continue
        
        fields = line.strip().split(',')
        if len(fields) < len(headers):
            # Bỏ qua các dòng không đúng định dạng
            continue
        
        record = {}
        for i, header in enumerate(headers):
            if i < len(fields):
                record[header] = fields[i].strip()
        
        records.append(record)
    
    # Chuyển đổi thành DataFrame
    df = pd.DataFrame(records)
    
    return stock_name, quote_date, df

def extract_iv_surface(df, min_volume=0):
    """
    Trích xuất bề mặt Implied Volatility từ DataFrame dữ liệu
    
    Tham số:
        df (DataFrame): DataFrame chứa dữ liệu option
        min_volume (int): Khối lượng giao dịch tối thiểu để lọc
        
    Kết quả:
        DataFrame: Dữ liệu bề mặt IV đã tổ chức
    """
    if df is None or df.empty:
        return None
    
    # Làm sạch dữ liệu
    # Kiểm tra và chuyển đổi các cột số nếu tồn tại
    for col in ['IV', 'Strike', 'Volume', 'Delta']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Kiểm tra nếu cột IV không tồn tại, thử tìm các cột thay thế hoặc tính toán IV
    if 'IV' not in df.columns:
        print("Cột 'IV' không tồn tại trong dữ liệu, đang tìm kiếm cột thay thế...")
        # Kiểm tra các tên cột thường gặp cho IV
        possible_iv_cols = ['ImpliedVolatility', 'Implied_Volatility', 'iv', 'impl_vol']
        for col in possible_iv_cols:
            if col in df.columns:
                print(f"Đã tìm thấy cột {col}, sử dụng làm IV")
                df['IV'] = pd.to_numeric(df[col], errors='coerce')
                break
        
        # Nếu vẫn không tìm thấy, thử tìm trong các cột có chứa từ "IV"
        if 'IV' not in df.columns:
            iv_related_cols = [col for col in df.columns if 'IV' in col or 'iv' in col]
            if iv_related_cols:
                print(f"Sử dụng cột {iv_related_cols[0]} làm IV")
                df['IV'] = pd.to_numeric(df[iv_related_cols[0]], errors='coerce')
    
    # Nếu vẫn không tìm thấy cột IV, sử dụng giá trị mặc định
    if 'IV' not in df.columns:
        print("Không thể tìm thấy cột IV, tạo cột IV mặc định")
        # Thêm cột IV mặc định (có thể điều chỉnh logic này tùy vào dữ liệu)
        df['IV'] = 0.2  # Mặc định IV là 20%
    
    # Kiểm tra và lọc theo Volume nếu cột này tồn tại
    if 'Volume' in df.columns:
        df = df[df['IV'].notna() & (df['Volume'] >= min_volume)]
    else:
        df = df[df['IV'].notna()]
    
    if df.empty:
        print("Sau khi lọc, không có dữ liệu IV hợp lệ")
        return None
    
    # Đảm bảo có cột Strike
    if 'Strike' not in df.columns:
        print("Cột 'Strike' không tồn tại trong dữ liệu, đang tìm kiếm cột thay thế...")
        # Thử tìm cột Strike từ các tên thông dụng
        possible_strike_cols = ['StrikePrice', 'STRIKE', 'strike', 'Strike_Price']
        for col in possible_strike_cols:
            if col in df.columns:
                print(f"Đã tìm thấy cột {col}, sử dụng làm Strike")
                df['Strike'] = pd.to_numeric(df[col], errors='coerce')
                break
        
        # Nếu vẫn không tìm thấy, trích xuất từ tên option
        if 'Strike' not in df.columns and ('Calls' in df.columns or 'Puts' in df.columns):
            print("Trích xuất Strike từ tên option")
            # Thử trích xuất Strike từ tên Calls/Puts sử dụng biểu thức chính quy
            call_strike = []
            if 'Calls' in df.columns:
                call_strike = [re.search(r'C(\d+)', str(x)) for x in df['Calls']]
                call_strike = [int(x.group(1)) if x else None for x in call_strike]
            
            put_strike = []
            if 'Puts' in df.columns:
                put_strike = [re.search(r'P(\d+)', str(x)) for x in df['Puts']]
                put_strike = [int(x.group(1)) if x else None for x in put_strike]
            
            # Kết hợp các giá trị Strike
            if call_strike and put_strike:
                df['Strike'] = [c if c else p for c, p in zip(call_strike, put_strike)]
            elif call_strike:
                df['Strike'] = call_strike
            elif put_strike:
                df['Strike'] = put_strike
    
    # Nếu vẫn không có Strike, tạo giá trị giả
    if 'Strike' not in df.columns or df['Strike'].isna().all():
        print("Không thể trích xuất hoặc tìm thấy cột Strike, tạo giá trị giả")
        # Tạo một dải giá Strike giả từ các giá trị có liên quan
        df['Strike'] = range(400, 400 + len(df))
    
    # Trích xuất ngày hết hạn
    if 'Expiration Date' in df.columns:
        try:
            df['Expiration Date'] = pd.to_datetime(df['Expiration Date'], errors='coerce')
        except:
            print("Không thể chuyển đổi 'Expiration Date' sang định dạng ngày")
    
    # Tính Days to Maturity
    if 'Expiration Date' in df.columns and df['Expiration Date'].notna().any():
        try:
            today = df['Expiration Date'].min()
            df['Days to Maturity'] = (df['Expiration Date'] - today).dt.days
        except:
            print("Không thể tính 'Days to Maturity'")
            df['Days to Maturity'] = 30  # Giá trị mặc định
    else:
        print("Không có cột 'Expiration Date', đặt 'Days to Maturity' mặc định")
        df['Days to Maturity'] = 30  # Giá trị mặc định
    
    # Tạo bề mặt IV dựa trên Days to Maturity và Strike
    # Đầu tiên xử lý riêng cho Calls và Puts
    calls_df = df[df['Calls'].notna()] if 'Calls' in df.columns else pd.DataFrame()
    puts_df = df[df['Puts'].notna()] if 'Puts' in df.columns else pd.DataFrame()
    
    # Tạo pivot table cho IV
    calls_iv = pd.DataFrame()
    puts_iv = pd.DataFrame()
    
    if not calls_df.empty and 'IV' in calls_df.columns and 'Strike' in calls_df.columns:
        try:
            calls_iv = calls_df.pivot_table(
                values='IV', 
                index='Days to Maturity', 
                columns='Strike',
                aggfunc='mean'
            )
            calls_iv.columns = [f'IV_Call_{int(col)}' for col in calls_iv.columns]
        except Exception as e:
            print(f"Không thể tạo pivot table cho calls: {e}")
    
    if not puts_df.empty and 'IV' in puts_df.columns and 'Strike' in puts_df.columns:
        try:
            puts_iv = puts_df.pivot_table(
                values='IV', 
                index='Days to Maturity', 
                columns='Strike',
                aggfunc='mean'
            )
            puts_iv.columns = [f'IV_Put_{int(col)}' for col in puts_iv.columns]
        except Exception as e:
            print(f"Không thể tạo pivot table cho puts: {e}")
    
    # Nếu không thể tạo pivot table cho calls hoặc puts, tạo bảng đơn giản
    if calls_iv.empty and puts_iv.empty:
        print("Không thể tạo pivot table, tạo bảng IV đơn giản")
        
        # Tạo một DataFrame mới với cấu trúc phù hợp
        unique_days = sorted(df['Days to Maturity'].unique())
        unique_strikes = sorted(df['Strike'].unique())
        
        if not unique_days or not unique_strikes:
            print("Không có đủ dữ liệu để tạo bề mặt IV")
            return None
        
        # Tạo bảng trống với các Strike làm cột và Days to Maturity làm chỉ mục
        iv_surface = pd.DataFrame(index=unique_days, columns=[f'IV_Call_{int(s)}' for s in unique_strikes])
        
        # Điền dữ liệu IV vào bảng
        for day in unique_days:
            for strike in unique_strikes:
                day_strike_df = df[(df['Days to Maturity'] == day) & (df['Strike'] == strike)]
                if not day_strike_df.empty and 'IV' in day_strike_df.columns:
                    iv_value = day_strike_df['IV'].mean()
                    iv_surface.at[day, f'IV_Call_{int(strike)}'] = iv_value
        
        iv_surface = iv_surface.reset_index()
        iv_surface.rename(columns={'index': 'Days to Maturity'}, inplace=True)
    else:
        # Kết hợp IV của calls và puts
        iv_surface = pd.concat([calls_iv, puts_iv], axis=1)
        iv_surface = iv_surface.reset_index()
    
    # Thêm thông tin Log Forward Moneyness (sẽ tính sau khi có giá chứng khoán)
    # Tạm thời thêm cột Close_Index giả
    iv_surface['Close_Index'] = 0
    
    return iv_surface

def calculate_moneyness(iv_surface, current_price):
    """
    Tính Log Forward Moneyness cho bề mặt IV
    
    Tham số:
        iv_surface (DataFrame): DataFrame chứa dữ liệu bề mặt IV
        current_price (float): Giá hiện tại của chứng khoán
    
    Kết quả:
        DataFrame: Dữ liệu bề mặt IV đã thêm Log Forward Moneyness
    """
    if iv_surface is None or iv_surface.empty:
        return None
    
    # Cập nhật giá Close_Index
    iv_surface['Close_Index'] = current_price
    
    # Lấy danh sách các cột IV
    iv_cols = [col for col in iv_surface.columns if col.startswith('IV_')]
    
    # Tạo DataFrame mới chứa moneyness
    moneyness_df = pd.DataFrame(index=iv_surface.index)
    moneyness_df['Days to Maturity'] = iv_surface['Days to Maturity']
    
    # Tính Log Forward Moneyness cho từng Strike
    for col in iv_cols:
        # Lấy Strike từ tên cột
        if 'Call' in col:
            strike_str = col.replace('IV_Call_', '')
        else:
            strike_str = col.replace('IV_Put_', '')
        
        try:
            strike = float(strike_str)
            moneyness = np.log(strike / current_price)
            moneyness_df[f'LFM_{strike_str}'] = moneyness
        except:
            pass
    
    # Kết hợp với IV surface
    result = pd.concat([iv_surface, moneyness_df.drop('Days to Maturity', axis=1)], axis=1)
    
    return result

def extract_current_price_from_dat(file_path):
    """
    Trích xuất giá hiện tại từ file .dat
    
    Tham số:
        file_path (str): Đường dẫn đến file .dat
    
    Kết quả:
        float: Giá hiện tại của chứng khoán
    """
    with open(file_path, 'r') as f:
        first_line = f.readline()
    
    # Tìm giá trong dòng đầu tiên
    price_match = re.search(r'Last:\s+([\d.]+)', first_line)
    if price_match:
        try:
            return float(price_match.group(1))
        except ValueError:
            pass
    
    return None

def save_iv_surface(iv_surface, output_path, prefix):
    """
    Lưu bề mặt IV vào các file CSV riêng biệt cho từng strike
    
    Tham số:
        iv_surface (DataFrame): DataFrame chứa dữ liệu bề mặt IV
        output_path (str): Thư mục đầu ra
        prefix (str): Tiền tố cho tên file
    """
    if iv_surface is None or iv_surface.empty:
        print("Không có dữ liệu bề mặt IV để lưu")
        return
    
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_path, exist_ok=True)
    
    # Lưu toàn bộ bề mặt IV
    full_path = os.path.join(output_path, f"{prefix}_iv_surface.csv")
    iv_surface.to_csv(full_path, index=False)
    print(f"Đã lưu bề mặt IV đầy đủ vào {full_path}")
    
    # Lấy danh sách các cột IV call
    call_iv_cols = [col for col in iv_surface.columns if col.startswith('IV_Call_')]
    
    # Lấy danh sách các strikes duy nhất từ các cột IV call
    unique_strikes = sorted(set([int(col.replace('IV_Call_', '')) for col in call_iv_cols]))
    
    # Tạo các file pivot_df cho từng strike
    for strike in unique_strikes:
        # Lọc các cột liên quan đến strike này
        call_col = f'IV_Call_{strike}'
        put_col = f'IV_Put_{strike}'
        lfm_col = f'LFM_{strike}'
        
        # Kiểm tra nếu các cột tồn tại
        cols_to_keep = ['Days to Maturity', 'Close_Index']
        if call_col in iv_surface.columns:
            cols_to_keep.append(call_col)
        if put_col in iv_surface.columns:
            cols_to_keep.append(put_col)
        if lfm_col in iv_surface.columns:
            cols_to_keep.append(lfm_col)
        
        # Tạo DataFrame mới cho strike này
        pivot_df = iv_surface[cols_to_keep].copy()
        
        # Đổi tên các cột IV để phù hợp với định dạng cần thiết
        if call_col in pivot_df.columns:
            pivot_df.rename(columns={call_col: 'IV_Call'}, inplace=True)
        if put_col in pivot_df.columns:
            pivot_df.rename(columns={put_col: 'IV_Put'}, inplace=True)
        if lfm_col in pivot_df.columns:
            pivot_df.rename(columns={lfm_col: 'Log Forward Moneyness'}, inplace=True)
        
        # Thêm cột IV (trung bình của IV call và put nếu có cả hai)
        if 'IV_Call' in pivot_df.columns and 'IV_Put' in pivot_df.columns:
            pivot_df['IV'] = pivot_df[['IV_Call', 'IV_Put']].mean(axis=1)
        elif 'IV_Call' in pivot_df.columns:
            pivot_df['IV'] = pivot_df['IV_Call']
        elif 'IV_Put' in pivot_df.columns:
            pivot_df['IV'] = pivot_df['IV_Put']
        
        # Lưu vào file
        strike_file_path = os.path.join(output_path, f"pivot_df_{strike}.csv")
        pivot_df.to_csv(strike_file_path, index=False)
        print(f"Đã lưu dữ liệu cho strike {strike} vào {strike_file_path}")

def plot_iv_surface(iv_surface, title, output_path=None):
    """
    Vẽ biểu đồ bề mặt Implied Volatility
    
    Tham số:
        iv_surface (DataFrame): DataFrame chứa dữ liệu bề mặt IV
        title (str): Tiêu đề biểu đồ
        output_path (str): Đường dẫn để lưu biểu đồ (nếu cần)
    """
    if iv_surface is None or iv_surface.empty:
        print("Không có dữ liệu bề mặt IV để vẽ biểu đồ")
        return
    
    # Lấy cột Days to Maturity
    if 'Days to Maturity' not in iv_surface.columns:
        print("Không tìm thấy cột 'Days to Maturity' trong dữ liệu")
        return
    
    # Lấy các cột IV
    iv_cols = [col for col in iv_surface.columns if col.startswith('IV_')]
    if not iv_cols:
        print("Không tìm thấy cột IV trong dữ liệu")
        return
    
    # Tạo DataFrame cho plot
    plot_df = iv_surface.copy()
    
    # Lọc các hàng có ít nhất một giá trị IV không phải NaN
    plot_df = plot_df.dropna(subset=iv_cols, how='all')
    
    # Lấy danh sách các strike
    strikes = []
    for col in iv_cols:
        if 'Call' in col:
            strike = int(col.replace('IV_Call_', ''))
        else:
            strike = int(col.replace('IV_Put_', ''))
        strikes.append(strike)
    
    strikes = sorted(set(strikes))
    
    # Chuyển DataFrame thành định dạng phù hợp cho biểu đồ 3D
    X, Y = np.meshgrid(strikes, plot_df['Days to Maturity'])
    Z = np.zeros((len(plot_df), len(strikes)))
    
    for i, strike in enumerate(strikes):
        call_col = f'IV_Call_{strike}'
        put_col = f'IV_Put_{strike}'
        
        if call_col in plot_df.columns and put_col in plot_df.columns:
            # Lấy trung bình của IV call và put
            Z[:, i] = plot_df[[call_col, put_col]].mean(axis=1)
        elif call_col in plot_df.columns:
            Z[:, i] = plot_df[call_col]
        elif put_col in plot_df.columns:
            Z[:, i] = plot_df[put_col]
    
    # Vẽ biểu đồ 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Vẽ bề mặt
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    
    # Thêm colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Implied Volatility')
    
    # Đặt nhãn
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Days to Maturity')
    ax.set_zlabel('Implied Volatility')
    
    # Xoay biểu đồ để có góc nhìn tốt hơn
    ax.view_init(elev=30, azim=45)
    
    # Lưu hoặc hiển thị
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu biểu đồ bề mặt IV vào {output_path}")
    else:
        plt.tight_layout()
        plt.show()
    
    plt.close()

def process_dat_file(file_path, output_dir):
    """
    Xử lý file .dat để trích xuất và lưu bề mặt IV
    
    Tham số:
        file_path (str): Đường dẫn đến file .dat
        output_dir (str): Thư mục đầu ra
    """
    # Trích xuất tên file không bao gồm đường dẫn và phần mở rộng
    file_name = os.path.basename(file_path)
    file_name_wo_ext = os.path.splitext(file_name)[0]
    
    print(f"Đang xử lý file {file_path}...")
    
    # Phân tích cú pháp file .dat
    stock_name, quote_date, df = parse_dat_file(file_path)
    
    if df is None or df.empty:
        print(f"Không thể trích xuất dữ liệu từ file {file_path}")
        return
    
    print(f"Đã tìm thấy {len(df)} bản ghi cho {stock_name}")
    
    # Trích xuất giá hiện tại
    current_price = extract_current_price_from_dat(file_path)
    if current_price is None:
        print(f"Không thể trích xuất giá hiện tại từ file {file_path}, sử dụng giá mặc định")
        current_price = 100.0
    
    print(f"Giá hiện tại của {stock_name}: {current_price}")
    
    # Trích xuất bề mặt IV
    iv_surface = extract_iv_surface(df, min_volume=0)
    
    if iv_surface is None or iv_surface.empty:
        print(f"Không thể trích xuất bề mặt IV từ file {file_path}")
        return
    
    # Tính Log Forward Moneyness
    iv_surface = calculate_moneyness(iv_surface, current_price)
    
    # Lưu bề mặt IV vào file CSV
    save_iv_surface(iv_surface, output_dir, file_name_wo_ext)
    
    # Vẽ và lưu biểu đồ bề mặt IV
    plot_path = os.path.join(output_dir, f"{file_name_wo_ext}_iv_surface.png")
    plot_iv_surface(iv_surface, f"Implied Volatility Surface - {stock_name}", plot_path)
    
    print(f"Đã hoàn thành xử lý file {file_path}")

def main():
    """
    Hàm chính để xử lý các file .dat
    """
    # Thư mục chứa file .dat
    input_dir = "CSVs used for Code"
    
    # Thư mục đầu ra cho file CSV
    output_dir = "."
    
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Danh sách file .dat
    dat_files = [f for f in os.listdir(input_dir) if f.endswith('.dat')]
    
    if not dat_files:
        print(f"Không tìm thấy file .dat trong thư mục {input_dir}")
        return
    
    print(f"Tìm thấy {len(dat_files)} file .dat để xử lý")
    
    # Xử lý từng file .dat
    for file in dat_files:
        file_path = os.path.join(input_dir, file)
        process_dat_file(file_path, output_dir)
    
    print("Đã hoàn thành việc xử lý tất cả các file .dat")

if __name__ == "__main__":
    main() 