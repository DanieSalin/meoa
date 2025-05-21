#!/usr/bin/env python
# -*- coding: utf-8 -*-

import PyPDF2
import os
import re
import sys

def extract_text_from_pdf(pdf_path):
    """
    Trích xuất văn bản từ file PDF
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            print(f"PDF có {num_pages} trang")
            
            # Trích xuất văn bản từ mỗi trang
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
                
        return text
    except Exception as e:
        print(f"Lỗi khi đọc PDF: {e}")
        return None

def extract_abstract(text):
    """
    Trích xuất phần Abstract từ văn bản
    """
    abstract_pattern = r'Abstract(?:[^\n]*\n){1,3}((?:.+\n)+?)(?:\n\n|\n[A-Z])'
    match = re.search(abstract_pattern, text, re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    else:
        # Thử một pattern khác
        abstract_pattern2 = r'Abstract[^\n]*\n((?:.+\n)+?)(?:\n\n|\nKeywords)'
        match = re.search(abstract_pattern2, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        else:
            return "Không tìm thấy Abstract"

def extract_intro(text):
    """
    Trích xuất phần Introduction từ văn bản
    """
    intro_pattern = r'Introduction(?:[^\n]*\n){1,3}((?:.+\n)+?)(?:\n\n|\n[A-Z1-9]\.)'
    match = re.search(intro_pattern, text, re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    else:
        # Thử một pattern khác
        intro_pattern2 = r'1\.?\s*Introduction[^\n]*\n((?:.+\n)+?)(?:\n\n|\n2\.)'
        match = re.search(intro_pattern2, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        else:
            return "Không tìm thấy Introduction"

def extract_conclusion(text):
    """
    Trích xuất phần Conclusion từ văn bản
    """
    conclusion_pattern = r'Conclusion(?:[^\n]*\n){1,3}((?:.+\n)+?)(?:\n\n|\nReferences)'
    match = re.search(conclusion_pattern, text, re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    else:
        # Thử một pattern khác
        conclusion_pattern2 = r'\d\.?\s*Conclusion[^\n]*\n((?:.+\n)+?)(?:\n\n|\nReferences)'
        match = re.search(conclusion_pattern2, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        else:
            return "Không tìm thấy Conclusion"

def summarize_pdf(pdf_path):
    """
    Tóm tắt nội dung của file PDF
    """
    print(f"\nĐọc file: {os.path.basename(pdf_path)}")
    text = extract_text_from_pdf(pdf_path)
    
    if text:
        print("\n=== Tóm tắt ===")
        
        # Trích xuất thông tin
        abstract = extract_abstract(text)
        print("\n--- ABSTRACT ---")
        print(abstract[:500] + "..." if len(abstract) > 500 else abstract)
        
        intro = extract_intro(text)
        print("\n--- INTRODUCTION ---")
        print(intro[:500] + "..." if len(intro) > 500 else intro)
        
        conclusion = extract_conclusion(text)
        print("\n--- CONCLUSION ---")
        print(conclusion[:500] + "..." if len(conclusion) > 500 else conclusion)
    
    else:
        print("Không thể đọc file PDF")

if __name__ == "__main__":
    # Liệt kê các file PDF trong thư mục hiện tại
    pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("Không tìm thấy file PDF nào trong thư mục")
        sys.exit(1)
    
    print(f"Tìm thấy {len(pdf_files)} file PDF:")
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"{i}. {pdf_file}")
    
    try:
        # Đọc từng file PDF
        for pdf_file in pdf_files:
            summarize_pdf(pdf_file)
    
    except KeyboardInterrupt:
        print("\nĐã hủy quá trình đọc")
        sys.exit(0) 