#!/usr/bin/env python
"""
批量从多个评估结果中提取 FID 并更新 metrics.csv
支持评估多个 checkpoint
"""
import argparse
import re
import pandas as pd
import os
import torch
from os.path import join as pjoin
from glob import glob

def extract_fid_from_log(log_file):
    """从评估 log 文件中提取 FID 值"""
    if not os.path.exists(log_file):
        return None
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # 查找 FID Summary 部分
    fid_summary_match = re.search(
        r'========== FID Summary ==========\n.*?---> \[text2motion\] Mean: ([\d.]+)',
        content,
        re.DOTALL
    )
    
    if fid_summary_match:
        fid_value = float(fid_summary_match.group(1))
        return fid_value
    return None

def get_checkpoint_iteration(checkpoint_path):
    """从 checkpoint 文件中获取迭代数"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        iteration = checkpoint.get('total_it', None)
        return iteration
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        return None

def update_metrics_csv(metrics_csv, iteration, fid_value):
    """更新 metrics.csv 中指定迭代数的 FID 值"""
    if not os.path.exists(metrics_csv):
        print(f"Error: {metrics_csv} not found")
        return False
    
    df = pd.read_csv(metrics_csv)
    
    # 找到对应的迭代数行
    mask = df['iteration'] == iteration
    if not mask.any():
        # 如果迭代数不在 CSV 中，找到最接近的迭代数
        closest_idx = (df['iteration'] - iteration).abs().idxmin()
        closest_iter = df.loc[closest_idx, 'iteration']
        print(f"Warning: Iteration {iteration} not found, using closest {closest_iter}")
        df.loc[closest_idx, 'fid'] = fid_value
    else:
        df.loc[mask, 'fid'] = fid_value
    
    # 保存更新后的 CSV
    df.to_csv(metrics_csv, index=False)
    print(f"Updated iteration {iteration} -> FID {fid_value:.4f}")
    return True

def find_eval_logs(experiment_dir):
    """查找所有评估 log 文件"""
    eval_dir = pjoin(experiment_dir, 'eval')
    if not os.path.exists(eval_dir):
        return []
    
    log_files = glob(pjoin(eval_dir, '*.log'))
    return sorted(log_files)

def extract_iteration_from_log_name(log_file):
    """尝试从 log 文件名中提取迭代数（如果文件名包含迭代数）"""
    # 例如: eval_10000.log -> 10000
    basename = os.path.basename(log_file)
    match = re.search(r'(\d+)', basename)
    if match:
        return int(match.group(1))
    return None

def main():
    parser = argparse.ArgumentParser(description='Batch update metrics.csv with FID from evaluation logs')
    parser.add_argument('--experiment_dir', type=str, required=True,
                       help='Experiment directory')
    parser.add_argument('--model_dir', type=str, default=None,
                       help='Model directory (default: experiment_dir/model)')
    
    args = parser.parse_args()
    
    metrics_csv = pjoin(args.experiment_dir, 'metrics.csv')
    if not os.path.exists(metrics_csv):
        print(f"Error: {metrics_csv} not found")
        return
    
    model_dir = args.model_dir or pjoin(args.experiment_dir, 'model')
    
    # 查找所有 log 文件
    log_files = find_eval_logs(args.experiment_dir)
    if not log_files:
        print(f"No log files found in {pjoin(args.experiment_dir, 'eval')}")
        return
    
    print(f"Found {len(log_files)} log file(s)")
    
    # 处理每个 log 文件
    updated_count = 0
    for log_file in log_files:
        fid_value = extract_fid_from_log(log_file)
        if fid_value is None:
            print(f"Skipping {log_file}: No FID found")
            continue
        
        # 尝试从文件名获取迭代数
        iteration = extract_iteration_from_log_name(log_file)
        
        # 如果文件名中没有，尝试从 checkpoint 获取
        if iteration is None:
            # 假设 log 文件名对应 checkpoint 名
            checkpoint_name = os.path.basename(log_file).replace('.log', '').replace('dpmsolver_10steps_ema', 'latest')
            checkpoint_path = pjoin(model_dir, checkpoint_name + '.tar')
            if os.path.exists(checkpoint_path):
                iteration = get_checkpoint_iteration(checkpoint_path)
            else:
                # 默认使用 latest
                checkpoint_path = pjoin(model_dir, 'latest.tar')
                iteration = get_checkpoint_iteration(checkpoint_path)
        
        if iteration is None:
            print(f"Warning: Could not determine iteration for {log_file}")
            continue
        
        update_metrics_csv(metrics_csv, iteration, fid_value)
        updated_count += 1
    
    print(f"\nTotal updated: {updated_count} FID value(s)")

if __name__ == '__main__':
    main()

