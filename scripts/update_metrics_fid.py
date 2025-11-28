#!/usr/bin/env python
"""
从评估结果 log 文件中提取 FID，并更新 metrics.csv
"""
import argparse
import re
import pandas as pd
import os
from os.path import join as pjoin

def extract_fid_from_log(log_file):
    """从评估 log 文件中提取 FID 值"""
    if not os.path.exists(log_file):
        print(f"Error: {log_file} not found")
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
    else:
        print(f"Warning: Could not find FID Summary in {log_file}")
        return None

def get_checkpoint_iteration(checkpoint_path):
    """从 checkpoint 文件中获取迭代数"""
    import torch
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
        print(f"Warning: Iteration {iteration} not found in {metrics_csv}")
        # 可以选择添加新行
        new_row = pd.DataFrame({
            'iteration': [iteration],
            'hours': [None],  # 如果不知道时间，留空
            'fid': [fid_value]
        })
        df = pd.concat([df, new_row], ignore_index=True)
        df = df.sort_values('iteration')
    else:
        df.loc[mask, 'fid'] = fid_value
    
    # 保存更新后的 CSV
    df.to_csv(metrics_csv, index=False)
    print(f"Updated {metrics_csv}: iteration {iteration} -> FID {fid_value:.4f}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Update metrics.csv with FID from evaluation log')
    parser.add_argument('--experiment_dir', type=str, required=True,
                       help='Experiment directory (e.g., checkpoints/t2m/t2m_clip)')
    parser.add_argument('--log_file', type=str, default=None,
                       help='Path to evaluation log file (default: auto-detect)')
    parser.add_argument('--checkpoint', type=str, default='latest.tar',
                       help='Checkpoint file name (default: latest.tar)')
    parser.add_argument('--iteration', type=int, default=None,
                       help='Iteration number (if not provided, will read from checkpoint)')
    
    args = parser.parse_args()
    
    # 确定 metrics.csv 路径
    metrics_csv = pjoin(args.experiment_dir, 'metrics.csv')
    
    # 确定 log 文件路径
    if args.log_file is None:
        # 自动查找 eval 目录下的 log 文件
        eval_dir = pjoin(args.experiment_dir, 'eval')
        if os.path.exists(eval_dir):
            log_files = [f for f in os.listdir(eval_dir) if f.endswith('.log')]
            if log_files:
                # 使用最新的 log 文件
                log_files.sort(key=lambda x: os.path.getmtime(pjoin(eval_dir, x)), reverse=True)
                args.log_file = pjoin(eval_dir, log_files[0])
            else:
                print(f"Error: No log files found in {eval_dir}")
                return
        else:
            print(f"Error: Eval directory not found: {eval_dir}")
            return
    
    # 提取 FID
    fid_value = extract_fid_from_log(args.log_file)
    if fid_value is None:
        return
    
    # 获取迭代数
    if args.iteration is None:
        checkpoint_path = pjoin(args.experiment_dir, 'model', args.checkpoint)
        iteration = get_checkpoint_iteration(checkpoint_path)
        if iteration is None:
            print("Error: Could not determine iteration number")
            return
    else:
        iteration = args.iteration
    
    # 更新 metrics.csv
    update_metrics_csv(metrics_csv, iteration, fid_value)

if __name__ == '__main__':
    main()

