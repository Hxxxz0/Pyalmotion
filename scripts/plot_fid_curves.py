#!/usr/bin/env python
"""
绘制 FID 收敛曲线图
支持多个实验对比，显示训练步数和训练时间 vs FID
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os
from os.path import join as pjoin

def load_metrics(csv_path):
    """加载 metrics.csv 文件"""
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found")
        return None
    
    df = pd.read_csv(csv_path)
    # 过滤掉 FID 为空的行（如果只想画有 FID 的点）
    df_with_fid = df[df['fid'].notna() & (df['fid'] != '')].copy()
    if len(df_with_fid) == 0:
        print(f"Warning: No FID data in {csv_path}")
        return None
    
    # 确保数据类型正确
    df_with_fid['iteration'] = df_with_fid['iteration'].astype(int)
    df_with_fid['hours'] = df_with_fid['hours'].astype(float)
    df_with_fid['fid'] = df_with_fid['fid'].astype(float)
    
    return df_with_fid

def plot_fid_curves(experiment_paths, labels=None, output_path='fid_curves.png', 
                    fid_threshold=None, threshold_label=None):
    """
    绘制 FID 收敛曲线
    
    Args:
        experiment_paths: list of paths to metrics.csv files or experiment directories
        labels: list of labels for each experiment (optional)
        output_path: output figure path
        fid_threshold: FID 阈值线（可选）
        threshold_label: 阈值线的标签（可选）
    """
    fig, ax_iter = plt.subplots(figsize=(10, 6))
    ax_time = ax_iter.twiny()  # 创建第二个 x 轴（训练时间）
    
    # 设置颜色和样式
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    if labels is None:
        labels = [f'Experiment {i+1}' for i in range(len(experiment_paths))]
    
    # 加载并绘制每个实验的数据
    for idx, exp_path in enumerate(experiment_paths):
        # 如果是目录，自动查找 metrics.csv
        if os.path.isdir(exp_path):
            csv_path = pjoin(exp_path, 'metrics.csv')
        else:
            csv_path = exp_path
        
        df = load_metrics(csv_path)
        if df is None or len(df) == 0:
            continue
        
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        # 绘制迭代数 vs FID（实线 + 圆点）
        ax_iter.plot(df['iteration'], df['fid'], 
                    color=color, marker=marker, linestyle='-', 
                    linewidth=2, markersize=6, 
                    label=f'{labels[idx]} (iter)', alpha=0.8)
        
        # 绘制训练时间 vs FID（虚线 + 三角）
        ax_time.plot(df['hours'], df['fid'], 
                    color=color, marker='^', linestyle='--', 
                    linewidth=2, markersize=6, 
                    label=f'{labels[idx]} (time)', alpha=0.8)
    
    # 设置轴标签和刻度
    ax_iter.set_xlabel('Training steps', fontsize=12, fontweight='bold')
    ax_time.set_xlabel('Training time (hours)', fontsize=12, fontweight='bold')
    ax_iter.set_ylabel('FID', fontsize=12, fontweight='bold')
    
    # 使用对数刻度（根据论文中的图）
    ax_iter.set_xscale('log')
    ax_time.set_xscale('log')
    ax_iter.set_yscale('log')
    
    # 设置刻度
    ax_iter.set_xticks([5000, 10000, 15000, 20000, 50000])
    ax_iter.set_xticklabels(['5k', '10k', '15k', '20k', '50k'])
    
    # 时间轴刻度（需要根据实际数据调整）
    # ax_time.set_xticks([1, 2, 3, 4, 6, 8, 10])
    # ax_time.set_xticklabels(['1h', '2h', '3h', '4h', '6h', '8h', '10h'])
    
    # 绘制 FID 阈值线（如果提供）
    if fid_threshold is not None:
        ax_iter.axhline(y=fid_threshold, color='gray', linestyle=':', 
                       linewidth=1.5, alpha=0.7, 
                       label=threshold_label if threshold_label else f'FID={fid_threshold}')
    
    # 图例
    lines1, labels1 = ax_iter.get_legend_handles_labels()
    lines2, labels2 = ax_time.get_legend_handles_labels()
    ax_iter.legend(lines1 + lines2, labels1 + labels2, 
                  loc='upper right', fontsize=10, framealpha=0.9)
    
    # 网格
    ax_iter.grid(True, alpha=0.3, linestyle='--')
    
    # 标题
    plt.title('FID Convergence Curves', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot FID convergence curves')
    parser.add_argument('--experiments', nargs='+', required=True,
                       help='Paths to experiment directories or metrics.csv files')
    parser.add_argument('--labels', nargs='+', default=None,
                       help='Labels for each experiment (optional)')
    parser.add_argument('--output', type=str, default='fid_curves.png',
                       help='Output figure path')
    parser.add_argument('--fid_threshold', type=float, default=None,
                       help='FID threshold line (optional)')
    parser.add_argument('--threshold_label', type=str, default=None,
                       help='Label for threshold line')
    
    args = parser.parse_args()
    
    if args.labels and len(args.labels) != len(args.experiments):
        print("Warning: Number of labels doesn't match number of experiments")
        args.labels = None
    
    plot_fid_curves(args.experiments, args.labels, args.output, 
                   args.fid_threshold, args.threshold_label)

if __name__ == '__main__':
    main()

