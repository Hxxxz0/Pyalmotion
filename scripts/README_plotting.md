# FID 收敛曲线绘图说明

## 使用方法

### 1. 单个实验绘图

```bash
python scripts/plot_fid_curves.py \
  --experiments checkpoints/t2m/t2m_clip \
  --labels "LUMA (ours)" \
  --output fid_curve_single.png
```

### 2. 多个实验对比

```bash
python scripts/plot_fid_curves.py \
  --experiments \
    checkpoints/t2m/t2m_clip \
    checkpoints/t2m/t2m_no_qwen \
    checkpoints/t2m/t2m_no_film \
  --labels \
    "LUMA (ours)" \
    "w/o Qwen" \
    "w/o Film" \
  --output fid_curves_comparison.png \
  --fid_threshold 0.078 \
  --threshold_label "FID=0.078"
```

### 3. 直接指定 metrics.csv 文件

```bash
python scripts/plot_fid_curves.py \
  --experiments \
    checkpoints/t2m/t2m_clip/metrics.csv \
    checkpoints/t2m/t2m_no_qwen/metrics.csv \
  --labels "With Qwen" "Without Qwen" \
  --output comparison.png
```

## 参数说明

- `--experiments`: 实验目录路径或 metrics.csv 文件路径（可多个）
- `--labels`: 每个实验的标签（可选，默认使用 "Experiment 1", "Experiment 2" 等）
- `--output`: 输出图片路径（默认: `fid_curves.png`）
- `--fid_threshold`: FID 阈值线（可选，用于标记目标性能）
- `--threshold_label`: 阈值线的标签（可选）

## 输出说明

生成的图片包含：
- **实线 + 圆点**: 训练步数 vs FID
- **虚线 + 三角**: 训练时间 vs FID
- **双 x 轴**: 底部是训练步数，顶部是训练时间
- **对数刻度**: 所有轴都使用对数刻度（与论文一致）

## 评估和更新 FID 数据

### 1. 运行评估脚本（和以前一样）

```bash
CUDA_VISIBLE_DEVICES=6 python -m scripts.evaluation \
  --opt_path checkpoints/t2m/t2m_clip/opt.txt \
  --gpu_id 0 \
  --which_ckpt latest
```

### 2. 自动更新 metrics.csv

评估完成后，运行以下命令自动从评估结果中提取 FID 并更新 metrics.csv：

```bash
python scripts/update_metrics_fid.py \
  --experiment_dir checkpoints/t2m/t2m_clip
```

脚本会自动：
- 查找 `eval/` 目录下最新的 log 文件
- 从 log 文件中提取 FID Summary 的 Mean 值
- 从 checkpoint 中读取迭代数
- 更新 `metrics.csv` 中对应迭代数的 FID 值

### 3. 手动指定参数（可选）

```bash
python scripts/update_metrics_fid.py \
  --experiment_dir checkpoints/t2m/t2m_clip \
  --log_file checkpoints/t2m/t2m_clip/eval/dpmsolver_10steps_ema.log \
  --checkpoint latest.tar \
  --iteration 5000
```

## 注意事项

1. 确保 `metrics.csv` 文件存在且包含 FID 数据（FID 列不为空）
2. 如果某个实验没有 FID 数据，该实验会被跳过
3. 图片保存为 300 DPI，适合论文使用
4. 评估脚本运行方式和以前完全一样，只是需要额外运行 `update_metrics_fid.py` 来更新 CSV

