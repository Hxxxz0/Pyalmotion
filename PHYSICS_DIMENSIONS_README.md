# 物理表征对齐 - 关键维度索引指南

## 概述

本文档记录了StableMoFusion模型中263维运动表示的物理意义和关键维度索引，用于物理表征对齐任务。

## 总体维度结构 (263维)

```
Motion Representation (263 dimensions):
├── Root Data (4 dimensions): [0:4]
├── RIC Data (63 dimensions): [4:67] 
├── Rotation Data (126 dimensions): [67:193]
├── Local Velocity (66 dimensions): [193:259]
└── Foot Contact (4 dimensions): [259:263]
```

## 详细维度分解

### 1. Root Data - 根关节信息 [0:4]
| 维度索引 | 物理含义 | 描述 |
|---------|---------|------|
| 0 | Root Angular Velocity Y | 根关节绕Y轴的角速度 |
| 1-2 | Root Linear Velocity XZ | 根关节在XZ平面的线性速度 |
| 3 | Root Height Y | 根关节的高度 |

### 2. RIC Data - 关节局部位置 [4:67]
- **总计**: 21个关节 × 3坐标 = 63维
- **索引范围**: [4:67]
- **物理含义**: 除根关节外所有关节的局部位置坐标

#### 重要关节位置索引
| 关节名称 | 维度索引 | 物理重要性 |
|---------|---------|-----------|
| 左髋关节 | [10:13] | 支撑链运动学 |
| 右髋关节 | [7:10] | 支撑链运动学 |
| 左脚踝 | [58:61] | 末端执行器约束 |
| 右脚踝 | [55:58] | 末端执行器约束 |

### 3. Rotation Data - 关节旋转 [67:193]
- **总计**: 21个关节 × 6D旋转 = 126维
- **索引范围**: [67:193]
- **物理含义**: 除根关节外所有关节的6D连续旋转表示

### 4. Local Velocity - 关节速度 [193:259]
- **总计**: 22个关节 × 3坐标 = 66维
- **索引范围**: [193:259]
- **物理含义**: 所有关节的3D局部速度

#### 关键速度维度索引
| 关节名称 | 维度索引 | 物理重要性 |
|---------|---------|-----------|
| 根关节速度 | [193:196] | 质心动量 |
| 左脚速度 | [214:217] | 足地接触约束 |
| 右脚速度 | [217:220] | 足地接触约束 |

### 5. Foot Contact - 足部接触 [259:263]
| 维度索引 | 物理含义 | 描述 |
|---------|---------|------|
| 259-260 | 左脚接触状态 | 左脚与地面的接触信息 |
| 261-262 | 右脚接触状态 | 右脚与地面的接触信息 |

## 🎯 物理表征对齐推荐维度

### 最有价值的29维选择

```python
# 物理对齐关键维度索引
PHYSICS_ALIGNMENT_DIMS = {
    # 优先级1: 核心物理约束 (10维)
    'root_angular_vel': [0],                    # 根角速度Y
    'root_linear_vel': [1, 2],                  # 根线性速度XZ  
    'root_height': [3],                         # 根高度Y
    'foot_contact': [259, 260, 261, 262],       # 足部接触状态
    
    # 优先级2: 运动学约束 (19维)
    'hip_positions': [7, 8, 9, 10, 11, 12],    # 髋关节位置
    'ankle_positions': [55, 56, 57, 58, 59, 60], # 脚踝位置
    'root_velocity_3d': [193, 194, 195],        # 根关节3D速度
    'left_foot_vel': [214, 215, 216],           # 左脚速度
    'right_foot_vel': [217, 218, 219],          # 右脚速度
}

# 扁平化索引列表 (29维)
PHYSICS_DIMS_FLAT = [0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 
                     55, 56, 57, 58, 59, 60, 193, 194, 195,
                     214, 215, 216, 217, 218, 219, 
                     259, 260, 261, 262]
```

### 维度选择理由

| 维度类别 | 物理重要性 | 约束类型 |
|---------|-----------|----------|
| 根部运动 | ⭐⭐⭐⭐⭐ | 动量守恒、惯性约束 |
| 足部接触 | ⭐⭐⭐⭐⭐ | 支撑约束、无滑动条件 |
| 关节速度 | ⭐⭐⭐⭐ | 运动连续性、能量约束 |
| 关键位置 | ⭐⭐⭐ | 运动学链约束 |

## 🔧 实现记录 - 物理对齐系统

### 2024-08-28 更新: 中间层物理表征对齐

#### 1. 模型修改 (`models/unet.py`)
- **新增**: `CondUnet1D` 中的物理投影头
  ```python
  self.physics_head = nn.Sequential(
      nn.Conv1d(dims[-1], dims[-1] // 2, kernel_size=1),  # 降维
      nn.GroupNorm(4, dims[-1] // 2),
      nn.Mish(),
      nn.Conv1d(dims[-1] // 2, physics_dim, kernel_size=1)  # 投影到物理空间
  )
  ```
- **投影位置**: 在 `mid_block1` 之前提取深层语义特征
- **输出格式**: `[B, physics_dim, T_downsampled]` → `[B, T_downsampled, physics_dim]`

#### 2. 训练器修改 (`trainers/ddpm_trainer.py`)
- **物理目标提取**: `extract_physics_info(x_start)` - 从真值运动提取29维物理特征
- **维度组织**: 按权重分组 (根运动2.0, 足接触3.0, 关节速度1.5, 关节位置1.0)
- **时间对齐**: 使用插值解决中间层下采样导致的时间维度不匹配

#### 3. 关键Bug修复: 时间维度对齐
**问题**: 预测物理特征 `[B, 13, 29]` vs 目标物理特征 `[B, 196, 29]`

**根因**: UNet中间层经过多次下采样，时间分辨率从196降到13

**解决方案**: 智能插值对齐
```python
if pred_T < target_T:
    # 下采样目标到预测分辨率 (推荐)
    physics_target_tensor = torch.nn.functional.interpolate(
        physics_target_tensor.transpose(1, 2),  # [B, 29, T]
        size=pred_T,
        mode='linear',
        align_corners=False
    ).transpose(1, 2)  # [B, pred_T, 29]
```

**优势**:
- ✅ 精确维度匹配，避免广播错误
- ✅ 保持物理量连续性
- ✅ 支持任意下采样比例
- ✅ 梯度可导，训练稳定

#### 4. 损失函数设计
```python
def masked_physics_l2(self, pred_physics, target_physics, physics_weights, mask):
    # 加权MSE: 不同物理量使用不同权重
    # 时间掩码: 只在有效帧计算损失
    # 总损失 = 运动重建损失 + λ × 物理对齐损失
```

#### 5. 性能监控
- **TensorBoard记录**: 分别记录运动重建损失、物理对齐损失、总损失
- **权重配置**: `physics_loss_weight = 0.1` (可调)

### 待优化建议

#### 1. 分类型插值策略
- **连续量** (根运动、关节速度/位置): 线性插值或自适应平均池化
- **离散量** (足接触): 最近邻或多数投票，避免0~1软值

#### 2. 多尺度对齐
- 保留当前中间层对齐 (语义约束)
- 增加解码器末端全分辨率对齐 (高频约束，轻权重)

## 使用建议

### 1. 物理损失函数设计
```python
def physics_loss(pred_motion, target_motion):
    # 提取物理关键维度
    pred_physics = pred_motion[:, :, PHYSICS_DIMS_FLAT]
    target_physics = target_motion[:, :, PHYSICS_DIMS_FLAT]
    
    # 加权物理损失
    weights = {
        'root_motion': 2.0,      # 根部运动权重
        'foot_contact': 3.0,     # 足部接触权重  
        'joint_velocity': 1.5,   # 关节速度权重
        'position': 1.0          # 位置权重
    }
    
    return weighted_mse_loss(pred_physics, target_physics, weights)
```

### 2. 物理约束验证
```python
def validate_physics_constraints(motion):
    # 检查足地穿透
    foot_heights = motion[:, :, [58, 61]]  # 脚踝高度
    contact_states = motion[:, :, [259, 260, 261, 262]]
    
    # 检查动量守恒
    root_velocity = motion[:, :, [1, 2]]
    
    # 检查运动连续性
    joint_velocities = motion[:, :, [193:259]]
    
    return physics_score
```

## 数据集信息

- **HumanML3D**: 22个关节，263维表示
- **KIT-ML**: 21个关节，251维表示
- **最大序列长度**: 196帧
- **采样频率**: 20 FPS

## 更新日志

- **2024-08-28**: 实现中间层物理表征对齐系统
  - 添加物理投影头到UNet中间层
  - 实现29维物理特征提取和对齐损失
  - 修复时间维度不匹配bug (插值对齐)
  - 集成训练监控和权重配置
- **2024-12**: 初始版本，基于StableMoFusion模型分析
- 维度索引基于 `utils/motion_process.py` 中的特征提取函数

---

*本文档用于指导物理表征对齐任务中的维度选择和约束设计*
