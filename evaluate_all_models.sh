#!/bin/bash

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate stablemofusion

# 配置参数
OPT_PATH="/data/wenshuo/project/StableMoFusion/checkpoints/t2m/t2m_condunet1d/opt.txt"
GPU_ID=6
OUTPUT_FILE="all_models_evaluation_results.txt"
MODEL_DIR="/data/wenshuo/project/StableMoFusion/checkpoints/t2m/t2m_condunet1d/model"

echo "============================================================"
echo "批量模型评估脚本"
echo "============================================================"
echo "配置文件路径: $OPT_PATH"
echo "GPU ID: $GPU_ID"
echo "模型目录: $MODEL_DIR"
echo "结果输出文件: $OUTPUT_FILE"
echo "============================================================"

# 检查模型目录是否存在
if [ ! -d "$MODEL_DIR" ]; then
    echo "错误: 模型目录不存在: $MODEL_DIR"
    exit 1
fi

# 获取所有模型文件（去除.tar扩展名）
MODEL_FILES=($(ls $MODEL_DIR/*.tar | xargs -n 1 basename | sed 's/\.tar$//'))

echo "找到 ${#MODEL_FILES[@]} 个模型文件:"
for model in "${MODEL_FILES[@]}"; do
    echo "  - $model.tar"
done
echo "============================================================"

# 创建输出文件
echo "================================================================================" > $OUTPUT_FILE
echo "批量模型评估结果" >> $OUTPUT_FILE
echo "================================================================================" >> $OUTPUT_FILE
echo "评估开始时间: $(date)" >> $OUTPUT_FILE
echo "模型目录: $MODEL_DIR" >> $OUTPUT_FILE
echo "模型数量: ${#MODEL_FILES[@]}" >> $OUTPUT_FILE
echo "================================================================================" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# 循环评估每个模型
total=${#MODEL_FILES[@]}
for i in "${!MODEL_FILES[@]}"; do
    model="${MODEL_FILES[$i]}"
    current=$((i+1))
    
    echo ""
    echo "=================================================="
    echo "正在评估模型 $current/$total: $model"
    echo "=================================================="
    
    # 记录到输出文件
    echo "模型: $model" >> $OUTPUT_FILE
    echo "评估时间: $(date)" >> $OUTPUT_FILE
    echo "----------------------------------------" >> $OUTPUT_FILE
    
    # 运行评估命令
    start_time=$(date +%s)
    
    # 创建临时日志文件来捕获输出
    temp_log="temp_${model}_evaluation.log"
    
    if python -m scripts.evaluation --opt_path "$OPT_PATH" --gpu_id $GPU_ID --which_ckpt "$model" > "$temp_log" 2>&1; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        
        echo "模型 $model 评估成功，用时 ${duration} 秒"
        echo "评估成功，用时: ${duration} 秒" >> $OUTPUT_FILE
        
        # 将临时日志内容添加到输出文件
        echo "评估详细结果:" >> $OUTPUT_FILE
        cat "$temp_log" >> $OUTPUT_FILE
    else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        
        echo "模型 $model 评估失败，用时 ${duration} 秒"
        echo "评估失败，用时: ${duration} 秒" >> $OUTPUT_FILE
        echo "错误日志:" >> $OUTPUT_FILE
        cat "$temp_log" >> $OUTPUT_FILE
    fi
    
    # 删除临时日志文件
    rm -f "$temp_log"
    
    echo "================================================================================" >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
    
    # 清理GPU内存
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
done

# 添加完成标记
echo "================================================================================" >> $OUTPUT_FILE
echo "所有模型评估完成" >> $OUTPUT_FILE
echo "完成时间: $(date)" >> $OUTPUT_FILE
echo "================================================================================" >> $OUTPUT_FILE

echo ""
echo "============================================================"
echo "批量评估完成！"
echo "结果已保存到: $OUTPUT_FILE"
echo "============================================================" 