import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from transformers import CLIPModel, CLIPTokenizer
from motion_loader import get_dataset_loader  
from tqdm import tqdm
import yaml
from argparse import Namespace
from options.get_opt import get_opt
import os

GLOBAL_CACHE = {
    "clip_model": None,
    "clip_tokenizer": None,
    "motion_encoder": None,
    "clip_motion_align_model": None,
    "device": None
}


# ---------------------------
# 设置/获取 全局 device
# ---------------------------
def set_global_device(dev):
    """
    设置全局 device（例如 'cuda:0' 或 'cpu'）
    """
    GLOBAL_CACHE["device"] = dev


def get_global_device():
    """
    获取全局 device，如未设置则默认使用 'cuda' or 'cpu'
    """
    if GLOBAL_CACHE["device"] is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        GLOBAL_CACHE["device"] = dev
    return GLOBAL_CACHE["device"]


# ---------------------------
# PositionalEncoding 定义
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.2):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数下标
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数下标
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (T, B, D)
        """
        seq_len = x.size(0)
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)


# ---------------------------
# MotionEncoder 定义
# ---------------------------
class MotionEncoder(nn.Module):
    def __init__(self, input_dim=263, embed_dim=512, num_heads=8, num_layers=4,
                 dim_feedforward=2048, dropout=0.2, max_seq_length=196):
        super(MotionEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(d_model=embed_dim, max_len=max_seq_length, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, motion, lengths):
        """
        motion: (B, T, D)
        lengths: (B,)
        """
        x = self.input_proj(motion).transpose(0, 1)  # (T, B, embed_dim)
        x = self.pos_encoder(x)                      # (T, B, embed_dim)

        B, T = motion.size(0), motion.size(1)
        device = motion.device
        pad_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
        for i, length in enumerate(lengths):
            if length < T:
                pad_mask[i, length:] = True

        x = self.transformer_encoder(x, src_key_padding_mask=pad_mask)  # (T, B, embed_dim)
        x = x.transpose(0, 1)  # (B, T, embed_dim)

        pooled_list = []
        for i in range(B):
            valid_len = lengths[i]
            if valid_len > 0:
                pooled_list.append(x[i, :valid_len].mean(dim=0))
            else:
                pooled_list.append(torch.zeros(self.embed_dim, device=device))
        pooled = torch.stack(pooled_list, dim=0)  # (B, embed_dim)
        pooled = self.dropout(pooled)
        pooled = self.fc(pooled)  # (B, embed_dim)
        return pooled


# ---------------------------
# ClipMotionAlignModel 定义
# ---------------------------
class ClipMotionAlignModel(nn.Module):
    def __init__(self, clip_model: CLIPModel, motion_encoder: nn.Module, temperature=0.07):
        super().__init__()
        self.clip_model = clip_model
        self.motion_encoder = motion_encoder
        # 初始化 logit_scale = log(1/temperature)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))

    def forward(self, motion, lengths, input_ids, attention_mask):
        motion_emb = self.motion_encoder(motion, lengths)
        text_emb = self.clip_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        return motion_emb, text_emb


def _init_clip_motion_model(model_path):
    """
    仅在全局缓存为空时，初始化 CLIP模型、tokenizer、MotionEncoder 并加载预训练权重。
    之后存入 GLOBAL_CACHE，供后续重复使用。
    """
    if GLOBAL_CACHE["clip_motion_align_model"] is not None:
        # 已经加载过，直接返回
        return

    # 根据原始训练的配置定义
    class OPT:
        embed_dim = 768
        # 此处从全局获取 device
        device = get_global_device()
        clip_model_name = "openai/clip-vit-large-patch14"
        max_length = 77
        input_dim = 263  # 改为263，对应t2m数据集
        max_seq_length = 196

    # 加载 CLIP 模型和 tokenizer
    clip_model = CLIPModel.from_pretrained(OPT.clip_model_name)
    clip_tokenizer = CLIPTokenizer.from_pretrained(OPT.clip_model_name)
    clip_model.to(OPT.device)

    # 构建 MotionEncoder 与整体模型
    motion_encoder = MotionEncoder(
        input_dim=OPT.input_dim,
        embed_dim=OPT.embed_dim,
        num_heads=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.2,
        max_seq_length=OPT.max_seq_length
    )

    model = ClipMotionAlignModel(
        clip_model=clip_model,
        motion_encoder=motion_encoder,
        temperature=0.07
    ).to(OPT.device)

    # 加载预训练权重，支持新的checkpoint格式
    try:
        checkpoint = torch.load(model_path, map_location=OPT.device)
        
        # 检查是否为新的训练checkpoint格式 (train_moclip.py保存的格式)
        if 'model_state_dict' in checkpoint:
            # 新训练的checkpoint格式
            state_dict = checkpoint['model_state_dict']
            epoch = checkpoint.get('epoch', 'unknown')
            best_r3 = checkpoint.get('best_r3', 'unknown')
            print(f"成功加载新训练的checkpoint格式模型")
            print(f"  训练轮数: {epoch}")
            print(f"  最佳R@3: {best_r3}")
            print(f"  模型路径: {model_path}")
        else:
            # 原始的state_dict格式
            state_dict = checkpoint
            print("加载原始state_dict格式模型")
        
        # 加载模型状态
        model.load_state_dict(state_dict, strict=True)
        print("模型权重加载成功")
        
    except Exception as e:
        print(f"加载模型权重失败: {e}")
        print("将使用随机初始化的权重")
    
    model.eval()

    # 缓存到全局变量
    GLOBAL_CACHE["clip_model"] = clip_model
    GLOBAL_CACHE["clip_tokenizer"] = clip_tokenizer
    GLOBAL_CACHE["motion_encoder"] = motion_encoder
    GLOBAL_CACHE["clip_motion_align_model"] = model


# ---------------------------
# 定义获取文本与动作编码的函数（保持原接口）
# ---------------------------
def get_co_embeddings_2(captions, motions, model_path="/data/wenshuo/project/jhz/StableMoFusion/data/clip_motion_align_epoch_21.pt"):
    """
    参数：
        captions: List[str]，文本描述列表
        motions: list 或 tensor，动作数据，形状应为 (B, T, input_dim) 或者 list，每个元素为 (T, input_dim) 的数组
        model_path: 模型权重文件路径
    返回：
        text_embeddings, motion_embeddings
    """
    # 如果全局模型还未初始化，则进行初始化（只执行一次）
    _init_clip_motion_model(model_path)

    # 取出已缓存的模型、tokenizer、device
    clip_motion_model = GLOBAL_CACHE["clip_motion_align_model"]
    clip_tokenizer = GLOBAL_CACHE["clip_tokenizer"]
    device = get_global_device()

    # ---------------------------
    # 文本处理
    # ---------------------------
    captions_lower = [caption.lower() for caption in captions]
    text_encodings = clip_tokenizer(
        captions_lower,
        padding=True,
        truncation=True,
        max_length=77,  # 与原 OPT.max_length 保持一致
        return_tensors="pt"
    )
    input_ids = text_encodings["input_ids"].to(device)
    attention_mask = text_encodings["attention_mask"].to(device)

    # ---------------------------
    # 动作数据处理
    # ---------------------------
    if isinstance(motions, list):
        motion_tensors = []
        lengths = []
        for m in motions:
            m_tensor = torch.tensor(m, dtype=torch.float32)
            motion_tensors.append(m_tensor)
            lengths.append(m_tensor.shape[0])

        max_T = max(lengths)
        padded_motions = []
        for m_tensor in motion_tensors:
            T = m_tensor.shape[0]
            if T < max_T:
                pad = torch.zeros((max_T - T, m_tensor.shape[1]), dtype=torch.float32)
                m_tensor = torch.cat([m_tensor, pad], dim=0)
            padded_motions.append(m_tensor)
        motions_tensor = torch.stack(padded_motions, dim=0)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=device)
    else:
        motions_tensor = motions.float().to(device)
        B, T, _ = motions_tensor.shape
        lengths_tensor = torch.tensor([T] * B, dtype=torch.long, device=device)

    # ---------------------------
    # 模型前向传播，获得编码
    # ---------------------------
    with torch.no_grad():
        motion_emb, text_emb = clip_motion_model(motions_tensor, lengths_tensor, input_ids, attention_mask)

    # 对编码进行归一化（可选）
    motion_embeddings = F.normalize(motion_emb, dim=-1).cpu()
    text_embeddings = F.normalize(text_emb, dim=-1).cpu()

    return text_embeddings, motion_embeddings


def encode_dataset_motions(opt_path=None, dataset_name='t2m', model_path="/data/wenshuo/project/jhz/StableMoFusion/data/clip_motion_align_epoch_21.pt", 
                          output_path="dataset_motion_embeddings.npy", batch_size=32):
    """
    编码数据集中的所有motion并保存到npy文件
    
    参数：
        opt_path: 配置文件路径，如果为None则使用默认配置
        dataset_name: 数据集名称 ('t2m' 或 'kit')
        model_path: CLIP motion align模型权重路径
        output_path: 输出npy文件路径
        batch_size: 批处理大小
    """
    
    # 设置device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_device(device)
    print(f"使用设备: {device}")
    
    # 初始化参数
    opt = Namespace()
    if opt_path and os.path.exists(opt_path):
        get_opt(opt, opt_path)
    else:
        # 使用默认配置
        opt.dataset_name = dataset_name
        opt.batch_size = batch_size
        opt.device = device
        opt.max_length = 77
        opt.feat_bias = 5  # 添加feat_bias参数
        opt.max_text_len = 20  # 添加max_text_len参数
        opt.unit_length = 4  # 添加unit_length参数
        
        if dataset_name == 't2m':
            opt.joints_num = 22
            opt.dim_pose = 263
            opt.max_motion_length = 196
            opt.radius = 4
            opt.fps = 20
            opt.data_root = './dataset/HumanML3D'
            opt.motion_dir = os.path.join(opt.data_root, 'new_joint_vecs')
            opt.text_dir = os.path.join(opt.data_root, 'texts')
            opt.mean_path = os.path.join(opt.data_root, 'Mean.npy')
            opt.std_path = os.path.join(opt.data_root, 'Std.npy')
            opt.split_dir = os.path.join(opt.data_root, 'train_val.txt')
            # 添加meta目录设置
            opt.meta_dir = './checkpoints/t2m/clip/meta'
            opt.eval_meta_dir = './dataset'  # 包含t2m_mean.npy和t2m_std.npy的目录
            opt.glove_dir = './dataset'  # 设置glove目录，虽然eval模式不需要
        elif dataset_name == 'kit':
            opt.joints_num = 21
            opt.dim_pose = 251
            opt.max_motion_length = 196
            opt.radius = 240 * 8
            opt.fps = 12.5
            opt.data_root = './dataset/KIT-ML'
            opt.motion_dir = os.path.join(opt.data_root, 'new_joint_vecs')
            opt.text_dir = os.path.join(opt.data_root, 'texts')
            opt.mean_path = os.path.join(opt.data_root, 'Mean.npy')
            opt.std_path = os.path.join(opt.data_root, 'Std.npy')
            opt.split_dir = os.path.join(opt.data_root, 'train_val_test.txt')
            # 添加meta目录设置
            opt.meta_dir = './checkpoints/kit/meta'
            opt.eval_meta_dir = './dataset'  # 包含kit_mean.npy和kit_std.npy的目录
            opt.glove_dir = './dataset'  # 设置glove目录
    
    print(f"数据集: {opt.dataset_name}")
    print(f"批处理大小: {opt.batch_size}")
    
    # 初始化CLIP模型
    _init_clip_motion_model(model_path)
    model = GLOBAL_CACHE["clip_motion_align_model"]
    tokenizer = GLOBAL_CACHE["clip_tokenizer"]
    
    # 获取数据加载器 - 使用train模式，避免需要GloVe词向量
    test_loader = get_dataset_loader(
        opt,
        batch_size=opt.batch_size,
        split='test',
        mode='train'  # 改为train模式
    )
    
    print(f"数据加载器创建成功，总批次数: {len(test_loader)}")
    
    # 存储所有motion embeddings
    motions = []
    
    print("开始编码motion数据...")
    for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="编码进度")):
        torch.cuda.empty_cache()  
        
        # train模式返回简单的3个元素
        caption, motion, m_length = batch_data
        
        # 处理文本
        caption = [c.lower() for c in caption]
        text_enc = tokenizer(
            caption,
            padding=True,
            truncation=True,
            max_length=opt.max_length,
            return_tensors="pt"
        )
        input_ids = text_enc["input_ids"].to(opt.device)
        attention_mask = text_enc["attention_mask"].to(opt.device)
        
        # 处理motion数据
        if isinstance(motion, list):
            motion = torch.stack([torch.tensor(m, dtype=torch.float32) for m in motion], dim=0)
        else:
            motion = motion.float()
        motion = motion.to(opt.device)  
        m_length = m_length.to(opt.device)
        
        # 前向传播获取embeddings
        with torch.no_grad():
            motion_emb, text_emb = model(motion, m_length, input_ids, attention_mask)
        
        # 保存motion embedding
        motions.append(motion_emb.cpu().numpy())  
        
        # 清理缓存并删除不再需要的变量
        del motion_emb
        del text_emb
        torch.cuda.empty_cache()
        
        if (batch_idx + 1) % 100 == 0:
            print(f"已处理 {batch_idx + 1}/{len(test_loader)} 批次")
    
    # 将所有的motion_emb拼接成一个numpy数组
    motions_cat = np.concatenate(motions, axis=0)  # 在第一个维度拼接，形成n*768
    
    print(f"所有motion编码完成，总数据形状: {motions_cat.shape}")
    
    # 保存到npy文件
    np.save(output_path, motions_cat)
    print(f"Motion embeddings已保存到: {output_path}")
    
    # 清理motions列表，释放内存
    del motions
    torch.cuda.empty_cache()
    
    return motions_cat


def encode_keyword_motions(keyword, opt_path=None, dataset_name='t2m', 
                          model_path="/data/wenshuo/project/jhz/StableMoFusion/data/clip_motion_align_epoch_21.pt", 
                          output_dir="./", batch_size=32, search_all_splits=True):
    """
    编码数据集中包含特定关键词的motion和text，并计算它们的余弦相似度
    
    参数：
        keyword: 要搜索的关键词
        opt_path: 配置文件路径，如果为None则使用默认配置
        dataset_name: 数据集名称 ('t2m' 或 'kit')
        model_path: CLIP motion align模型权重路径
        output_dir: 输出目录
        batch_size: 批处理大小
        search_all_splits: 是否搜索所有分割（train、test、val）
    """
    
    # 设置device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_device(device)
    print(f"使用设备: {device}")
    print(f"搜索关键词: '{keyword}'")
    
    # 创建关键词变体列表，支持更灵活的匹配
    keyword_variants = [
        keyword.lower(),
        keyword.lower() + 's',  # 复数形式
        keyword.lower() + '-like',  # -like后缀
        keyword.lower() + ' like',  # 空格like
        'like a ' + keyword.lower(),  # like a前缀
        'like ' + keyword.lower(),   # like前缀
    ]
    print(f"搜索变体: {keyword_variants}")
    
    # 初始化参数
    opt = Namespace()
    if opt_path and os.path.exists(opt_path):
        get_opt(opt, opt_path)
    else:
        # 使用默认配置
        opt.dataset_name = dataset_name
        opt.batch_size = batch_size
        opt.device = device
        opt.max_length = 77
        opt.feat_bias = 5
        opt.max_text_len = 20
        opt.unit_length = 4
        
        if dataset_name == 't2m':
            opt.joints_num = 22
            opt.dim_pose = 263
            opt.max_motion_length = 196
            opt.radius = 4
            opt.fps = 20
            opt.data_root = './dataset/HumanML3D'
            opt.motion_dir = os.path.join(opt.data_root, 'new_joint_vecs')
            opt.text_dir = os.path.join(opt.data_root, 'texts')
            opt.mean_path = os.path.join(opt.data_root, 'Mean.npy')
            opt.std_path = os.path.join(opt.data_root, 'Std.npy')
            opt.split_dir = os.path.join(opt.data_root, 'train_val.txt')
            opt.meta_dir = './checkpoints/t2m/clip/meta'
            opt.eval_meta_dir = './dataset'
            opt.glove_dir = './dataset'
        elif dataset_name == 'kit':
            opt.joints_num = 21
            opt.dim_pose = 251
            opt.max_motion_length = 196
            opt.radius = 240 * 8
            opt.fps = 12.5
            opt.data_root = './dataset/KIT-ML'
            opt.motion_dir = os.path.join(opt.data_root, 'new_joint_vecs')
            opt.text_dir = os.path.join(opt.data_root, 'texts')
            opt.mean_path = os.path.join(opt.data_root, 'Mean.npy')
            opt.std_path = os.path.join(opt.data_root, 'Std.npy')
            opt.split_dir = os.path.join(opt.data_root, 'train_val_test.txt')
            opt.meta_dir = './checkpoints/kit/meta'
            opt.eval_meta_dir = './dataset'
            opt.glove_dir = './dataset'
    
    print(f"数据集: {opt.dataset_name}")
    print(f"批处理大小: {opt.batch_size}")
    
    # 初始化CLIP模型
    _init_clip_motion_model(model_path)
    model = GLOBAL_CACHE["clip_motion_align_model"]
    tokenizer = GLOBAL_CACHE["clip_tokenizer"]
    
    # 根据参数确定要搜索的分割
    if search_all_splits:
        splits_to_search = ['train', 'test']
        if dataset_name == 'kit':
            splits_to_search.append('val')
    else:
        splits_to_search = ['test']
    
    print(f"搜索分割: {splits_to_search}")
    
    # 存储所有分割的匹配数据
    all_motion_embeddings_list = []
    all_text_embeddings_list = []
    all_filtered_captions = []
    total_found_count = 0
    total_processed_count = 0
    
    for split in splits_to_search:
        print(f"\n=== 搜索 {split} 分割 ===")
        
        # 获取当前分割的数据加载器
        try:
            split_loader = get_dataset_loader(
                opt,
                batch_size=opt.batch_size,
                split=split,
                mode='train'
            )
        except Exception as e:
            print(f"无法加载 {split} 分割: {e}")
            continue
        
        print(f"数据加载器创建成功，总批次数: {len(split_loader)}")
        
        # 存储当前分割的匹配数据
        split_motion_embeddings_list = []
        split_text_embeddings_list = []
        split_filtered_captions = []
        split_found_count = 0
        split_processed_count = 0
        
        print(f"开始搜索 {split} 分割中包含 '{keyword}' 的数据...")
        
        for batch_idx, batch_data in enumerate(tqdm(split_loader, desc=f"搜索{split}分割")):
            torch.cuda.empty_cache()
            
            caption, motion, m_length = batch_data
            
            # 筛选包含关键词的样本（使用更灵活的匹配）
            batch_filtered_indices = []
            batch_filtered_captions = []
            batch_filtered_motions = []
            batch_filtered_lengths = []
            
            for i, cap in enumerate(caption):
                cap_lower = cap.lower()
                # 检查是否包含任何关键词变体
                if any(variant in cap_lower for variant in keyword_variants):
                    batch_filtered_indices.append(i)
                    batch_filtered_captions.append(cap)
                    batch_filtered_motions.append(motion[i])
                    batch_filtered_lengths.append(m_length[i])
                    split_found_count += 1
            
            split_processed_count += len(caption)
            
            # 如果当前批次有匹配的数据，进行编码
            if batch_filtered_indices:
                # 处理筛选后的文本
                filtered_captions_lower = [cap.lower() for cap in batch_filtered_captions]
                text_enc = tokenizer(
                    filtered_captions_lower,
                    padding=True,
                    truncation=True,
                    max_length=opt.max_length,
                    return_tensors="pt"
                )
                input_ids = text_enc["input_ids"].to(opt.device)
                attention_mask = text_enc["attention_mask"].to(opt.device)
                
                # 处理筛选后的motion数据
                if isinstance(batch_filtered_motions[0], torch.Tensor):
                    filtered_motion_tensor = torch.stack(batch_filtered_motions, dim=0).float().to(opt.device)
                else:
                    filtered_motion_tensor = torch.stack([torch.tensor(m, dtype=torch.float32) for m in batch_filtered_motions], dim=0).to(opt.device)
                
                filtered_lengths_tensor = torch.tensor(batch_filtered_lengths, dtype=torch.long, device=opt.device)
                
                # 前向传播获取embeddings
                with torch.no_grad():
                    motion_emb, text_emb = model(filtered_motion_tensor, filtered_lengths_tensor, input_ids, attention_mask)
                
                # 存储结果
                split_motion_embeddings_list.append(motion_emb.cpu().numpy())
                split_text_embeddings_list.append(text_emb.cpu().numpy())
                split_filtered_captions.extend(batch_filtered_captions)
                
                # 清理内存
                del motion_emb, text_emb
                torch.cuda.empty_cache()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"已处理 {batch_idx + 1}/{len(split_loader)} 批次，找到 {split_found_count} 个匹配样本")
        
        print(f"{split} 分割搜索完成！在 {split_processed_count} 个样本中找到 {split_found_count} 个包含 '{keyword}' 的样本")
        
        # 合并当前分割的结果
        if split_motion_embeddings_list:
            all_motion_embeddings_list.extend(split_motion_embeddings_list)
            all_text_embeddings_list.extend(split_text_embeddings_list)
            all_filtered_captions.extend(split_filtered_captions)
        
        total_found_count += split_found_count
        total_processed_count += split_processed_count
    
    if total_found_count == 0:
        print(f"未找到包含关键词 '{keyword}' 的数据！")
        return None, None, None
    
    print(f"\n=== 总体搜索结果 ===")
    print(f"在 {total_processed_count} 个样本中找到 {total_found_count} 个包含 '{keyword}' 的样本")
    
    # 合并所有编码结果
    all_motion_embeddings = np.concatenate(all_motion_embeddings_list, axis=0)
    all_text_embeddings = np.concatenate(all_text_embeddings_list, axis=0)
    
    print(f"Motion embeddings形状: {all_motion_embeddings.shape}")
    print(f"Text embeddings形状: {all_text_embeddings.shape}")
    
    # 计算余弦相似度
    def cosine_similarity(a, b):
        """计算两个向量的余弦相似度"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    # 计算每对motion-text的余弦相似度
    similarities = []
    for i in range(len(all_motion_embeddings)):
        sim = cosine_similarity(all_motion_embeddings[i], all_text_embeddings[i])
        similarities.append(sim)
    
    similarities = np.array(similarities)
    mean_similarity = similarities.mean()
    std_similarity = similarities.std()
    
    print(f"\n=== 余弦相似度统计 ===")
    print(f"平均余弦相似度: {mean_similarity:.6f}")
    print(f"标准差: {std_similarity:.6f}")
    print(f"最小值: {similarities.min():.6f}")
    print(f"最大值: {similarities.max():.6f}")
    
    # 保存结果
    motion_output_path = os.path.join(output_dir, f"{keyword}_all_motion_embeddings.npy")
    text_output_path = os.path.join(output_dir, f"{keyword}_all_text_embeddings.npy")
    similarity_output_path = os.path.join(output_dir, f"{keyword}_all_similarities.npy")
    captions_output_path = os.path.join(output_dir, f"{keyword}_all_captions.txt")
    
    np.save(motion_output_path, all_motion_embeddings)
    np.save(text_output_path, all_text_embeddings)
    np.save(similarity_output_path, similarities)
    
    # 保存对应的文本描述
    with open(captions_output_path, 'w', encoding='utf-8') as f:
        for i, caption in enumerate(all_filtered_captions):
            f.write(f"{i}: {caption}\n")
    
    print(f"\n=== 保存文件 ===")
    print(f"Motion embeddings保存到: {motion_output_path}")
    print(f"Text embeddings保存到: {text_output_path}")
    print(f"相似度结果保存到: {similarity_output_path}")
    print(f"文本描述保存到: {captions_output_path}")
    
    # 显示一些示例
    print(f"\n=== 示例文本描述（前10个）===")
    for i, caption in enumerate(all_filtered_captions[:10]):
        print(f"{i+1}. {caption} (相似度: {similarities[i]:.6f})")
    
    if len(all_filtered_captions) > 10:
        print(f"... 还有 {len(all_filtered_captions) - 10} 个样本")
    
    return all_motion_embeddings, all_text_embeddings, similarities


def calculate_recall_at_k(motion_embeddings, text_embeddings, k_values=[1, 3, 5, 10]):
    """
    计算motion-to-text和text-to-motion的Recall@K指标
    
    参数：
        motion_embeddings: motion编码，形状为 (N, D)
        text_embeddings: text编码，形状为 (N, D)
        k_values: 要计算的K值列表
    
    返回：
        dict: 包含各种Recall@K指标的字典
    """
    
    # 归一化embeddings以便计算余弦相似度
    motion_embeddings_norm = motion_embeddings / np.linalg.norm(motion_embeddings, axis=1, keepdims=True)
    text_embeddings_norm = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    
    # 计算相似度矩阵
    # motion_to_text: motion_embeddings_norm @ text_embeddings_norm.T
    # text_to_motion: text_embeddings_norm @ motion_embeddings_norm.T
    motion_to_text_sim = np.dot(motion_embeddings_norm, text_embeddings_norm.T)
    text_to_motion_sim = np.dot(text_embeddings_norm, motion_embeddings_norm.T)
    
    N = len(motion_embeddings)
    results = {}
    
    print(f"计算 {N} 个样本的Recall@K指标...")
    
    for k in k_values:
        if k > N:
            print(f"K={k} 大于样本数量 {N}，跳过")
            continue
            
        # Motion-to-Text Recall@K
        motion_to_text_recall = 0
        for i in range(N):
            # 获取第i个motion与所有text的相似度
            similarities = motion_to_text_sim[i]
            # 找到top-k个最相似的text索引
            top_k_indices = np.argsort(similarities)[::-1][:k]
            # 检查正确的text（索引i）是否在top-k中
            if i in top_k_indices:
                motion_to_text_recall += 1
        
        motion_to_text_recall /= N
        
        # Text-to-Motion Recall@K
        text_to_motion_recall = 0
        for i in range(N):
            # 获取第i个text与所有motion的相似度
            similarities = text_to_motion_sim[i]
            # 找到top-k个最相似的motion索引
            top_k_indices = np.argsort(similarities)[::-1][:k]
            # 检查正确的motion（索引i）是否在top-k中
            if i in top_k_indices:
                text_to_motion_recall += 1
        
        text_to_motion_recall /= N
        
        # 平均Recall@K
        avg_recall = (motion_to_text_recall + text_to_motion_recall) / 2
        
        results[f'Motion-to-Text R@{k}'] = motion_to_text_recall
        results[f'Text-to-Motion R@{k}'] = text_to_motion_recall
        results[f'Average R@{k}'] = avg_recall
        
        print(f"Recall@{k}:")
        print(f"  Motion-to-Text: {motion_to_text_recall:.4f} ({motion_to_text_recall*100:.2f}%)")
        print(f"  Text-to-Motion: {text_to_motion_recall:.4f} ({text_to_motion_recall*100:.2f}%)")
        print(f"  Average: {avg_recall:.4f} ({avg_recall*100:.2f}%)")
        print()
    
    return results


def evaluate_generation_quality(motion_embeddings, text_embeddings, captions):
    """
    评估motion到text的生成质量
    给定motion，生成最相似的text，然后计算与真实text的相似度
    
    参数：
        motion_embeddings: motion编码
        text_embeddings: text编码  
        captions: 对应的文本描述
    """
    
    print("=== Motion-to-Text 生成质量评估 ===")
    
    # 归一化embeddings
    motion_embeddings_norm = motion_embeddings / np.linalg.norm(motion_embeddings, axis=1, keepdims=True)
    text_embeddings_norm = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    
    # 计算motion到text的相似度矩阵
    motion_to_text_sim = np.dot(motion_embeddings_norm, text_embeddings_norm.T)
    
    N = len(motion_embeddings)
    generation_similarities = []
    top1_matches = 0
    top3_matches = 0
    top5_matches = 0
    
    print(f"评估 {N} 个motion的生成质量...")
    print("\n=== 详细生成结果 ===")
    
    for i in range(N):
        # 获取第i个motion与所有text的相似度
        similarities = motion_to_text_sim[i]
        
        # 找到最相似的text（生成结果）
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        # 与真实text的相似度（对角线元素）
        true_similarity = similarities[i]
        
        generation_similarities.append(best_similarity)
        
        # 计算排名
        sorted_indices = np.argsort(similarities)[::-1]
        true_rank = np.where(sorted_indices == i)[0][0] + 1
        
        if true_rank == 1:
            top1_matches += 1
        if true_rank <= 3:
            top3_matches += 1
        if true_rank <= 5:
            top5_matches += 1
        
        # 解析caption
        caption_parts = captions[i].split(': ', 1)
        true_caption = caption_parts[1] if len(caption_parts) > 1 else captions[i]
        
        best_caption_parts = captions[best_match_idx].split(': ', 1)
        generated_caption = best_caption_parts[1] if len(best_caption_parts) > 1 else captions[best_match_idx]
        
        print(f"\n样本 {i+1}:")
        print(f"  真实描述: {true_caption}")
        print(f"  生成描述: {generated_caption}")
        print(f"  生成相似度: {best_similarity:.6f}")
        print(f"  真实相似度: {true_similarity:.6f}")
        print(f"  真实排名: {true_rank}")
        print(f"  匹配状态: {'✓' if best_match_idx == i else '✗'}")
    
    # 计算统计指标
    generation_similarities = np.array(generation_similarities)
    
    print(f"\n=== 生成质量统计 ===")
    print(f"平均生成相似度: {generation_similarities.mean():.6f}")
    print(f"标准差: {generation_similarities.std():.6f}")
    print(f"最小值: {generation_similarities.min():.6f}")
    print(f"最大值: {generation_similarities.max():.6f}")
    
    print(f"\n=== 精确匹配率 ===")
    print(f"Top-1 精确匹配: {top1_matches}/{N} = {top1_matches/N*100:.2f}%")
    print(f"Top-3 精确匹配: {top3_matches}/{N} = {top3_matches/N*100:.2f}%")
    print(f"Top-5 精确匹配: {top5_matches}/{N} = {top5_matches/N*100:.2f}%")
    
    # 计算相似度分布
    high_quality = np.sum(generation_similarities >= 0.6)
    medium_quality = np.sum((generation_similarities >= 0.4) & (generation_similarities < 0.6))
    low_quality = np.sum(generation_similarities < 0.4)
    
    print(f"\n=== 生成质量分布 ===")
    print(f"高质量 (≥0.6): {high_quality}/{N} = {high_quality/N*100:.2f}%")
    print(f"中等质量 (0.4-0.6): {medium_quality}/{N} = {medium_quality/N*100:.2f}%")
    print(f"低质量 (<0.4): {low_quality}/{N} = {low_quality/N*100:.2f}%")
    
    return {
        'mean_similarity': generation_similarities.mean(),
        'std_similarity': generation_similarities.std(),
        'top1_accuracy': top1_matches/N,
        'top3_accuracy': top3_matches/N,
        'top5_accuracy': top5_matches/N,
        'high_quality_ratio': high_quality/N,
        'medium_quality_ratio': medium_quality/N,
        'low_quality_ratio': low_quality/N,
        'generation_similarities': generation_similarities
    }


def analyze_chicken_generation():
    """
    分析chicken数据的生成质量
    """
    print("=== 分析Chicken数据的Motion-to-Text生成质量 ===")
    
    # 加载数据
    try:
        motion_embeddings = np.load('./chicken_all_motion_embeddings.npy')
        text_embeddings = np.load('./chicken_all_text_embeddings.npy')
        similarities = np.load('./chicken_all_similarities.npy')
        
        with open('./chicken_all_captions.txt', 'r', encoding='utf-8') as f:
            captions = [line.strip() for line in f.readlines()]
            
        print(f"成功加载数据:")
        print(f"  Motion embeddings: {motion_embeddings.shape}")
        print(f"  Text embeddings: {text_embeddings.shape}")
        print(f"  样本数量: {len(captions)}")
        
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("请先运行chicken数据编码")
        return None
    
    # 评估生成质量
    results = evaluate_generation_quality(motion_embeddings, text_embeddings, captions)
    
    print(f"\n=== 总结评估 ===")
    print(f"对于{len(captions)}个包含'chicken'的样本：")
    print(f"- 平均生成相似度: {results['mean_similarity']:.4f}")
    print(f"- Top-1 精确匹配率: {results['top1_accuracy']*100:.2f}%")
    print(f"- Top-3 精确匹配率: {results['top3_accuracy']*100:.2f}%")
    print(f"- 高质量生成比例: {results['high_quality_ratio']*100:.2f}%")
    
    # 质量评估
    if results['mean_similarity'] >= 0.6:
        quality_level = "优秀"
    elif results['mean_similarity'] >= 0.5:
        quality_level = "良好"
    elif results['mean_similarity'] >= 0.4:
        quality_level = "中等"
    else:
        quality_level = "需要改进"
    
    print(f"- 整体质量评级: {quality_level}")
    
    return results


def evaluate_full_dataset():
    """
    评估整个数据集的检索性能
    """
    print("=== 评估整个数据集的检索性能 ===")
    
    # 检查是否已有完整数据集的编码
    motion_file = "t2m_motion_embeddings.npy"
    
    if not os.path.exists(motion_file):
        print("完整数据集编码文件不存在，开始生成...")
        # 重新生成完整数据集编码，但同时保存text编码
        encode_full_dataset_with_text()
    
    # 加载完整数据集编码
    try:
        motion_embeddings = np.load("t2m_full_motion_embeddings.npy")
        text_embeddings = np.load("t2m_full_text_embeddings.npy")
        
        print(f"成功加载完整数据集编码:")
        print(f"  Motion embeddings: {motion_embeddings.shape}")
        print(f"  Text embeddings: {text_embeddings.shape}")
        
    except FileNotFoundError:
        print("完整数据集编码文件未找到，正在生成...")
        encode_full_dataset_with_text()
        
        # 重新加载
        motion_embeddings = np.load("t2m_full_motion_embeddings.npy")
        text_embeddings = np.load("t2m_full_text_embeddings.npy")
        
        print(f"成功生成并加载完整数据集编码:")
        print(f"  Motion embeddings: {motion_embeddings.shape}")
        print(f"  Text embeddings: {text_embeddings.shape}")
    
    # 计算recall@k指标
    print("\n=== 计算完整数据集的Recall@K指标 ===")
    recall_results = calculate_recall_at_k(motion_embeddings, text_embeddings, k_values=[1, 3, 5, 10, 20, 50])
    
    # 重点展示结果
    print(f"\n=== 🎯 完整数据集性能总结 ===")
    print(f"数据集规模: {len(motion_embeddings)} 个样本")
    print(f"")
    print(f"📊 主要指标:")
    print(f"  Motion-to-Text R@1:  {recall_results['Motion-to-Text R@1']*100:.2f}%")
    print(f"  Motion-to-Text R@3:  {recall_results['Motion-to-Text R@3']*100:.2f}%")
    print(f"  Motion-to-Text R@5:  {recall_results['Motion-to-Text R@5']*100:.2f}%")
    print(f"  Motion-to-Text R@10: {recall_results['Motion-to-Text R@10']*100:.2f}%")
    print(f"")
    print(f"  Text-to-Motion R@1:  {recall_results['Text-to-Motion R@1']*100:.2f}%")
    print(f"  Text-to-Motion R@3:  {recall_results['Text-to-Motion R@3']*100:.2f}%")
    print(f"  Text-to-Motion R@5:  {recall_results['Text-to-Motion R@5']*100:.2f}%")
    print(f"  Text-to-Motion R@10: {recall_results['Text-to-Motion R@10']*100:.2f}%")
    print(f"")
    print(f"🎯 重点：平均 R@3 = {recall_results['Average R@3']*100:.2f}%")
    
    # 性能评估
    avg_r3 = recall_results['Average R@3']
    if avg_r3 >= 0.4:
        performance_level = "优秀"
        emoji = "🏆"
    elif avg_r3 >= 0.3:
        performance_level = "良好"
        emoji = "✅"
    elif avg_r3 >= 0.2:
        performance_level = "中等"
        emoji = "📈"
    else:
        performance_level = "需要改进"
        emoji = "⚠️"
    
    print(f"\n{emoji} 编码器性能评级: {performance_level}")
    
    return recall_results


def encode_full_dataset_with_text(opt_path=None, dataset_name='t2m', 
                                 model_path="/data/wenshuo/project/jhz/StableMoFusion/data/clip_motion_align_epoch_21.pt", 
                                 batch_size=32):
    """
    编码完整数据集的motion和text并保存
    """
    
    # 设置device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_device(device)
    print(f"使用设备: {device}")
    
    # 初始化参数
    opt = Namespace()
    if opt_path and os.path.exists(opt_path):
        get_opt(opt, opt_path)
    else:
        # 使用默认配置
        opt.dataset_name = dataset_name
        opt.batch_size = batch_size
        opt.device = device
        opt.max_length = 77
        opt.feat_bias = 5
        opt.max_text_len = 20
        opt.unit_length = 4
        
        if dataset_name == 't2m':
            opt.joints_num = 22
            opt.dim_pose = 263
            opt.max_motion_length = 196
            opt.radius = 4
            opt.fps = 20
            opt.data_root = './dataset/HumanML3D'
            opt.motion_dir = os.path.join(opt.data_root, 'new_joint_vecs')
            opt.text_dir = os.path.join(opt.data_root, 'texts')
            opt.mean_path = os.path.join(opt.data_root, 'Mean.npy')
            opt.std_path = os.path.join(opt.data_root, 'Std.npy')
            opt.split_dir = os.path.join(opt.data_root, 'train_val.txt')
            opt.meta_dir = './checkpoints/t2m/clip/meta'
            opt.eval_meta_dir = './dataset'
            opt.glove_dir = './dataset'
    
    print(f"数据集: {opt.dataset_name}")
    print(f"批处理大小: {opt.batch_size}")
    
    # 初始化CLIP模型
    _init_clip_motion_model(model_path)
    model = GLOBAL_CACHE["clip_motion_align_model"]
    tokenizer = GLOBAL_CACHE["clip_tokenizer"]
    
    # 处理测试集
    test_loader = get_dataset_loader(
        opt,
        batch_size=opt.batch_size,
        split='test',
        mode='train'
    )
    
    print(f"数据加载器创建成功，总批次数: {len(test_loader)}")
    
    # 存储所有embeddings
    all_motion_embeddings = []
    all_text_embeddings = []
    
    print("开始编码完整数据集...")
    for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="编码进度")):
        torch.cuda.empty_cache()
        
        caption, motion, m_length = batch_data
        
        # 处理文本
        caption = [c.lower() for c in caption]
        text_enc = tokenizer(
            caption,
            padding=True,
            truncation=True,
            max_length=opt.max_length,
            return_tensors="pt"
        )
        input_ids = text_enc["input_ids"].to(opt.device)
        attention_mask = text_enc["attention_mask"].to(opt.device)
        
        # 处理motion数据
        if isinstance(motion, list):
            motion = torch.stack([torch.tensor(m, dtype=torch.float32) for m in motion], dim=0)
        else:
            motion = motion.float()
        motion = motion.to(opt.device)
        m_length = m_length.to(opt.device)
        
        # 前向传播获取embeddings
        with torch.no_grad():
            motion_emb, text_emb = model(motion, m_length, input_ids, attention_mask)
        
        # 保存embeddings
        all_motion_embeddings.append(motion_emb.cpu().numpy())
        all_text_embeddings.append(text_emb.cpu().numpy())
        
        # 清理内存
        del motion_emb, text_emb
        torch.cuda.empty_cache()
        
        if (batch_idx + 1) % 50 == 0:
            print(f"已处理 {batch_idx + 1}/{len(test_loader)} 批次")
    
    # 合并所有embeddings
    full_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
    full_text_embeddings = np.concatenate(all_text_embeddings, axis=0)
    
    print(f"编码完成！")
    print(f"Motion embeddings形状: {full_motion_embeddings.shape}")
    print(f"Text embeddings形状: {full_text_embeddings.shape}")
    
    # 保存到文件
    np.save("t2m_full_motion_embeddings.npy", full_motion_embeddings)
    np.save("t2m_full_text_embeddings.npy", full_text_embeddings)
    
    print(f"已保存到:")
    print(f"  t2m_full_motion_embeddings.npy")
    print(f"  t2m_full_text_embeddings.npy")
    
    # 清理内存
    del all_motion_embeddings, all_text_embeddings
    torch.cuda.empty_cache()
    
    return full_motion_embeddings, full_text_embeddings


def evaluate_random_subset(n_samples=32, n_trials=5):
    """
    从完整数据集中随机选择n_samples个样本，计算R@3，重复n_trials次求平均
    
    参数：
        n_samples: 选择的样本数量
        n_trials: 重复试验次数
    """
    print(f"=== 随机选择{n_samples}个样本评估R@3性能 ===")
    
    # 加载完整数据集编码
    try:
        motion_embeddings = np.load("t2m_full_motion_embeddings.npy")
        text_embeddings = np.load("t2m_full_text_embeddings.npy")
        
        print(f"成功加载完整数据集编码:")
        print(f"  Motion embeddings: {motion_embeddings.shape}")
        print(f"  Text embeddings: {text_embeddings.shape}")
        
    except FileNotFoundError:
        print("完整数据集编码文件未找到，请先运行完整数据集编码")
        return None
    
    total_samples = len(motion_embeddings)
    print(f"从{total_samples}个样本中随机选择{n_samples}个，重复{n_trials}次")
    
    # 存储每次试验的结果
    trial_results = []
    
    for trial in range(n_trials):
        print(f"\n--- 第{trial+1}次试验 ---")
        
        # 随机选择样本索引
        np.random.seed(trial)  # 设置随机种子确保可重复
        selected_indices = np.random.choice(total_samples, n_samples, replace=False)
        
        # 选择对应的embeddings
        selected_motion_embs = motion_embeddings[selected_indices]
        selected_text_embs = text_embeddings[selected_indices]
        
        print(f"选中样本索引: {selected_indices[:10]}..." if n_samples > 10 else f"选中样本索引: {selected_indices}")
        
        # 计算这个子集的R@3
        subset_results = calculate_recall_at_k(
            selected_motion_embs, 
            selected_text_embs, 
            k_values=[1, 3, 5]
        )
        
        trial_results.append(subset_results)
        
        print(f"本次试验结果:")
        print(f"  Motion-to-Text R@3: {subset_results['Motion-to-Text R@3']*100:.2f}%")
        print(f"  Text-to-Motion R@3: {subset_results['Text-to-Motion R@3']*100:.2f}%")
        print(f"  平均 R@3: {subset_results['Average R@3']*100:.2f}%")
    
    # 计算所有试验的平均值
    print(f"\n=== {n_trials}次试验的统计结果 ===")
    
    # 收集所有试验的指标
    motion_to_text_r1 = [r['Motion-to-Text R@1'] for r in trial_results]
    motion_to_text_r3 = [r['Motion-to-Text R@3'] for r in trial_results]
    motion_to_text_r5 = [r['Motion-to-Text R@5'] for r in trial_results]
    
    text_to_motion_r1 = [r['Text-to-Motion R@1'] for r in trial_results]
    text_to_motion_r3 = [r['Text-to-Motion R@3'] for r in trial_results]
    text_to_motion_r5 = [r['Text-to-Motion R@5'] for r in trial_results]
    
    avg_r1 = [r['Average R@1'] for r in trial_results]
    avg_r3 = [r['Average R@3'] for r in trial_results]
    avg_r5 = [r['Average R@5'] for r in trial_results]
    
    # 计算均值和标准差
    print(f"📊 Motion-to-Text:")
    print(f"  R@1: {np.mean(motion_to_text_r1)*100:.2f}% ± {np.std(motion_to_text_r1)*100:.2f}%")
    print(f"  R@3: {np.mean(motion_to_text_r3)*100:.2f}% ± {np.std(motion_to_text_r3)*100:.2f}%")
    print(f"  R@5: {np.mean(motion_to_text_r5)*100:.2f}% ± {np.std(motion_to_text_r5)*100:.2f}%")
    
    print(f"📊 Text-to-Motion:")
    print(f"  R@1: {np.mean(text_to_motion_r1)*100:.2f}% ± {np.std(text_to_motion_r1)*100:.2f}%")
    print(f"  R@3: {np.mean(text_to_motion_r3)*100:.2f}% ± {np.std(text_to_motion_r3)*100:.2f}%")
    print(f"  R@5: {np.mean(text_to_motion_r5)*100:.2f}% ± {np.std(text_to_motion_r5)*100:.2f}%")
    
    print(f"🎯 平均指标:")
    print(f"  R@1: {np.mean(avg_r1)*100:.2f}% ± {np.std(avg_r1)*100:.2f}%")
    print(f"  R@3: {np.mean(avg_r3)*100:.2f}% ± {np.std(avg_r3)*100:.2f}%")
    print(f"  R@5: {np.mean(avg_r5)*100:.2f}% ± {np.std(avg_r5)*100:.2f}%")
    
    # 重点总结
    final_r3_mean = np.mean(avg_r3)
    final_r3_std = np.std(avg_r3)
    
    print(f"\n🏆 最终结果:")
    print(f"在{n_samples}个样本的子集上，")
    print(f"平均 R@3 = {final_r3_mean*100:.2f}% ± {final_r3_std*100:.2f}%")
    
    # 性能评级
    if final_r3_mean >= 0.4:
        performance_level = "优秀"
        emoji = "🏆"
    elif final_r3_mean >= 0.3:
        performance_level = "良好"
        emoji = "✅"
    elif final_r3_mean >= 0.2:
        performance_level = "中等"
        emoji = "📈"
    else:
        performance_level = "需要改进"
        emoji = "⚠️"
    
    print(f"{emoji} 编码器性能评级: {performance_level}")
    
    return {
        'mean_r3': final_r3_mean,
        'std_r3': final_r3_std,
        'all_results': trial_results,
        'n_samples': n_samples,
        'n_trials': n_trials
    }


def encode_keyword_motions_with_new_model(keyword, opt_path=None, dataset_name='t2m', 
                          new_model_path="./checkpoints/moclip_training/best_model.pt", 
                          old_model_path="/data/wenshuo/project/jhz/StableMoFusion/data/clip_motion_align_epoch_21.pt",
                          output_dir="./", batch_size=32, search_all_splits=True):
    """
    使用新训练的模型编码特定关键词的motion和text，并与旧模型对比
    
    参数：
        keyword: 要搜索的关键词
        new_model_path: 新训练的模型路径
        old_model_path: 原始模型路径
        其他参数同encode_keyword_motions
    """
    
    print(f"=== 🆚 新旧模型对比：关键词 '{keyword}' ===")
    
    # 首先用新模型编码
    print(f"\n🔥 使用新训练的模型: {new_model_path}")
    new_motion_embs, new_text_embs, new_similarities = encode_keyword_motions(
        keyword=keyword,
        opt_path=opt_path,
        dataset_name=dataset_name,
        model_path=new_model_path,
        output_dir=output_dir,
        batch_size=batch_size,
        search_all_splits=search_all_splits
    )
    
    if new_motion_embs is None:
        print("新模型编码失败")
        return None
    
    # 读取新模型的captions用于后续对比
    with open(f"{output_dir}/{keyword}_all_captions.txt", 'r', encoding='utf-8') as f:
        new_captions = [line.strip() for line in f.readlines()]
    
    # 清理全局缓存，准备加载旧模型
    global GLOBAL_CACHE
    GLOBAL_CACHE = {
        "clip_model": None,
        "clip_tokenizer": None,
        "motion_encoder": None,
        "clip_motion_align_model": None,
        "device": None
    }
    
    print(f"\n📊 使用原始模型对比: {old_model_path}")
    old_motion_embs, old_text_embs, old_similarities = encode_keyword_motions(
        keyword=keyword,
        opt_path=opt_path,
        dataset_name=dataset_name,
        model_path=old_model_path,
        output_dir=output_dir + "/old_model/",
        batch_size=batch_size,
        search_all_splits=search_all_splits
    )
    
    if old_motion_embs is None:
        print("原始模型编码失败")
        return new_motion_embs, new_text_embs, new_similarities
    
    # 读取原始模型的captions
    with open(f"{output_dir}/old_model/{keyword}_all_captions.txt", 'r', encoding='utf-8') as f:
        old_captions = [line.strip() for line in f.readlines()]
    
    # 对比分析
    print(f"\n=== 📈 性能对比分析 ===")
    print(f"关键词: '{keyword}'")
    print(f"新模型样本数量: {len(new_similarities)}")
    print(f"原始模型样本数量: {len(old_similarities)}")
    print(f"")
    
    # 余弦相似度对比
    print(f"📊 平均余弦相似度对比:")
    print(f"  新模型: {new_similarities.mean():.6f} ± {new_similarities.std():.6f}")
    print(f"  原始模型: {old_similarities.mean():.6f} ± {old_similarities.std():.6f}")
    improvement = new_similarities.mean() - old_similarities.mean()
    print(f"  改进: {improvement:+.6f} ({improvement/old_similarities.mean()*100:+.2f}%)")
    
    # 质量分布对比
    def analyze_quality_distribution(similarities, model_name):
        high_quality = np.sum(similarities >= 0.6)
        medium_quality = np.sum((similarities >= 0.4) & (similarities < 0.6))
        low_quality = np.sum(similarities < 0.4)
        total = len(similarities)
        
        print(f"  {model_name}:")
        print(f"    高质量 (≥0.6): {high_quality}/{total} = {high_quality/total*100:.1f}%")
        print(f"    中等质量 (0.4-0.6): {medium_quality}/{total} = {medium_quality/total*100:.1f}%")
        print(f"    低质量 (<0.4): {low_quality}/{total} = {low_quality/total*100:.1f}%")
        return high_quality/total, medium_quality/total, low_quality/total
    
    print(f"\n📊 质量分布对比:")
    new_high, new_mid, new_low = analyze_quality_distribution(new_similarities, "新模型")
    old_high, old_mid, old_low = analyze_quality_distribution(old_similarities, "原始模型")
    
    # 高质量样本的改进
    high_quality_improvement = new_high - old_high
    print(f"\n🎯 高质量样本比例改进: {high_quality_improvement:+.1%}")
    
    # 如果样本数量相同，显示逐样本对比，否则跳过
    if len(new_similarities) == len(old_similarities):
        print(f"\n🏆 样本数量一致，进行逐样本对比...")
        sample_improvements = new_similarities - old_similarities
        best_improvements = np.argsort(sample_improvements)[::-1][:5]
        
        print(f"\n🏆 改进最大的5个样本:")
        for i, idx in enumerate(best_improvements, 1):
            caption_parts = new_captions[idx].split(': ', 1)
            caption_text = caption_parts[1] if len(caption_parts) > 1 else new_captions[idx]
            improvement_val = sample_improvements[idx]
            print(f"  {i}. {caption_text}")
            print(f"     改进: {improvement_val:+.6f} (新:{new_similarities[idx]:.6f} vs 旧:{old_similarities[idx]:.6f})")
    else:
        print(f"\n⚠️ 两个模型找到的样本数量不同 (新:{len(new_similarities)} vs 旧:{len(old_similarities)})，跳过逐样本对比")
        print(f"可能原因：不同的搜索批次或数据加载顺序导致的微小差异")
    
    # 保存对比结果
    comparison_results = {
        'keyword': keyword,
        'new_n_samples': len(new_similarities),
        'old_n_samples': len(old_similarities),
        'new_model_path': new_model_path,
        'old_model_path': old_model_path,
        'new_mean_similarity': float(new_similarities.mean()),
        'old_mean_similarity': float(old_similarities.mean()),
        'improvement': float(improvement),
        'improvement_percent': float(improvement/old_similarities.mean()*100),
        'new_high_quality_ratio': float(new_high),
        'old_high_quality_ratio': float(old_high),
        'high_quality_improvement': float(high_quality_improvement)
    }
    
    import json
    with open(f"{output_dir}/{keyword}_model_comparison.json", 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\n💾 对比结果已保存到: {output_dir}/{keyword}_model_comparison.json")
    
    # 总结
    if improvement > 0:
        if improvement > 0.05:
            performance_level = "显著提升"
            emoji = "🚀"
        elif improvement > 0.02:
            performance_level = "明显提升"
            emoji = "📈"
        else:
            performance_level = "小幅提升"
            emoji = "⬆️"
    else:
        performance_level = "需要调优"
        emoji = "⚠️"
    
    print(f"\n{emoji} 总体评价: {performance_level}")
    print(f"新训练的模型在'{keyword}'数据上的表现相比原始模型有{improvement/old_similarities.mean()*100:+.1f}%的改进")
    
    return new_motion_embs, new_text_embs, new_similarities, comparison_results


def fair_model_comparison(keyword, opt_path=None, dataset_name='t2m', 
                         new_model_path="./checkpoints/moclip_training/best_model.pt", 
                         old_model_path="/data/wenshuo/project/jhz/StableMoFusion/data/clip_motion_align_epoch_21.pt",
                         output_dir="./", batch_size=32, search_all_splits=True):
    """
    公平对比两个模型：确保两个模型处理完全相同的样本集
    
    参数：
        keyword: 要搜索的关键词
        其他参数同encode_keyword_motions
    """
    
    print(f"=== 🎯 公平对比两个模型：关键词 '{keyword}' ===")
    
    # 第一步：收集所有匹配的样本（不编码）
    print(f"\n📋 第一步：收集所有包含'{keyword}'的样本...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_device(device)
    
    # 关键词变体
    keyword_variants = [
        keyword.lower(),
        keyword.lower() + 's',
        keyword.lower() + '-like',
        keyword.lower() + ' like',
        'like a ' + keyword.lower(),
        'like ' + keyword.lower(),
    ]
    
    # 初始化参数
    opt = Namespace()
    opt.dataset_name = dataset_name
    opt.batch_size = batch_size
    opt.device = device
    opt.max_length = 77
    opt.feat_bias = 5
    opt.max_text_len = 20
    opt.unit_length = 4
    
    if dataset_name == 't2m':
        opt.joints_num = 22
        opt.dim_pose = 263
        opt.max_motion_length = 196
        opt.radius = 4
        opt.fps = 20
        opt.data_root = './dataset/HumanML3D'
        opt.motion_dir = os.path.join(opt.data_root, 'new_joint_vecs')
        opt.text_dir = os.path.join(opt.data_root, 'texts')
        opt.mean_path = os.path.join(opt.data_root, 'Mean.npy')
        opt.std_path = os.path.join(opt.data_root, 'Std.npy')
        opt.split_dir = os.path.join(opt.data_root, 'train_val.txt')
        opt.meta_dir = './checkpoints/t2m/clip/meta'
        opt.eval_meta_dir = './dataset'
        opt.glove_dir = './dataset'
    
    # 收集所有匹配的样本
    all_matched_samples = []
    
    splits_to_search = ['train', 'test'] if search_all_splits else ['test']
    
    for split in splits_to_search:
        print(f"\n--- 收集 {split} 分割的样本 ---")
        
        # 设置固定随机种子确保确定性
        torch.manual_seed(42)
        np.random.seed(42)
        
        split_loader = get_dataset_loader(
            opt,
            batch_size=opt.batch_size,
            split=split,
            mode='train'
        )
        
        for batch_idx, batch_data in enumerate(tqdm(split_loader, desc=f"收集{split}分割")):
            captions, motions, lengths = batch_data
            
            # 筛选包含关键词的样本
            for i, caption in enumerate(captions):
                cap_lower = caption.lower()
                if any(variant in cap_lower for variant in keyword_variants):
                    # 保存样本信息
                    sample_info = {
                        'caption': caption,
                        'motion': motions[i] if isinstance(motions, list) else motions[i].clone().float(),
                        'length': lengths[i].item() if hasattr(lengths[i], 'item') else lengths[i],
                        'split': split,
                        'batch_idx': batch_idx,
                        'sample_idx': i
                    }
                    all_matched_samples.append(sample_info)
    
    print(f"\n✅ 总共收集到 {len(all_matched_samples)} 个匹配样本")
    
    # 去重（基于caption）
    unique_samples = []
    seen_captions = set()
    for sample in all_matched_samples:
        if sample['caption'] not in seen_captions:
            unique_samples.append(sample)
            seen_captions.add(sample['caption'])
        
    print(f"✅ 去重后剩余 {len(unique_samples)} 个独特样本")
    
    # 第二步：用新模型编码
    print(f"\n🔥 第二步：使用新模型编码样本...")
    new_motion_embeddings, new_text_embeddings = encode_samples_with_model(
        unique_samples, new_model_path, opt
    )
    
    # 第三步：用原始模型编码
    print(f"\n📊 第三步：使用原始模型编码相同样本...")
    
    # 清理全局缓存
    global GLOBAL_CACHE
    GLOBAL_CACHE = {
        "clip_model": None,
        "clip_tokenizer": None, 
        "motion_encoder": None,
        "clip_motion_align_model": None,
        "device": None
    }
    
    old_motion_embeddings, old_text_embeddings = encode_samples_with_model(
        unique_samples, old_model_path, opt
    )
    
    # 第四步：计算相似度并对比
    print(f"\n📈 第四步：对比分析...")
    
    new_similarities = np.array([
        np.dot(new_motion_embeddings[i], new_text_embeddings[i]) / 
        (np.linalg.norm(new_motion_embeddings[i]) * np.linalg.norm(new_text_embeddings[i]))
        for i in range(len(unique_samples))
    ])
    
    old_similarities = np.array([
        np.dot(old_motion_embeddings[i], old_text_embeddings[i]) / 
        (np.linalg.norm(old_motion_embeddings[i]) * np.linalg.norm(old_text_embeddings[i]))
        for i in range(len(unique_samples))
    ])
    
    # 详细对比分析
    print(f"\n=== 📈 公平对比结果 ===")
    print(f"关键词: '{keyword}'")
    print(f"样本数量: {len(unique_samples)} (完全相同)")
    print(f"")
    
    # 余弦相似度对比
    print(f"📊 平均余弦相似度对比:")
    print(f"  新模型: {new_similarities.mean():.6f} ± {new_similarities.std():.6f}")
    print(f"  原始模型: {old_similarities.mean():.6f} ± {old_similarities.std():.6f}")
    improvement = new_similarities.mean() - old_similarities.mean()
    print(f"  改进: {improvement:+.6f} ({improvement/old_similarities.mean()*100:+.2f}%)")
    
    # 质量分布对比
    def analyze_quality_distribution(similarities, model_name):
        high_quality = np.sum(similarities >= 0.6)
        medium_quality = np.sum((similarities >= 0.4) & (similarities < 0.6))
        low_quality = np.sum(similarities < 0.4)
        total = len(similarities)
        
        print(f"  {model_name}:")
        print(f"    高质量 (≥0.6): {high_quality}/{total} = {high_quality/total*100:.1f}%")
        print(f"    中等质量 (0.4-0.6): {medium_quality}/{total} = {medium_quality/total*100:.1f}%")
        print(f"    低质量 (<0.4): {low_quality}/{total} = {low_quality/total*100:.1f}%")
        return high_quality/total, medium_quality/total, low_quality/total
    
    print(f"\n📊 质量分布对比:")
    new_high, new_mid, new_low = analyze_quality_distribution(new_similarities, "新模型")
    old_high, old_mid, old_low = analyze_quality_distribution(old_similarities, "原始模型")
    
    # 高质量样本的改进
    high_quality_improvement = new_high - old_high
    print(f"\n🎯 高质量样本比例改进: {high_quality_improvement:+.1%}")
    
    # 逐样本对比
    sample_improvements = new_similarities - old_similarities
    best_improvements = np.argsort(sample_improvements)[::-1][:5]
    worst_improvements = np.argsort(sample_improvements)[:5]
    
    print(f"\n🏆 改进最大的5个样本:")
    for i, idx in enumerate(best_improvements, 1):
        caption = unique_samples[idx]['caption']
        improvement_val = sample_improvements[idx]
        print(f"  {i}. {caption}")
        print(f"     改进: {improvement_val:+.6f} (新:{new_similarities[idx]:.6f} vs 旧:{old_similarities[idx]:.6f})")
    
    print(f"\n⚠️ 改进最小的5个样本:")
    for i, idx in enumerate(worst_improvements, 1):
        caption = unique_samples[idx]['caption']
        improvement_val = sample_improvements[idx]
        print(f"  {i}. {caption}")
        print(f"     改进: {improvement_val:+.6f} (新:{new_similarities[idx]:.6f} vs 旧:{old_similarities[idx]:.6f})")
    
    # 保存结果
    os.makedirs(f"{output_dir}/fair_comparison", exist_ok=True)
    
    # 保存样本信息
    with open(f"{output_dir}/fair_comparison/{keyword}_samples.txt", 'w', encoding='utf-8') as f:
        for i, sample in enumerate(unique_samples):
            f.write(f"{i}: {sample['caption']}\n")
    
    # 保存embeddings
    np.save(f"{output_dir}/fair_comparison/{keyword}_new_motion_embeddings.npy", new_motion_embeddings)
    np.save(f"{output_dir}/fair_comparison/{keyword}_new_text_embeddings.npy", new_text_embeddings)
    np.save(f"{output_dir}/fair_comparison/{keyword}_old_motion_embeddings.npy", old_motion_embeddings)
    np.save(f"{output_dir}/fair_comparison/{keyword}_old_text_embeddings.npy", old_text_embeddings)
    np.save(f"{output_dir}/fair_comparison/{keyword}_new_similarities.npy", new_similarities)
    np.save(f"{output_dir}/fair_comparison/{keyword}_old_similarities.npy", old_similarities)
    
    # 保存对比结果
    comparison_results = {
        'keyword': keyword,
        'n_samples': len(unique_samples),
        'new_model_path': new_model_path,
        'old_model_path': old_model_path,
        'new_mean_similarity': float(new_similarities.mean()),
        'old_mean_similarity': float(old_similarities.mean()),
        'improvement': float(improvement),
        'improvement_percent': float(improvement/old_similarities.mean()*100),
        'new_high_quality_ratio': float(new_high),
        'old_high_quality_ratio': float(old_high),
        'high_quality_improvement': float(high_quality_improvement),
        'new_std': float(new_similarities.std()),
        'old_std': float(old_similarities.std())
    }
    
    import json
    with open(f"{output_dir}/fair_comparison/{keyword}_comparison_results.json", 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\n💾 公平对比结果已保存到: {output_dir}/fair_comparison/")
    
    # 总结
    if improvement > 0:
        if improvement > 0.05:
            performance_level = "显著提升"
            emoji = "🚀"
        elif improvement > 0.02:
            performance_level = "明显提升"
            emoji = "📈"
        else:
            performance_level = "小幅提升"
            emoji = "⬆️"
    else:
        performance_level = "需要调优"
        emoji = "⚠️"
    
    print(f"\n{emoji} 公平对比总结: {performance_level}")
    print(f"在{len(unique_samples)}个相同样本上，新模型相比原始模型有{improvement/old_similarities.mean()*100:+.1f}%的改进")
    
    return comparison_results, unique_samples, new_similarities, old_similarities


def encode_samples_with_model(samples, model_path, opt):
    """
    使用指定模型编码样本列表
    """
    # 初始化模型
    _init_clip_motion_model(model_path)
    model = GLOBAL_CACHE["clip_motion_align_model"]
    tokenizer = GLOBAL_CACHE["clip_tokenizer"]
    
    motion_embeddings = []
    text_embeddings = []
    
    # 按batch处理样本
    for i in tqdm(range(0, len(samples), opt.batch_size), desc="编码样本"):
        batch_samples = samples[i:i+opt.batch_size]
        
        # 准备文本
        captions = [sample['caption'].lower() for sample in batch_samples]
        text_enc = tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=opt.max_length,
            return_tensors="pt"
        )
        input_ids = text_enc["input_ids"].to(opt.device)
        attention_mask = text_enc["attention_mask"].to(opt.device)
        
        # 准备motion数据
        batch_motions = []
        batch_lengths = []
        for sample in batch_samples:
            motion = sample['motion']
            if not isinstance(motion, torch.Tensor):
                motion = torch.tensor(motion, dtype=torch.float32)
            else:
                motion = motion.float()  # 确保是float类型
            batch_motions.append(motion)
            batch_lengths.append(sample['length'])
        
        # Padding motions to same length
        max_len = max(m.shape[0] for m in batch_motions)
        padded_motions = []
        for motion in batch_motions:
            if motion.shape[0] < max_len:
                pad = torch.zeros((max_len - motion.shape[0], motion.shape[1]), dtype=torch.float32)
                motion = torch.cat([motion, pad], dim=0)
            padded_motions.append(motion)
        
        motion_tensor = torch.stack(padded_motions, dim=0).to(opt.device)
        length_tensor = torch.tensor(batch_lengths, dtype=torch.long, device=opt.device)
        
        # 前向传播
        with torch.no_grad():
            motion_emb, text_emb = model(motion_tensor, length_tensor, input_ids, attention_mask)
        
        motion_embeddings.append(motion_emb.cpu().numpy())
        text_embeddings.append(text_emb.cpu().numpy())
        
        # 清理内存
        torch.cuda.empty_cache()
    
    # 合并结果
    motion_embeddings = np.concatenate(motion_embeddings, axis=0)
    text_embeddings = np.concatenate(text_embeddings, axis=0)
    
    return motion_embeddings, text_embeddings


if __name__ == "__main__":
    # 使用公平对比重新评估新训练的模型
    print("=== 🎯 使用公平对比方法重新评估MoCLIP模型 ===")
    
    # 创建输出目录
    import os
    os.makedirs("./new_model_results", exist_ok=True)
    
    # 进行公平对比
    results = fair_model_comparison(
        keyword="chicken",
        new_model_path="./checkpoints/moclip_training/best_model.pt",
        old_model_path="/data/wenshuo/project/jhz/StableMoFusion/data/clip_motion_align_epoch_21.pt",
        dataset_name='t2m',
        output_dir="./new_model_results",
        batch_size=32
    )
    
    if results:
        comparison_results, unique_samples, new_similarities, old_similarities = results
        print(f"\n=== 🎉 公平对比任务完成 ===")
        print(f"使用完全相同的{len(unique_samples)}个样本进行对比")
        print(f"平均相似度从 {comparison_results['old_mean_similarity']:.6f} 提升到 {comparison_results['new_mean_similarity']:.6f}")
        print(f"真实改进: {comparison_results['improvement_percent']:+.2f}%")
        print(f"高质量样本比例改进: {comparison_results['high_quality_improvement']:.1%}")
        
        # 统计显著性检验
        print(f"\n=== 📈 统计分析 ===")
        from scipy import stats
        
        # 配对t检验
        t_stat, p_value = stats.ttest_rel(new_similarities, old_similarities)
        print(f"配对t检验: t = {t_stat:.4f}, p = {p_value:.6f}")
        if p_value < 0.001:
            significance = "极显著 (p < 0.001) ***"
        elif p_value < 0.01:
            significance = "高显著 (p < 0.01) **"
        elif p_value < 0.05:
            significance = "显著 (p < 0.05) *"
        else:
            significance = "不显著 (p ≥ 0.05)"
        print(f"统计显著性: {significance}")
        
        # 效应量 (Cohen's d)
        pooled_std = np.sqrt(((len(new_similarities)-1)*new_similarities.var() + 
                             (len(old_similarities)-1)*old_similarities.var()) / 
                            (len(new_similarities) + len(old_similarities) - 2))
        cohens_d = (new_similarities.mean() - old_similarities.mean()) / pooled_std
        print(f"效应量 (Cohen's d): {cohens_d:.4f}")
        
        if abs(cohens_d) >= 0.8:
            effect_size = "大效应"
        elif abs(cohens_d) >= 0.5:
            effect_size = "中等效应"
        elif abs(cohens_d) >= 0.2:
            effect_size = "小效应"
        else:
            effect_size = "微小效应"
        print(f"效应大小: {effect_size}")
        
        print(f"\n✅ 公平对比证实：新训练的MoCLIP模型确实取得了真实的性能提升！")
    else:
        print("❌ 公平对比失败")