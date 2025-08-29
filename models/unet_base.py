import clip
import math

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from einops.layers.torch import Rearrange
from einops import rearrange

# 导入MoCLIP相关模块
try:
    from transformers import CLIPModel, CLIPTokenizer
    from MoCLIP import _init_clip_motion_model, GLOBAL_CACHE
    MOCLIP_AVAILABLE = True
except ImportError:
    print("Warning: MoCLIP not available. Using original CLIP encoder.")
    MOCLIP_AVAILABLE = False

class TimestepEmbedder(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(TimestepEmbedder, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[x]

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim_in, dim_out=None):
        super().__init__()
        dim_out = dim_out or dim_in
        self.conv = nn.ConvTranspose1d(dim_in, dim_out, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''
    def __init__(self,
                 inp_channels,
                 out_channels,
                 kernel_size,
                 n_groups=4,
                 zero=False):
        super().__init__()
        self.out_channels = out_channels
        self.block =nn.Conv1d(inp_channels,
                      out_channels,
                      kernel_size,
                      padding=kernel_size // 2)
        self.norm = nn.GroupNorm(n_groups, out_channels)
        self.activation =  nn.Mish()

        if zero:
            # zero init the convolution
            nn.init.zeros_(self.block.weight)
            nn.init.zeros_(self.block.bias)

    def forward(self, x):
        """
        Args:
            x: [bs, nfeat, nframes]
        """
        x = self.block(x)

        batch_size, channels, horizon = x.size()
        x = rearrange(x,'batch channels horizon -> (batch horizon) channels') # [bs*seq, nfeats]
        x = self.norm(x)
        x = rearrange(x.reshape(batch_size,horizon,channels),'batch horizon channels -> batch channels horizon')

        return self.activation(x)

class Conv1dAdaGNBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> scale,shift --> Mish
    '''
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=4):
        super().__init__()
        self.out_channels = out_channels
        self.block = nn.Conv1d(inp_channels,
                      out_channels,
                      kernel_size,
                      padding=kernel_size // 2)
        self.group_norm = nn.GroupNorm(n_groups, out_channels)
        self.avtication = nn.Mish()

    def forward(self, x, scale, shift):
        """
        Args:
            x: [bs, nfeat, nframes]
            scale: [bs, out_feat, 1]
            shift: [bs, out_feat, 1]
        """
        x = self.block(x)

        batch_size, channels, horizon = x.size()
        x = rearrange(x,'batch channels horizon -> (batch horizon) channels') # [bs*seq, nfeats]
        x = self.group_norm(x)
        x = rearrange(x.reshape(batch_size,horizon,channels),'batch horizon channels -> batch channels horizon')

        x = ada_shift_scale(x, shift, scale)

        return self.avtication(x)

def ada_shift_scale(x, shift, scale):
    return x * (1 + scale) + shift

class ResidualTemporalBlock(nn.Module):
    def __init__(self,
                 inp_channels,
                 out_channels,
                 embed_dim,
                 kernel_size=5,
                 zero=True,
                 n_groups=8,
                 dropout: float = 0.1,
                 adagn=True):
        super().__init__()
        self.adagn = adagn
        
        self.blocks = nn.ModuleList([
            # adagn only the first conv (following guided-diffusion)
            (Conv1dAdaGNBlock(inp_channels, out_channels, kernel_size, n_groups) if adagn
            else Conv1dBlock(inp_channels, out_channels, kernel_size)),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups, zero=zero),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            # adagn = scale and shift
            nn.Linear(embed_dim, out_channels * 2 if adagn else out_channels),
            Rearrange('batch t -> batch t 1'),
        )
        self.dropout = nn.Dropout(dropout)    
        if zero:
            nn.init.zeros_(self.time_mlp[1].weight)
            nn.init.zeros_(self.time_mlp[1].bias)

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, time_embeds=None):
        '''
            x : [ batch_size x inp_channels x nframes ]
            t : [ batch_size x embed_dim ]
            returns: [ batch_size x out_channels x nframes ]
        '''
        if self.adagn:
            scale, shift = self.time_mlp(time_embeds).chunk(2, dim=1)
            out = self.blocks[0](x, scale, shift)
        else:
            out = self.blocks[0](x) + self.time_mlp(time_embeds)
        out = self.blocks[1](out)
        out = self.dropout(out)
        return out + self.residual_conv(x)

class CrossAttention(nn.Module):

    def __init__(
        self, 
        latent_dim, 
        text_latent_dim, 
        num_heads:int = 8,
        dropout: float = 0.0
        ):
        super().__init__()
        self.num_head = num_heads
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, xf):
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_head
        # B, T, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, N, D
        key = self.key(self.text_norm(xf)).unsqueeze(1)
        query = query.view(B, T, H, -1)
        key = key.view(B, N, H, -1)
        # B, T, N, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.text_norm(xf)).view(B, N, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        return y

class LinearCrossAttention(nn.Module):
    def __init__(
        self, 
        latent_dim, 
        text_latent_dim, 
        num_heads:int = 8,
        dropout: float = 0.0
        ):
        super().__init__()
        self.num_head = num_heads
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_tensor, condition_tensor):
        """
        input_tensor: B, T, D  
        condition_tensor: B, N, L 
        """
        B, T, D = input_tensor.shape
        N = condition_tensor.shape[1]    
        H = self.num_head
        # B, T, D
        query = self.query(self.norm(input_tensor))
        # B, N, D
        key = self.key(self.text_norm(condition_tensor))
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, N, H, -1), dim=1)
        # B, N, H, HD
        value = self.value(self.text_norm(condition_tensor)).view(B, N, H, -1)
        # B, H, HD, HD
        attention = self.dropout(torch.einsum('bnhd,bnhl->bhdl', key, value))
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        return y

class ResidualCrossAttentionLayer(nn.Module):
    def __init__(
        self, 
        dim1, 
        dim2, 
        num_heads:int = 8,
        dropout: float = 0.1,
        no_eff: bool = False
    ):
        super(ResidualCrossAttentionLayer, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.num_heads = num_heads
        
        # Multi-Head Attention Layer
        if no_eff:
            self.cross_attention = CrossAttention(
                latent_dim=dim1, 
                text_latent_dim = dim2,
                num_heads=num_heads,
                dropout=dropout
            )  
        else:
             self.cross_attention = LinearCrossAttention(
                latent_dim=dim1, 
                text_latent_dim = dim2,
                num_heads=num_heads,
                dropout=dropout
            )  
        
    def forward(self, input_tensor, condition_tensor, cond_indices):
        '''
        input_tensor :B, D, L
        condition_tensor: B, L, D
        '''
        if cond_indices.numel() == 0:
            return input_tensor
        
        x = input_tensor

        # Ensure that the dimensions match for the MultiheadAttention
        x = x[cond_indices].permute(0, 2, 1)  # (batch_size, seq_length, feat_dim)
        
        # Compute cross-attention
        x = self.cross_attention(x, condition_tensor[cond_indices])
        
        # Rearrange output tensor
        x = x.permute(0, 2, 1)  # (batch_size, feat_dim, seq_length)
        
        input_tensor[cond_indices] = input_tensor[cond_indices] + x
        return  input_tensor

class CondConv1DBlock(nn.Module):
    def __init__(self,
                 dim_in, 
                 dim_out, 
                 cond_dim, 
                 time_dim, 
                 adagn=True, 
                 zero=True,
                 no_eff=False,
                 dropout: float = 0.1,) -> None:
        super().__init__()
        self.conv1d = ResidualTemporalBlock(dim_in,
                                          dim_out,
                                          embed_dim=time_dim,
                                          adagn=adagn,
                                          zero=zero,
                                          dropout=dropout)
        self.cross_attn = ResidualCrossAttentionLayer(dim1=dim_out,
                                        dim2=cond_dim,
                                        no_eff=no_eff,
                                        dropout=dropout)
    def forward(self, x, t, cond, cond_indices=None):
        x = self.conv1d(x, t)
        x = self.cross_attn(x, cond, cond_indices)
        return x


class CondUnet1D(nn.Module):
    """
        Diffusion's style UNET with 1D convolution and adaptive group normalization for motion suquence denoising, 
        cross-attention to introduce conditional prompts (like text).
    """
    def __init__(
            self,
            input_dim,
            cond_dim,
            dim=128,
            dim_mults=(1, 2, 4, 8),
            dims = None,
            time_dim=512,
            adagn=True,
            zero=True,
            dropout=0.1,
            no_eff=False,
            
    ):
        super().__init__()
        if not dims:
            dims = [input_dim, *map(lambda m: int(dim * m), dim_mults)]  ##[d, d,2d,4d]
        print('dims: ', dims, 'mults: ', dim_mults)
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_mlp = nn.Sequential(
            TimestepEmbedder(time_dim),
            nn.Linear(time_dim , time_dim  * 4),
            nn.Mish(),
            nn.Linear(time_dim  * 4, time_dim ),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            self.downs.append(
                nn.ModuleList([
                    CondConv1DBlock(dim_in,
                                 dim_out,
                                 cond_dim,
                                 time_dim,
                                 adagn=adagn,
                                 zero=zero,
                                 no_eff=no_eff,
                                 dropout=dropout),
                    CondConv1DBlock(dim_out,
                                 dim_out,
                                 cond_dim,
                                 time_dim,
                                 adagn=adagn,
                                 zero=zero,
                                 no_eff=no_eff,
                                 dropout=dropout), 
                    Downsample1d(dim_out) 
                ]))

        mid_dim = dims[-1]
        self.mid_block1 = CondConv1DBlock(dim_in=mid_dim,
                                 dim_out=mid_dim,
                                 cond_dim=cond_dim,
                                 time_dim=time_dim,
                                 adagn=adagn,
                                 zero=zero,
                                 no_eff=no_eff,
                                 dropout=dropout)
        self.mid_block2 = CondConv1DBlock(dim_in=mid_dim,
                                 dim_out=mid_dim,
                                 cond_dim=cond_dim,
                                 time_dim=time_dim,
                                 adagn=adagn,
                                 zero=zero,
                                 no_eff=no_eff,
                                 dropout=dropout)

        last_dim = mid_dim
        for ind, dim_out in enumerate(reversed(dims[1:])):
            self.ups.append(
                nn.ModuleList([
                    Upsample1d(last_dim, dim_out),
                    CondConv1DBlock(dim_out*2,
                                 dim_out,
                                 cond_dim,
                                 time_dim,
                                 adagn=adagn,
                                 zero=zero,
                                 no_eff=no_eff,
                                 dropout=dropout),
                    CondConv1DBlock(dim_out,
                                 dim_out,
                                 cond_dim,
                                 time_dim,
                                 adagn=adagn,
                                 zero=zero,
                                 no_eff=no_eff,
                                 dropout=dropout),       
                ]))
            last_dim = dim_out
        self.final_conv = nn.Conv1d(dim_out, input_dim, 1)

        if zero:
            nn.init.zeros_(self.final_conv.weight)
            nn.init.zeros_(self.final_conv.bias)
        
    def forward(
        self, 
        x, 
        t,
        cond,
        cond_indices,   
    ):
        temb = self.time_mlp(t)

        h = []
        for block1, block2, downsample in self.downs:
            x = block1(x, temb, cond, cond_indices)
            x = block2(x, temb, cond, cond_indices)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, temb, cond, cond_indices)
        x = self.mid_block2(x, temb, cond, cond_indices)

        for upsample, block1, block2 in self.ups: 
            x = upsample(x) 
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, temb, cond, cond_indices)
            x = block2(x, temb, cond, cond_indices)

        x = self.final_conv(x)
        return x

class T2MUnet(nn.Module):
    """
    Diffuser's style UNET for text-to-motion task.
    """
    def __init__(self,         
                 input_feats,
                 base_dim = 128,
                 dim_mults=(1, 2, 2, 2),
                 dims=None,
                 adagn=True,
                 zero=True,
                 dropout=0.1,
                 no_eff=False,
                 time_dim=512,
                latent_dim=256,
                cond_mask_prob=0.1,
                 clip_dim=512,
                 clip_version='ViT-B/32',
                text_latent_dim=256,
                text_ff_size=2048,
                text_num_heads=4,
                activation="gelu", 
                num_text_layers=4,
                args=None,  # 添加args参数用于传递配置
                ):
        super().__init__()
        self.input_feats = input_feats
        self.dim_mults = dim_mults
        self.base_dim = base_dim
        self.latent_dim = latent_dim
        self.cond_mask_prob = cond_mask_prob
        self.text_latent_dim = text_latent_dim  # 添加这个属性，以便在初始化方法中访问

        print(f'The T2M Unet mask the text prompt by {self.cond_mask_prob} prob. in training')
        
        # text encoder - 使用MoCLIP或原始CLIP
        if MOCLIP_AVAILABLE:
            # 尝试使用MoCLIP的text encoder
            self.use_moclip = True
            self.clip_model = None
            self.clip_tokenizer = None
            self.moclip_model_path = getattr(args, 'moclip_model_path', 
                '/data/wenshuo/project/jhz/StableMoFusion/checkpoints/moclip_training/checkpoint_epoch_20.pt')
            self._init_moclip_text_encoder()
            
            # 根据最终是否使用MoCLIP来设置embed_text
            if self.use_moclip:
                # MoCLIP输出维度是768，需要投影到text_latent_dim
                self.embed_text = nn.Linear(768, text_latent_dim)
            else:
                # 回退到原始CLIP，维度是512
                self.embed_text = nn.Linear(clip_dim, text_latent_dim)
        else:
            # 使用原始CLIP
            self.use_moclip = False
            self.embed_text = nn.Linear(clip_dim, text_latent_dim)
            self.clip_version = clip_version
            self.clip_model = self.load_and_freeze_clip(clip_version)
            
        textTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=text_latent_dim,
            nhead=text_num_heads,
            dim_feedforward=text_ff_size,
            dropout=dropout,
            activation=activation)
        self.textTransEncoder = nn.TransformerEncoder(
            textTransEncoderLayer,
            num_layers=num_text_layers)
        self.text_ln = nn.LayerNorm(text_latent_dim)
        
        # 初始化检索相关的数据
        self.retrieval_texts = None
        self.retrieval_motion_embeddings = None
        self.retrieval_text_embeddings = None
        self._load_retrieval_data()
        
        self.unet = CondUnet1D(
            input_dim=self.input_feats,
            cond_dim=text_latent_dim,
            dim=self.base_dim,
            dim_mults=self.dim_mults,
            adagn=adagn,
            zero=zero,
            dropout=dropout,
            no_eff=no_eff,
            dims=dims,
            time_dim=time_dim
        )

    def _init_moclip_text_encoder(self):
        """
        初始化MoCLIP的text encoder
        """
        try:
            # 初始化MoCLIP模型
            _init_clip_motion_model(self.moclip_model_path)
            
            # 获取模型组件
            self.clip_model = GLOBAL_CACHE["clip_model"]
            self.clip_tokenizer = GLOBAL_CACHE["clip_tokenizer"]
            
            # 确保模型被冻结
            if self.clip_model is not None:
                self.clip_model.eval()
                for p in self.clip_model.parameters():
                    p.requires_grad = False
            
            print(f"成功初始化MoCLIP text encoder from {self.moclip_model_path}")
            
        except Exception as e:
            print(f"Failed to initialize MoCLIP text encoder: {e}")
            print("Falling back to original CLIP encoder")
            self.use_moclip = False
            # 设置原始CLIP的版本并加载
            self.clip_version = 'ViT-B/32'
            self.clip_model = self.load_and_freeze_clip(self.clip_version)

    def _load_retrieval_data(self):
        """
        加载检索用的文本和motion编码数据
        """
        import os
        import numpy as np
        
        # 检索数据文件路径
        samples_file = "/data/wenshuo/project/jhz/StableMoFusion/new_model_results/fair_comparison/chicken_samples.txt"
        motion_embeddings_file = "/data/wenshuo/project/jhz/StableMoFusion/new_model_results/fair_comparison/chicken_new_motion_embeddings.npy"
        
        try:
            # 加载样本文本
            if os.path.exists(samples_file):
                with open(samples_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    self.retrieval_texts = []
                    for line in lines:
                        line = line.strip()
                        if ': ' in line:
                            # 格式: "索引: 文本内容"
                            text = line.split(': ', 1)[1]
                            self.retrieval_texts.append(text)
                        else:
                            self.retrieval_texts.append(line)
                
                print(f"Loaded {len(self.retrieval_texts)} retrieval texts")
            
            # 加载motion编码
            if os.path.exists(motion_embeddings_file):
                self.retrieval_motion_embeddings = np.load(motion_embeddings_file)
                print(f"Loaded retrieval motion embeddings: {self.retrieval_motion_embeddings.shape}")
            
            # 预编码检索文本（延迟加载，在第一次forward时进行）
            self.retrieval_text_embeddings = None
            
        except Exception as e:
            print(f"Warning: Failed to load retrieval data: {e}")
            self.retrieval_texts = None
            self.retrieval_motion_embeddings = None
            self.retrieval_text_embeddings = None

    def _encode_retrieval_texts(self, device):
        """
        对检索文本进行编码（延迟加载）
        """
        if self.retrieval_texts is None or len(self.retrieval_texts) == 0:
            return
        
        try:
            print("Encoding retrieval texts...")
            
            # 使用当前模型对检索文本进行编码
            with torch.no_grad():
                # 批量处理文本编码以避免内存问题
                batch_size = 32
                text_embeddings_list = []
                
                for i in range(0, len(self.retrieval_texts), batch_size):
                    batch_texts = self.retrieval_texts[i:i+batch_size]
                    
                    # 使用 encode_text 方法对文本进行编码
                    batch_encoded = self.encode_text(batch_texts, device)  # [batch, seq_len, text_dim]
                    
                    # 对序列维度进行平均池化，得到固定大小的文本表示
                    batch_encoded_pooled = batch_encoded.mean(dim=1)  # [batch, text_dim]
                    text_embeddings_list.append(batch_encoded_pooled.cpu().numpy())
                
                # 合并所有批次的编码
                self.retrieval_text_embeddings = np.concatenate(text_embeddings_list, axis=0)
                print(f"Encoded retrieval text embeddings: {self.retrieval_text_embeddings.shape}")
                
        except Exception as e:
            print(f"Warning: Failed to encode retrieval texts: {e}")
            self.retrieval_text_embeddings = None

    def _retrieve_motion_embedding(self, enc_text, device):
        """
        根据输入文本编码检索最相似的motion编码
        
        Args:
            enc_text: [bs, seq_len, text_dim] 输入文本的编码
            device: 设备
            
        Returns:
            retrieved_motion_emb: [bs, motion_emb_dim] 检索到的motion编码
        """
        if (self.retrieval_texts is None or 
            self.retrieval_motion_embeddings is None or 
            len(self.retrieval_texts) == 0):
            # 如果没有检索数据，返回零向量
            motion_emb_dim = 768  # 假设motion编码维度是768
            return torch.zeros(enc_text.shape[0], motion_emb_dim, device=device)
        
        # 如果还没有编码检索文本，现在进行编码
        if self.retrieval_text_embeddings is None:
            self._encode_retrieval_texts(device)
        
        if self.retrieval_text_embeddings is None:
            # 编码失败，返回零向量
            motion_emb_dim = 768
            return torch.zeros(enc_text.shape[0], motion_emb_dim, device=device)
        
        try:
            # 对输入文本编码进行池化，得到固定大小的表示
            input_text_pooled = enc_text.mean(dim=1)  # [bs, text_dim]
            
            # 转换为numpy进行相似度计算
            input_text_np = input_text_pooled.detach().cpu().numpy()
            
            # 归一化向量以计算余弦相似度
            input_text_norm = input_text_np / (np.linalg.norm(input_text_np, axis=1, keepdims=True) + 1e-8)
            retrieval_text_norm = self.retrieval_text_embeddings / (np.linalg.norm(self.retrieval_text_embeddings, axis=1, keepdims=True) + 1e-8)
            
            # 计算余弦相似度
            similarities = np.dot(input_text_norm, retrieval_text_norm.T)  # [bs, num_retrieval_texts]
            
            # 找到最相似的文本索引
            best_indices = np.argmax(similarities, axis=1)  # [bs]
            
            # 获取对应的motion编码
            retrieved_motion_embeddings = self.retrieval_motion_embeddings[best_indices]  # [bs, motion_emb_dim]
            
            # 转换为tensor并移到正确设备
            retrieved_motion_emb = torch.from_numpy(retrieved_motion_embeddings).float().to(device)
            
            # 打印检索信息（仅在推理时）
            if not self.training:
                for i, (best_idx, sim) in enumerate(zip(best_indices, similarities[np.arange(len(best_indices)), best_indices])):
                    if i < 3:  # 只打印前3个
                        print(f"Retrieved text {i}: '{self.retrieval_texts[best_idx]}' (similarity: {sim:.4f})")
            
            return retrieved_motion_emb
            
        except Exception as e:
            print(f"Warning: Failed to retrieve motion embedding: {e}")
            # 返回零向量
            motion_emb_dim = self.retrieval_motion_embeddings.shape[1] if self.retrieval_motion_embeddings is not None else 768
            return torch.zeros(enc_text.shape[0], motion_emb_dim, device=device)

    def encode_text(self, raw_text, device):
        with torch.no_grad():
            if self.use_moclip and self.clip_tokenizer is not None:
                # 使用MoCLIP的text encoder
                texts_lower = [text.lower() for text in raw_text]
                text_encodings = self.clip_tokenizer(
                    texts_lower,
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt"
                )
                input_ids = text_encodings["input_ids"].to(device)
                attention_mask = text_encodings["attention_mask"].to(device)
                
                # 使用 MoCLIP 的 text_model 获取 token-level hidden states，而不是重复 pooled 向量
                # 这样可以保留 77 个 token 位置的差异化信息，避免了之前复制同一向量的问题
                text_outputs = self.clip_model.text_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                # text_outputs.last_hidden_state: [batch_size, seq_len, 768]
                x = text_outputs.last_hidden_state.permute(1, 0, 2)  # [seq_len, batch_size, 768]
                
            else:
                # 使用原始CLIP encoder
                texts = clip.tokenize(raw_text, truncate=True).to(device)
                x = self.clip_model.token_embedding(texts).type(self.clip_model.dtype)
                x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.clip_model.transformer(x)
                x = self.clip_model.ln_final(x).type(self.clip_model.dtype) #[len, batch_size, 512]

        x = self.embed_text(x) #[len, batch_size, 256]
        x = self.textTransEncoder(x)
        x = self.text_ln(x)
        # T, B, D -> B, T, D
        xf_out = x.permute(1, 0, 2)
        return xf_out
    
    def load_and_freeze_clip(self, clip_version):
        clip_model, _ = clip.load(  # clip_model.dtype=float32
            clip_version, device='cpu',
            jit=False)  # Must set jit=False for training

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model
    
    def mask_cond(self, bs, force_mask=False):
        '''
            mask motion condition , return contitional motion index in the batch
        '''
        if force_mask:
            cond_indices = torch.empty(0)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, ) * self.cond_mask_prob)  # 1-> use null_cond, 0-> use real cond
            mask =  (1. - mask)
            cond_indices = torch.nonzero(mask).squeeze(-1)
        else:
            cond_indices = torch.arange(bs)
        
        return cond_indices
        
    def forward(
            self, 
            x, 
            timesteps, 
            text=None, 
            uncond=False,
            enc_text=None,
        ):
            """
            Args:
                x: [batch_size, nframes, nfeats],
                timesteps: [batch_size] (int)
                text: list (batch_size length) of strings with input text prompts
                uncond: whethere using text condition

            Returns: [batch_size, seq_length, nfeats]
            """
            B, T, _ = x.shape
            x = x.transpose(1, 2) # [bs, nfeats, nframes]

            if enc_text is None:
                enc_text = self.encode_text(text, x.device) # [bs, seqlen, text_dim]

            cond_indices = self.mask_cond(x.shape[0], force_mask=uncond)

            # NOTE: need to pad to be the multiplier of 8 for the unet 
            PADDING_NEEEDED = (16 - (T % 16)) % 16

            padding = (0, PADDING_NEEEDED)
            x = F.pad(x, padding, value=0)

            x = self.unet(
                x,
                t=timesteps,
                cond=enc_text,
                cond_indices = cond_indices,
            )  # [bs, nfeats,, nframes]

            x = x[:, :, :T].transpose(1, 2) # [bs, nframes, nfeats,]

            return x
        
    def forward_with_cfg(
            self, 
            x, 
            timesteps, 
            text=None, 
            enc_text=None,
            cfg_scale=2.5
        ):
            """
            Args:
                x: [batch_size, nframes, nfeats],
                timesteps: [batch_size] (int)
                text: list (batch_size length) of strings with input text prompts

            Returns: [batch_size, max_frames, nfeats]
            """

            B, T, _ =x.shape
            x = x.transpose(1, 2) # [bs, nfeats, nframes]
            if enc_text is None:
                enc_text = self.encode_text(text, x.device) # [bs, seqlen, text_dim]

            cond_indices = self.mask_cond(B)

            # NOTE: need to pad to be the multiplier of 8 for the unet
            PADDING_NEEEDED = (16 - (T % 16)) % 16
            
            padding = (0, PADDING_NEEEDED)
            x = F.pad(x, padding, value=0)

            combined_x = torch.cat([x, x], dim=0)
            combined_t =torch.cat([timesteps, timesteps], dim=0)
            out = self.unet(
                x=combined_x,
                t=combined_t,
                cond=enc_text,
                cond_indices = cond_indices,        
            )  # [bs, nfeats, nframes]

            out = out[:, :, :T].transpose(1, 2) # [bs, nframes, nfeats,]

            out_cond, out_uncond = torch.split(out, len(out) // 2, dim=0)

            return out_uncond + (cfg_scale * (out_cond - out_uncond))



if __name__ == "__main__":

    device = 'cuda:0'
    n_feats = 263
    num_frames = 196
    text_latent_dim = 256
    dim_mults = [2,2,2,2]
    base_dim= 512
    # 创建一个简单的args对象用于测试
    from argparse import Namespace
    args = Namespace()
    args.moclip_model_path = '/data/wenshuo/project/jhz/StableMoFusion/checkpoints/moclip_training/checkpoint_epoch_20.pt'
    
    model =  T2MUnet(
        input_feats = n_feats,
        text_latent_dim = text_latent_dim,
        base_dim= base_dim,
        dim_mults = dim_mults,
        adagn = True,
        zero = True,
        dropout=0.1,
        no_eff=False,
        cond_mask_prob=0.1,
        args=args
    )

    model = model.to(device)
    dtype = torch.float32
    bs = 1
    x = torch.rand((bs, 196, 263),dtype=dtype ).to(device)
    timesteps = torch.randint(low=0, high=1000, size=(bs,)).to(device)
    y = ['A man jumps to his left.' for i in range(bs)]
    length = torch.randint(low=20, high=196, size=(bs,)).to(device)

    out = model(x, timesteps, text=y)
    print(out.shape)
    model.eval()
    out = model.forward_with_cfg(x, timesteps, text=y)
    print(out.shape)
