import clip
import math

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from einops.layers.torch import Rearrange
from einops import rearrange
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from models.BERT.BERT_encoder import load_bert
from models.CLIPS.Moclip.moclip import EvalWarperMoClip
from models.CLIPS.LongCLIP.model import longclip
import os
import matplotlib.pyplot as plt
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
    
    def forward_w_weight(self, input_tensor, condition_tensor):
        """
        与 forward 类似，但额外返回计算得到的注意力权重
        input_tensor: [B, T, D]
        condition_tensor: [B, N, L]
        返回:
            y: [B, T, D] 最终输出
            attention: [B, H, HD, HD] 计算后的注意力权重（经过 dropout）
        """
        B, T, D = input_tensor.shape
        N = condition_tensor.shape[1]
        H = self.num_head
        # 计算 query、key、value
        query = self.query(self.norm(input_tensor))
        key = self.key(self.text_norm(condition_tensor))
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, N, H, -1), dim=1)
        value = self.value(self.text_norm(condition_tensor)).view(B, N, H, -1)
        # 计算注意力权重
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        attention = self.dropout(attention)
        # 根据 attention 计算最终输出
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        return y, attention


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
class AdaLayerNorm(nn.Module):
    def __init__(self, embedding_dim: int, time_embedding_dim= None):
        super().__init__()

        if time_embedding_dim is None:
            time_embedding_dim = embedding_dim

        self.silu = nn.SiLU()
        self.linear = nn.Linear(time_embedding_dim, 2 * embedding_dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self, x, timestep_embedding
    ):
        emb = self.linear(self.silu(timestep_embedding))
        shift, scale = emb.view(len(x), 1, -1).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x


class SquaredReLU(nn.Module):
    def forward(self, x):
        return torch.square(torch.relu(x))

from collections import OrderedDict
class PerceiverAttentionBlock(nn.Module):
    def __init__(
        self, d_model, n_heads, time_embedding_dim = None,is_abstractor=True, enable_noise=True
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.is_abstractor = is_abstractor
        self.enable_noise = enable_noise
        if is_abstractor:
            pass
        else:
            self.mlp = nn.Sequential(
                OrderedDict(
                    [
                        ("c_fc", nn.Linear(d_model, d_model * 4)),
                        ("sq_relu", SquaredReLU()),
                        ("c_proj", nn.Linear(d_model * 4, d_model)),
                    ]
                )
            )

        self.ln_1 = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_2 = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_ff = AdaLayerNorm(d_model, time_embedding_dim)

    def attention(self, q, kv):
        attn_output, attn_output_weights = self.attn(q, kv, kv, need_weights=False)
        return attn_output

    def forward(
        self,
        x,
        latents,
        timestep_embedding = None,
    ):
        normed_latents = self.ln_1(latents, timestep_embedding)
        #print("q:")
        #print(normed_latents.shape)
        if self.enable_noise:
            k = torch.randn_like(normed_latents)
        else:
            k = normed_latents

        kv=torch.cat([k, self.ln_2(x, timestep_embedding)], dim=1)
        #print("kv:")
        #print(kv.shape)
        latents = latents + self.attention(
            q=normed_latents,
            kv=kv,
        )
        #print(latents.shape)
        if self.is_abstractor:
            return latents
        else:
            latents = latents + self.mlp(self.ln_ff(latents, timestep_embedding))
        return latents


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        width: int = 768,
        layers: int = 6,
        heads: int = 8,
        num_latents: int = 64,
        output_dim=None,
        input_dim=None,
        time_embedding_dim = None,
        is_abstractor=True,
        enable_noise = False
    ):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.latents = nn.Parameter(width**-0.5 * torch.randn(num_latents, width))
        self.time_aware_linear = nn.Linear(
            time_embedding_dim or width, width, bias=True
        )

        if self.input_dim is not None:
            self.proj_in = nn.Linear(input_dim, width)

        self.perceiver_blocks = nn.Sequential(
            *[
                PerceiverAttentionBlock(
                    width, heads, time_embedding_dim=time_embedding_dim,is_abstractor=is_abstractor,enable_noise=False
                )
                for _ in range(layers)
            ]
        )

        if self.output_dim is not None:
            self.proj_out = nn.Sequential(
                nn.Linear(width, output_dim), nn.LayerNorm(output_dim)
            )

    def forward(self, x, timestep_embedding = None):
        learnable_latents = self.latents.unsqueeze(dim=0).repeat(len(x), 1, 1)
        latents = learnable_latents + self.time_aware_linear(
            torch.nn.functional.silu(timestep_embedding)
        )
        if self.input_dim is not None:
            x = self.proj_in(x)
        
        for p_block in self.perceiver_blocks:
            latents = p_block(x, latents, timestep_embedding=timestep_embedding)

        if self.output_dim is not None:
            latents = self.proj_out(latents)

        return latents

class T5TextEmbedder(nn.Module):
    def __init__(self, pretrained_path="./T5", max_length=77):
        super().__init__()
        self.model = T5EncoderModel.from_pretrained(pretrained_path)
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_path)
        self.max_length = max_length

    def forward(
        self, caption, text_input_ids=None, attention_mask=None, max_length=None
    ):
        if max_length is None:
            max_length = self.max_length

        if text_input_ids is None or attention_mask is None:
            if max_length is not None:
                text_inputs = self.tokenizer(
                    caption,
                    return_tensors="pt",
                    add_special_tokens=True,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                )
            else:
                text_inputs = self.tokenizer(
                    caption,
                    padding=True,  # 启用填充
                    truncation=True,  # 启用截断
                    max_length=self.max_length,  # 设置最大长度
                    return_tensors="pt"  # 返回 PyTorch 张量
                )
            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask
        text_input_ids = text_input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        outputs = self.model(text_input_ids, attention_mask=attention_mask)

        embeddings = outputs.last_hidden_state
        return embeddings
class ELLA(nn.Module):
    def __init__(
        self,
        time_channel=320,
        time_embed_dim=512,
        act_fn= "silu",
        out_dim= None,
        width=256,
        layers=6,
        heads=8,
        num_latents=77,
        input_dim=2048,
        is_abstractor=True,
        enable_noise=False
    ):
        super().__init__()

        self.position = Timesteps(
            time_channel, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedding = TimestepEmbedding(
            in_channels=time_channel,
            time_embed_dim=time_embed_dim,
            act_fn=act_fn,
            out_dim=out_dim,
        )

        self.connector = PerceiverResampler(
            width=width,
            layers=layers,
            heads=heads,
            num_latents=num_latents,
            input_dim=input_dim,
            time_embedding_dim=time_embed_dim,
            is_abstractor=is_abstractor,
            enable_noise=enable_noise,
            output_dim=out_dim
        )

    def forward(self, text_encode_features, timesteps):
        device = text_encode_features.device
        dtype = text_encode_features.dtype

        ori_time_feature = self.position(timesteps.view(-1)).to(device, dtype=dtype)
        ori_time_feature = (
            ori_time_feature.unsqueeze(dim=1)
            if ori_time_feature.ndim == 2
            else ori_time_feature
        )
        ori_time_feature = ori_time_feature.expand(len(text_encode_features), -1, -1)
        # ori_time_feature = ori_time_feature.half()
        time_embedding = self.time_embedding(ori_time_feature)
        # text_encode_features = text_encode_features.half()
        encoder_hidden_states = self.connector(
            text_encode_features, timestep_embedding=time_embedding
        )

        return encoder_hidden_states

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
                t5_dim=2048,
                clip_dim = 512,
                text_latent_dim=1024,
                text_ff_size=2048,
                text_num_heads=4,
                activation="gelu", 
                num_text_layers=4,
                use_cfgpp =True,
                clip_version= 'ViT-B/32' ,
                text_encoder_type = "t5",
                device = 'cuda',
                enable_cfg_scheduler = False,
                cfg_scheduler_type = 'moclip',
                is_abstractor = True,
                disable_set = False,
                enable_noise = False,
                                ):
        super().__init__()
        self.input_feats = input_feats
        self.dim_mults = dim_mults
        self.base_dim = base_dim
        self.latent_dim = latent_dim
        self.cond_mask_prob = cond_mask_prob
        self.use_cfgpp = use_cfgpp
        self.text_encoder_type = text_encoder_type
        self.clip_version = clip_version
        self.enable_cfg_scheduler = enable_cfg_scheduler
        self.cfg_scheduler_type = cfg_scheduler_type
        self.device = device
        self.is_abstractor = is_abstractor
        self.disable_set = disable_set
        self.sim = []
        self.enable_noise = enable_noise
        '''
        debug and vis
        '''
        self.token_attn_history_dict = {}   # 每层记录 token 部分，shape: [77]
        self.timestep_history_dict = {}       # 每层对应的 timestep 值
        self.noise_follow_history_dict = {}   # 每层对应的 noise follow 百分比
        self.token_follow_history_dict = {}

        if self.is_abstractor:
            print("use abstractor")
        print(f'The T2M Unet mask the text prompt by {self.cond_mask_prob} prob. in training')
      
        # self.clip_version = clip_version
        # self.clip_model = self.load_and_freeze_clip(clip_version)
        if self.text_encoder_type == "clip":
            print('Loading CLIP...')
            self.embed_text = nn.Linear(clip_dim, text_latent_dim)
            if not self.disable_set:
                print("build ella model")
                self.ella_model= ELLA(input_dim=256,is_abstractor=self.is_abstractor,enable_noise=enable_noise)
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
            self.encode_text = self.encode_text_clip
        elif self.text_encoder_type == 'bert':
            print("Loading BERT...")
            #raise ValueError('not support now')
            bert_model_path = 'distilbert/distilbert-base-uncased'
            self.clip_model = load_bert(bert_model_path)  # Sorry for that, the naming is for backward compatibility
            self.encode_text = self.encode_text_bert
            self.clip_dim = 768
            if not self.disable_set:
                print("build ella model")
                self.ella_model= ELLA(input_dim=768,is_abstractor=self.is_abstractor,enable_noise=enable_noise)
            self.embed_text = nn.Linear(self.clip_dim, text_latent_dim)

        elif self.text_encoder_type == 't5':
            print("loading T5")
            #self.embed_text = nn.Linear(t5_dim, text_latent_dim)
            self.t5_linear = nn.Linear(t5_dim, text_latent_dim)
            if not self.disable_set:
                print("build ella model")
                self.ella_model= ELLA(is_abstractor=self.is_abstractor,enable_noise=enable_noise)
            self.t5_encoder = self.load_t5(self.device)
            self.encode_text = self.encode_text_t5
        elif self.text_encoder_type == 'longclip':
            model, preprocess = longclip.load("/data/kuimou/SET/models/CLIPS/LongCLIP/checkpoints/longclip-B.pt", device=self.device)
            for param in model.parameters():
                param.requires_grad = False
            if not self.disable_set:
                print("build ella model")
                self.ella_model= ELLA(input_dim=512,is_abstractor=self.is_abstractorm,enable_noise=enable_noise)
            self.clip_model = model
            self.encode_text = self.encode_text_long_clip
            self.embed_text = nn.Linear(512,text_latent_dim)
        elif self.text_encoder_type == 'moclip':
            self.clip_model = EvalWarperMoClip(model_path="/data/kuimou/SET/checkpoints/clip_motion_align_epoch_21.pt")
            if not self.disable_set:
                print("build ella model")
                self.ella_model= ELLA(input_dim=768,is_abstractor=self.is_abstractor,enable_noise=enable_noise)
            self.embed_text = nn.Linear(768,text_latent_dim)
            self.encode_text = self.encode_text_moclip
        else:
            raise ValueError('only support [clip,long_clip,bert,t5] text encoders') 
        
        #self.cfg_clip = EvalWarperMoClip(model_path="/data/kuimou/SET/checkpoints/clip_motion_align_epoch_21.pt",device=self.device)

        self.text_ln = nn.LayerNorm(text_latent_dim)
        
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



    # def encode_text(self, raw_text, device):
    #     with torch.no_grad():
    #         texts = clip.tokenize(raw_text, truncate=True).to(
    #                 device
    #             )  # [bs, context_length] # if n_tokens > 77 -> will truncate
    #         x = self.clip_model.token_embedding(texts).type(self.clip_model.dtype)  # [batch_size, n_ctx, d_model]
    #         x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
    #         x = x.permute(1, 0, 2)  # NLD -> LND
    #         x = self.clip_model.transformer(x)
    #         x = self.clip_model.ln_final(x).type(self.clip_model.dtype) #[len, batch_size, 512]

    #     x = self.embed_text(x) #[len, batch_size, 256]
    #     x = self.textTransEncoder(x)
    #     x = self.text_ln(x)
    #     # T, B, D -> B, T, D
    #     xf_out = x.permute(1, 0, 2)
    #     return xf_out
    
    def load_t5(self,device):
        t5_encoder = T5TextEmbedder().to(device, dtype=torch.float32)
        for param in t5_encoder.parameters():
            param.requires_grad = False
        return t5_encoder
    
    def encode_text_t5(self,prompt):
        xf_out = self.t5_encoder(prompt, max_length=None).to(self.device, torch.float32)
        if self.disable_set:
            xf_out = self.t5_linear(xf_out)
        return xf_out
    
    def encode_text_clip(self,prompt):
        device = self.device
        with torch.no_grad():
            texts = clip.tokenize(prompt, truncate=True).to(
                    device
                )  # [bs, context_length] # if n_tokens > 77 -> will truncate
            x = self.clip_model.token_embedding(texts).type(self.clip_model.dtype)  # [batch_size, n_ctx, d_model]
            x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.clip_model.transformer(x)
            x = self.clip_model.ln_final(x).type(self.clip_model.dtype) #[len, batch_size, 512]
        #print(x.shape)
        x = self.embed_text(x) #[len, batch_size, 256]
        x = self.textTransEncoder(x)
        x = self.text_ln(x)
        # T, B, D -> B, T, D
        xf_out = x.permute(1, 0, 2)
        return xf_out
    def encode_text_moclip(self, prompt):
        '''
        device = next(self.parameters()).device
        '''
        device =  self.device

        #text = clip.tokenize(raw_text, truncate=True).to(device)
        #feat_clip_text = self.clip_model.encode_text(text).float()


        batch_size = len(prompt)
        dummy_motions = torch.zeros(
            (batch_size, 196, 263))
        with torch.no_grad():
            text_emb, _ = self.clip_model.get_co_embeddings(
                captions=prompt,
                motions=dummy_motions)

        text_emb = text_emb.to(device).float()
        if self.disable_set:
            text_emb = self.embed_text(text_emb)
        text_emb = text_emb.unsqueeze(1)
        return text_emb
    def encode_text_long_clip(self,prompt):
        device =  self.device
        text = longclip.tokenize(prompt).to(device)
        text_features = self.clip_model.encode_text(text).float()
        #print(text_features.shape)
        if self.disable_set:
            text_features = self.embed_text(text_features)
        text_features = text_features.unsqueeze(1)
        #print(text_features.shape)
        return text_features

    def encode_text_bert(self,prompt):
        enc_text, _ = self.clip_model(prompt)  # self.clip_model.get_last_hidden_state(raw_text, return_mask=True)  # mask: False means no token there
        if self.disable_set:
            enc_text = self.embed_text(enc_text)
       #print(enc_text.shape)
       #enc_text = enc_text.permute(1, 0, 2)
        #mask = ~mask  # mask: True means no token there, we invert since the meaning of mask for transformer is inverted  https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        return enc_text


    def load_and_freeze_clip_f16(self, clip_version):
        clip_model, _ = clip.load(  # clip_model.dtype=float32
            clip_version, device=self.device,
            jit=False)  # Must set jit=False for training

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def load_and_freeze_clip(self, clip_version):
        clip_model, _ = clip.load(  # clip_model.dtype=float32
            clip_version, device='cpu',
            jit=False)  # Must set jit=False for training
        clip_model = clip_model.to(self.device)
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
            enc_text = self.encode_text(text) # [bs, seqlen, text_dim]
        
        #print(enc_text.shape)
        if not self.disable_set:
            #print(enc_text.shape)
            enc_text = self.ella_model(enc_text,timesteps)
        cond_indices = self.mask_cond(x.shape[0], force_mask=uncond)
        #print(enc_text.shape)
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
        opt=None
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
            enc_text = self.encode_text(text) # [bs, seqlen, text_dim]
        if not self.disable_set:
            enc_text_raw = enc_text
            #print(enc_text.shape)
            enc_text = self.ella_model(enc_text,timesteps)
            #print(enc_text.shape)
        cond_indices = self.mask_cond(B)

        # NOTE: need to pad to be the multiplier of 8 for the unet
        PADDING_NEEEDED = (16 - (T % 16)) % 16
        
        padding = (0, PADDING_NEEEDED)
        x = F.pad(x, padding, value=0)
        '''
        if int(timesteps[0]) < 500:
            cond_indices = self.mask_cond(B, force_mask=True)
            x = self.unet(
                x,
                t=timesteps,
                cond=enc_text,
                cond_indices = cond_indices,
            )  # [bs, nfeats,, nframes]

            x = x[:, :, :T].transpose(1, 2)
            return x
        '''
        s_m=opt.cfg_scale_adj_max
        s_s=opt.cfg_scale_adj_min
        cfg_scale = s_s+ 0.5*(1+math.cos((1-(int(timesteps[0])/1000))*math.pi*1.5))*(s_m-s_s)
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
        timesteps = timesteps.reshape(B,1) #[BS,1]
     
        cfg_scale=2.5
       
        return out_uncond + (cfg_scale * (out_cond - out_uncond))

    '''
    debug use
    '''
    def forward_with_cfg_draw(
        self, 
        x, 
        timesteps, 
        text=None, 
        enc_text=None,
        cfg_scale_adj_max=3.5,
        cfg_scale_adj_min=0,
        cfg_scale_adj_alpha = 2.5,
        cfg_tracker=None,
        timestep_tracker = None,
        opt=None
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
            enc_text = self.encode_text(text) # [bs, seqlen, text_dim]
        if not self.disable_set:
            enc_text_raw = enc_text
            enc_text = self.forward_with_attention_vis(enc_text,timesteps,caption=text)
        cond_indices = self.mask_cond(B)

        # NOTE: need to pad to be the multiplier of 8 for the unet
        PADDING_NEEEDED = (16 - (T % 16)) % 16
        
        padding = (0, PADDING_NEEEDED)
        x = F.pad(x, padding, value=0)

        # if int(timesteps[0]) < 100:
        #         x = self.unet(
        #             x,
        #             t=timesteps,
        #             cond=enc_text,
        #             cond_indices = cond_indices,
        #         )  # [bs, nfeats,, nframes]

        #         x = x[:, :, :T].transpose(1, 2)
        #         return x
        
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
        timesteps = timesteps.reshape(B,1) #[BS,1]
        cfg_scale = 3.5

        return out_uncond + (cfg_scale * (out_cond - out_uncond))
    def forward_with_unet_attention(self, x, timesteps, text=None, enc_text=None, opt=None):
        """
        劫持 UNet 内各个 cross_attn 模块，记录每次前向传播时计算得到的注意力数值，
        并以字典的形式存储（不进行绘图）。
        
        收集的数据格式存放在 self.unet_attn_history 中，其结构为：
        { module_id: { timestep (int): [flat_attn_vector, flat_attn_vector, ...], ... }, ... }
        
        注意：此方法仅收集数据，不进行绘图，绘图将在 DiffusePipeline 的其他步骤中调用专门的绘制函数。
        """
        import numpy as np
        import torch.nn.functional as F

        # 确保历史记录字典存在（不同 batch 调用中可累积）
        if not hasattr(self, "unet_attn_history"):
            self.unet_attn_history = {}  # 结构： {module_id: {timestep: [flat_attn, ...], ...}, ...}

        # 如果 enc_text 未计算，则调用对应文本编码接口
        if enc_text is None:
            enc_text = self.encode_text(text)
        
        if not self.disable_set:
            enc_text = self.ella_model(enc_text, timesteps)

        cond_indices = self.mask_cond(x.shape[0])
        B, T, _ = x.shape
        # 将 x 调整为 UNet 所需形状 [B, nfeats, T]，并进行必要的 padding（保证 T 为 16 的倍数）
        x = x.transpose(1, 2)  # [B, nfeats, T]
        PADDING_NEEDED = (16 - (T % 16)) % 16
        x = F.pad(x, (0, PADDING_NEEDED), value=0)

        # 收集 UNet 中所有包含 cross-attn 的模块（分别在 downs、mid、ups 内）
        cross_attn_modules = []
        for block in self.unet.downs:
            # block: [CondConv1DBlock, CondConv1DBlock, Downsample1d]
            for conv_block in block[:2]:
                cross_attn_modules.append(conv_block.cross_attn)
        cross_attn_modules.append(self.unet.mid_block1.cross_attn)
        cross_attn_modules.append(self.unet.mid_block2.cross_attn)
        for block in self.unet.ups:
            # block: [Upsample1d, CondConv1DBlock, CondConv1DBlock]
            for conv_block in block[1:]:
                cross_attn_modules.append(conv_block.cross_attn)

        # 保存每个 cross_attn 模块原始的 forward 方法以便后面恢复
        orig_forwards = {}

        # 劫持每个模块的内部 cross_attention 的 forward 方法（要求支持 forward_w_weight）
        for module in cross_attn_modules:
            if hasattr(module, "cross_attention") and hasattr(module.cross_attention, "forward_w_weight"):
                module_id = id(module)
                orig_forwards[module_id] = module.cross_attention.forward

                def new_forward(input_tensor, condition_tensor,
                                orig_forward=module.cross_attention.forward_w_weight, module_id=module_id):
                    # 调用 forward_w_weight 获取 (output, attn)
                    out, attn = orig_forward(input_tensor, condition_tensor)
                    # 假设 attn shape 为 [B, H, D1, D2]，在 head 维度上求均值后得到 [B, D1, D2]
                    attn_avg = attn.mean(dim=1)
                    # 取 batch 中第一个样本: [D1, D2]
                    attn_sample = attn_avg[0]
                    # 展平为一维向量
                    flat_attn = attn_sample.flatten().detach().cpu().numpy()
                    # 强制将 timestep 转为整数
                    current_timestep = int(timesteps[0].item() if torch.is_tensor(timesteps[0]) else timesteps[0])
                    # 保存记录到字典中
                    if module_id not in self.unet_attn_history:
                        self.unet_attn_history[module_id] = {}
                    if current_timestep not in self.unet_attn_history[module_id]:
                        self.unet_attn_history[module_id][current_timestep] = []
                    self.unet_attn_history[module_id][current_timestep].append(flat_attn)
                    return out

                module.cross_attention.forward = new_forward
            else:
                print(f"[forward_with_unet_attention] 模块 {module} 不支持 forward_w_weight，跳过。")

        # 执行 UNet 前向传播
        combined_x = torch.cat([x, x], dim=0)
        combined_t =torch.cat([timesteps, timesteps], dim=0)
        out_unet = self.unet(combined_x, t=combined_t, cond=enc_text, cond_indices=cond_indices)
        out_unet = out_unet[:, :, :T].transpose(1, 2)  # 恢复成 [B, T, nfeats]

        out_cond, out_uncond = torch.split(out_unet, len(out_unet) // 2, dim=0)
        cfg_scale = 3.5

        # 恢复所有被劫持模块的原始 forward 方法
        for module in cross_attn_modules:
            module_id = id(module)
            if module_id in orig_forwards:
                module.cross_attention.forward = orig_forwards[module_id]

        return out_uncond + (cfg_scale * (out_cond - out_uncond))
    
    def forward_with_unet_attnion_vis(self, x, timesteps, text=None, enc_text=None, opt=None, vis_dir="./unet_attn_vis"):
        """
        劫持 UNet 内的 cross_attn 模块，记录每一次前向传播时跨注意力的注意力数值，
        保存历史记录（每次记录为(timestep, flat_attn_vector)），
        当 timestep==0 时（表明扩散过程结束），对每个模块的历史记录计算直方图，
        使用热力图展示不同时间步下注意力数值的分布情况：
          - 横轴为注意力值的区间(bin)，
          - 纵轴为不同时间步，
          - 单元格数值表示该时间步下落在对应注意力值区间内的频数（或归一化频率）。
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 如果历史记录字典还不存在，则初始化
        if not hasattr(self, "unet_attn_history"):
            self.unet_attn_history = {}  # key: 模块 id, value: list of (timestep, flat_attn_vector)
        
        # 如 enc_text 未计算，则调用文本编码接口（同时通过 ELLA 做时间编码等处理）
        if enc_text is None:
            enc_text = self.encode_text(text)
        if not self.disable_set:
            enc_text = self.ella_model(enc_text, timesteps)
        
        cond_indices = self.mask_cond(x.shape[0])
        B, T, _ = x.shape
        # UNet 要求输入为 [B, nfeats, nframes]，同时帧数需要 pad 到16的倍数
        x = x.transpose(1, 2)  # [B, nfeats, T]
        PADDING_NEEDED = (16 - (T % 16)) % 16
        x = F.pad(x, (0, PADDING_NEEDED), value=0)
        
        # 收集 UNet 内所有含 cross-attn 的模块
        cross_attn_modules = []
        # downs 内部两个 CondConv1DBlock
        for block in self.unet.downs:
            for conv_block in block[:2]:
                cross_attn_modules.append(conv_block.cross_attn)
        # mid block 内部两个 CondConv1DBlock
        cross_attn_modules.append(self.unet.mid_block1.cross_attn)
        cross_attn_modules.append(self.unet.mid_block2.cross_attn)
        # ups 内部两个 CondConv1DBlock
        for block in self.unet.ups:
            for conv_block in block[1:]:
                cross_attn_modules.append(conv_block.cross_attn)
        
        # 保存每个模块原始 forward 方法以便恢复
        orig_forwards = {}
        
        # 劫持每个模块内部 cross_attention 的 forward 方法（要求支持 forward_w_weight）
        for module in cross_attn_modules:
            if hasattr(module, "cross_attention") and hasattr(module.cross_attention, "forward_w_weight"):
                module_id = id(module)
                orig_forwards[module_id] = module.cross_attention.forward
                
                def new_forward(input_tensor, condition_tensor, 
                                orig_forward=module.cross_attention.forward_w_weight, module_id=module_id):
                    # 通过 forward_w_weight 获得 (output, attn)
                    out, attn = orig_forward(input_tensor, condition_tensor)
                    # 假设 attn 的 shape 为 [B, H, D1, D2]；
                    # 在 head（H）维度上取平均，得到 [B, D1, D2]
                    attn_avg = attn.mean(dim=1)
                    # 取 batch 中第一个样本（[D1, D2]）
                    attn_sample = attn_avg[0]
                    # 展平为一维向量
                    flat_attn = attn_sample.flatten().detach().cpu().numpy()
                    # 当前 timestep（选取 batch 中第一个样本的时间步）
                    current_timestep = timesteps[0].item() if torch.is_tensor(timesteps[0]) else timesteps[0]
                    # 记录历史数据
                    if module_id not in self.unet_attn_history:
                        self.unet_attn_history[module_id] = []
                    self.unet_attn_history[module_id].append((current_timestep, flat_attn))
                    return out
                module.cross_attention.forward = new_forward
            else:
                print(f"[forward_with_unet_attnion_vis] 模块 {module} 不支持 forward_w_weight，跳过。")
        
        # 执行 UNet 前向传播
        out_unet = self.unet(x, t=timesteps, cond=enc_text, cond_indices=cond_indices)
        out_unet = out_unet[:, :, :T].transpose(1, 2)  # [B, T, nfeats]
        
        # 恢复被劫持模块的原始 forward 方法
        for module in cross_attn_modules:
            module_id = id(module)
            if module_id in orig_forwards:
                module.cross_attention.forward = orig_forwards[module_id]
        
        # 当 timestep==0 时，说明扩散过程结束，绘制历史记录的热力图  
        current_timestep = timesteps[0].item() if torch.is_tensor(timesteps[0]) else timesteps[0]
        if current_timestep == 0:
            output_dir = os.path.join(vis_dir, "history")
            os.makedirs(output_dir, exist_ok=True)
            # 对每个模块的记录生成热力图
            for module_id, records in self.unet_attn_history.items():
                # records 为列表，每个元素是 (timestep, flat_attn_vector)
                # 按 timestep 排序（这里假设较大值为较早时刻）
                records = sorted(records, key=lambda x: x[0], reverse=True)
                # 叠加所有记录得到二维矩阵：行对应不同 timestep，列对应 flat attention vector 的各个元素
                hist_matrix = np.stack([rec[1] for rec in records], axis=0)  # shape = (num_records, vector_length)
                # 同时获取所有时间步
                ts_list = np.array([rec[0] for rec in records])
                
                # 下面计算直方图：对每个时间步下的 attention 向量计算频数分布
                # 首先确定全体注意力值的 min 和 max（所有记录共用）
                all_attn = np.concatenate([rec[1] for rec in records])
                vmin, vmax = all_attn.min(), all_attn.max()
                nbins = 100
                bins = np.linspace(vmin, vmax, nbins + 1)
                # 对每个时间步的 attention 数值，计算直方图（返回计数数组）
                hist_data = []
                for (_, flat_attn) in records:
                    counts, _ = np.histogram(flat_attn, bins=bins)
                    hist_data.append(counts)
                # 构造一个矩阵，每行对应某个时间步所计算的直方图
                hist_matrix2D = np.stack(hist_data, axis=0)  # shape: (num_records, nbins)
                # 为便于观察，可以对直方图进行归一化
                hist_matrix2D = hist_matrix2D / hist_matrix2D.max()
                plt.figure(figsize=(10, 6))
                # 绘制热力图：横轴为直方图 bin 索引，纵轴为时间步
                plt.imshow(hist_matrix2D, aspect='auto', cmap='viridis', origin='upper')
                plt.colorbar(label="Normalized frequency")
                plt.xlabel("Attention value bin")
                plt.ylabel("Time step (record order)")
                # 用 bin 的中间值作为横轴标签
                bin_centers = 0.5 * (bins[:-1] + bins[1:])
                xticks = np.linspace(0, nbins - 1, min(nbins, 10)).astype(int)
                plt.xticks(xticks, [f"{bin_centers[i]:.2f}" for i in xticks])
                # 纵轴的 label 使用对应保存的 timestep 值
                yticks = np.arange(len(ts_list))
                plt.yticks(yticks, ts_list)
                plt.title(f"Module {module_id} Attention Value Distribution Over Time")
                save_path = os.path.join(output_dir, f"module_{module_id}_attention_heatmap.png")
                plt.savefig(save_path, bbox_inches="tight")
                plt.close()
                print(f"Saved attention heatmap plot to {save_path}")
            # 清空历史记录，方便后续新的记录
            self.unet_attn_history.clear()
        
        return out_unet


    def forward_with_attention_vis(self, text_encode_features, timesteps,caption=None):
        """
        实现流程：
          1. 劫持各层 PerceiverAttentionBlock 的 MultiheadAttention，使得调用时传入
             need_weights=True, average_attn_weights=True，从而得到形状 [B, 77, 154] 的注意力矩阵（已在 head 上平均）。
          2. 分别保存每一层（取 batch 中第一个样本）的 attention 图到：
             "./output/{timesteps[0]}/attention_map_layer_{i}.png"。
          3. 对每一层，计算第一层 attention 的统计数据：
             - 对第一层 attention（shape [77, 154]）沿 query 维度取均值，得到 1×154 向量。
             - 前 77 个位置代表 noise 部分，后 77 个位置代表 token 部分，分别求和得到百分比。
          4. 将每一层 token 部分（位置 77~153，1×77 向量）以及对应的 timestep、noise/token 百分比记录到内部字典中。
          5. 当 timesteps[0]==0 时，对每一层分别聚合记录的数据（形成 n×77 的矩阵），并生成聚合图；
             聚合图的 y 轴标签显示具体的 timestep 以及 noise/token 百分比。
        """
        # -----------------------------
        # 1. 劫持各层 MultiheadAttention 前向传播，记录 attention map
        # -----------------------------
        attn_maps = {}  # key: 模块 id, value: attention tensor, shape: [B, 77, 154]
        hook_handles = []
        original_attention_methods = []
        layer_keys = []  # 记录各层顺序
        
        def attn_hook(module, input, output):
            # output: (attn_out, attn_weights)
            attn_out, attn_weights = output
            # 得到的 attn_weights 的 shape 为 [B, 77, 154]
            attn_maps[id(module)] = attn_weights.detach().cpu()
        
        # 遍历所有 PerceiverAttentionBlock，假设在 self.ella_model.connector.perceiver_blocks 内
        for p_block in self.ella_model.connector.perceiver_blocks:
            handle = p_block.attn.register_forward_hook(attn_hook)
            hook_handles.append(handle)
            layer_keys.append(id(p_block.attn))
            original_attention_methods.append(p_block.attention)
            
            # 替换 attention 方法，传入 average_attn_weights=True
            def new_attention(q, kv, timestep_embedding=None, self_attn=p_block.attn):
                attn_out, attn_weight = self_attn(
                    q, kv, kv, need_weights=True, average_attn_weights=True
                )
                return attn_out
            p_block.attention = new_attention
        
        # -----------------------------
        # 2. 调用 ELLA 前向传播（hook 会记录各层 attention map）
        # -----------------------------
        outputs = self.ella_model(text_encode_features, timesteps)
        
        # 恢复 hook 与原始方法
        for handle in hook_handles:
            handle.remove()
        for p_block, orig_attn in zip(self.ella_model.connector.perceiver_blocks, original_attention_methods):
            p_block.attention = orig_attn
        
        if len(attn_maps) == 0:
            print("未捕获到 attention map，请检查内部调用。")
            return outputs
        
        # -----------------------------
        # 3. 设定保存目录 "./output/{timesteps[0]}"
        # -----------------------------
        timestep_value = timesteps[0].item() if torch.is_tensor(timesteps[0]) else timesteps[0]
        output_dir = os.path.join("./output_sample_55", str(timestep_value))
        os.makedirs(output_dir, exist_ok=True)
        
        # -----------------------------
        # 4. 分别保存每一层的注意力图（取 batch 中第一个样本）
        # -----------------------------
        for layer_idx, module_id in enumerate(layer_keys):
            attn_tensor = attn_maps[module_id]  # shape: [B, 77, 154]
            attn_to_plot = attn_tensor[0].numpy()  # 取 batch 第一个样本
            plt.figure(figsize=(6, 5))
            plt.imshow(attn_to_plot, cmap='viridis', aspect='auto')
            plt.colorbar()
            title_str = f"Layer {layer_idx}\nTimestep: {timestep_value}"
            plt.title(title_str)
            plt.xlabel("Key sequence position")
            plt.ylabel("Query sequence position")
            save_path = os.path.join(output_dir, f"attention_map_layer_{layer_idx}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved attention map: {save_path}")
        
        # -----------------------------
        # 5. 对每一层分别统计 noise/token 信息，并记录 token attention数据
        # -----------------------------
        # 遍历每一层
        for layer_idx, module_id in enumerate(layer_keys):
            attn_layer = attn_maps[module_id]         # shape: [B, 77, 154]
            first_sample_attn = attn_layer[0]            # shape: [77, 154]
            # 沿 query 维度 (77) 求均值，得到 1×154 的向量
            attn_vector = first_sample_attn.mean(dim=0)  # tensor, shape: [154]
            # 前 77 表示 noise 部分，后 77 表示 token 部分
            noise_sum = attn_vector[:77].sum()
            token_sum = attn_vector[77:154].sum()
            total_sum = attn_vector.sum()
            noise_percentage = (noise_sum / total_sum * 100).item()
            token_percentage = (token_sum / total_sum * 100).item()
            token_attn_vector = attn_vector[77:154]   # [77]
            
            # 将该层数据记录到字典中
            if layer_idx not in self.token_attn_history_dict:
                self.token_attn_history_dict[layer_idx] = []
                self.timestep_history_dict[layer_idx] = []
                self.noise_follow_history_dict[layer_idx] = []
                self.token_follow_history_dict[layer_idx] = []
            self.token_attn_history_dict[layer_idx].append(token_attn_vector.cpu())
            self.timestep_history_dict[layer_idx].append(timestep_value)
            self.noise_follow_history_dict[layer_idx].append(noise_percentage)
            self.token_follow_history_dict[layer_idx].append(token_percentage)
            print(f"Layer {layer_idx} Timestep {timestep_value}: Noise Follow: {noise_percentage:.2f}% , Token Follow: {token_percentage:.2f}%")
        
        # -----------------------------
        # 6. 当 timestep_value==0 时，为每一层分别绘制聚合图
        #     聚合图上每一行显示一个记录的 timestep 以及对应的 noise/token 百分比
        # -----------------------------
        if timestep_value == 0:
            # 如果有 caption，则利用 tokenizer 得到 token 序列并去除 pad token
            if caption is not None:
                text_inputs = self.t5_encoder.tokenizer(
                    caption,
                    return_tensors="pt",
                    add_special_tokens=True,
                    max_length=self.t5_encoder.max_length,
                    padding="max_length",
                    truncation=True,
                )
                token_ids = text_inputs.input_ids[0]  # shape: [max_length] (77)
                tokens_all = self.t5_encoder.tokenizer.convert_ids_to_tokens(token_ids)
                pad_token = self.t5_encoder.tokenizer.pad_token  # 通常为 "<pad>"
                # 取出非 pad token 的索引
                valid_indices = [i for i, t in enumerate(tokens_all) if t != pad_token]
                tokens = [tokens_all[i] for i in valid_indices]
            else:
                tokens = [str(i) for i in range(77)]
                valid_indices = list(range(77))
            
            for layer_idx in self.token_attn_history_dict:
                # 聚合 token attention 数据，得到 shape: [n, 77]
                token_attn_matrix = torch.stack(self.token_attn_history_dict[layer_idx], dim=0).numpy()  # shape: [n, 77]
                # 如果有 caption，则对 token_attn_matrix 按列截断，仅保留有效 token 列
                token_attn_matrix = token_attn_matrix[:, valid_indices]
                num_tokens = token_attn_matrix.shape[1]
                n = token_attn_matrix.shape[0]
                
                # 对整个 token_attn_matrix 进行归一化
                norm_min = token_attn_matrix.min()
                norm_max = token_attn_matrix.max()
                if norm_max - norm_min > 0:
                    token_attn_matrix_norm = (token_attn_matrix - norm_min) / (norm_max - norm_min)
                else:
                    token_attn_matrix_norm = token_attn_matrix
                # 构造 y 轴标签，每一行对应一个记录的时间步
                y_labels = [
                    f"{ts} "
                    for ts in self.timestep_history_dict[layer_idx]
                ]
                x = self.timestep_history_dict[layer_idx]
                y1 = self.noise_follow_history_dict[layer_idx]
                y2 = self.token_follow_history_dict[layer_idx]
                # 创建图形
                plt.figure(figsize=(15, 10))
                
                # 绘制第一条折线: noise follow history
                plt.plot(x, y1, marker='o', linestyle='-', color='b', label='Timestep Attention Weight Ratio')
                
                # 绘制第二条折线: token follow history
                plt.plot(x, y2, marker='s', linestyle='--', color='r', label='Text Feature Attention Weight Ratio')
                
                # 添加标题、轴标签、图例和网格
                #plt.title(f'Timestep Atten vs Text Feature Atten')
                plt.xlabel('Timestep')
                plt.ylabel('Attention Weight Ratio %')
                plt.legend()
                plt.grid(True)
                plt.gca().invert_xaxis()

                line_save_path = os.path.join(output_dir, f"timestep_vs_text_follow_{layer_idx}.png")
                plt.savefig(line_save_path, bbox_inches='tight')
                plt.close()
                print(f"Saved aggregated token attention for Layer {layer_idx}: {line_save_path}")

                plt.figure(figsize=(15, 7))
                plt.imshow(token_attn_matrix_norm, cmap='Reds', aspect='auto')
                plt.colorbar()
                plt.title(f"{caption[0]}")
                # 将 x 轴的 tick 改为 token，对齐对应的 token 序列
                ax = plt.gca()
                ax.set_xticks(np.arange(num_tokens))
                ax.set_xticklabels(tokens,rotation=90, fontsize=10)
                ax.tick_params(axis='x', pad=15)
                # 设置 y 轴 ticks 和标签
                ax.set_yticks(np.arange(n))
                ax.set_yticklabels(y_labels)
                plt.subplots_adjust(left=0.3)
                agg_save_path = os.path.join(output_dir, f"aggregated_token_attention_layer_{layer_idx}.png")
                plt.savefig(agg_save_path, bbox_inches='tight')
                plt.close()
                print(f"Saved aggregated token attention for Layer {layer_idx}: {agg_save_path}")

                    # 如有需要，这里可以清空对应历史数据：
                    # self.token_attn_history_dict[layer_idx] = []
                    # self.timestep_history_dict[layer_idx] = []
                    # self.noise_follow_history_dict[layer_idx] = []
                    # self.token_follow_history_dict[layer_idx] = []
        return outputs


if __name__ == "__main__":



    device = 'cuda:7'
    n_feats = 263
    num_frames = 196
    text_latent_dim = 256
    dim_mults = [2,2,2,2]
    base_dim= 512
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
        clip_version= 'ViT-B/32',
        text_encoder_type = 't5',
        disable_set=False,
        enable_cfg_scheduler=False,
        cfg_scheduler_type='moclip',
        device=device
    )

    model = model.to(device)
    dtype = torch.float32
    bs = 32
    x = torch.rand((bs, 196, 263),dtype=dtype ).to(device)
    timesteps = torch.randint(low=0, high=1000, size=(bs,)).to(device)
    y = ['A man jumps to his left.' for i in range(bs)]
    length = torch.randint(low=20, high=196, size=(bs,)).to(device)

    out = model(x, timesteps, text=y)
    print(out.shape)
    model.eval()
    out = model.forward_with_cfg(x, timesteps, text=y)
    print(out.shape)
    model.train()