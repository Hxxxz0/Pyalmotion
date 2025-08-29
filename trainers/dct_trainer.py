import torch
import time
import torch.optim as optim
from collections import OrderedDict
from utils.utils import print_current_loss
from os.path import join as pjoin

from diffusers import  DDPMScheduler
from torch.utils.tensorboard import SummaryWriter
import time
import pdb
import sys
import os
from torch.optim.lr_scheduler import ExponentialLR
import torch_dct
import torch.nn as nn
import torch.nn.functional as F

# 新增：导入MoCLIP的MotionEncoder
try:
    from MoCLIP import MotionEncoder, _init_clip_motion_model, GLOBAL_CACHE
    MOCLIP_AVAILABLE = True
except ImportError:
    print("Warning: MoCLIP not available. Representation alignment loss will be disabled.")
    MOCLIP_AVAILABLE = False 

class DDPMTrainer(object):

    def __init__(self, args, model,accelerator, model_ema=None):
        self.opt = args
        self.accelerator = accelerator
        self.device = self.accelerator.device
        self.model = model
        self.diffusion_steps = args.diffusion_steps
        self.noise_scheduler = DDPMScheduler(num_train_timesteps= self.diffusion_steps,
            beta_schedule=args.beta_schedule,
            variance_type="fixed_small",
            prediction_type= args.prediction_type,
            clip_sample=False)
        self.model_ema = model_ema
        if args.is_train:
            self.mse_criterion = torch.nn.MSELoss(reduction='none')

        accelerator.print('Diffusion_config:\n',self.noise_scheduler.config)

        # 新增：表征对齐损失相关初始化
        self._init_repr_align_loss(args)

        # 新增：跟踪当前步数
        self.current_step = 0

        if self.accelerator.is_main_process:
            starttime = time.strftime("%Y-%m-%d_%H:%M:%S")
            print("Start experiment:", starttime)
            self.writer = SummaryWriter(log_dir=pjoin(args.save_root,'logs_')+starttime[:16],comment=starttime[:16],flush_secs=60)#以实验时间命名，[:13]可以自定义，我是定义到小时基本能确定是哪个实验了
        self.accelerator.wait_for_everyone()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay)
        self.scheduler = ExponentialLR(self.optimizer, gamma=args.decay_rate) if args.decay_rate>0 else None

    def _init_repr_align_loss(self, args):
        """初始化表征对齐损失相关组件"""
        # 获取损失权重
        self.lambda_repr_align = getattr(args, 'lambda_repr_align', 0.3)
        
        # 初始化状态
        self.motion_style_extractor = None
        self.repr_align_enabled = False
        
        # 如果MoCLIP可用，则初始化相关组件
        if MOCLIP_AVAILABLE and args.is_train:
            self._init_motion_style_extractor(args)
            
        self.accelerator.print(f"Representation Alignment Loss - Enabled: {self.repr_align_enabled}")
        self.accelerator.print(f"Lambda repr_align: {self.lambda_repr_align}")

    def _init_motion_style_extractor(self, args):
        """初始化用于提取动作风格的MotionEncoder"""
        try:
            # 从args获取参数，现在支持自动检测数据集类型
            # motion_encoder_input_dim 在 train_options.py 中会根据数据集自动设置
            input_dim = getattr(args, 'motion_encoder_input_dim', args.dim_pose if hasattr(args, 'dim_pose') else 263)
            embed_dim = getattr(args, 'motion_encoder_embed_dim', 768)   # MoCLIP输出维度
            max_seq_len = getattr(args, 'motion_encoder_max_seq_len', 196)
            
            # 初始化MotionEncoder
            self.motion_style_extractor = MotionEncoder(
                input_dim=input_dim,
                embed_dim=embed_dim,
                max_seq_length=max_seq_len
            ).to(self.device)
            
            # 尝试加载预训练权重
            self._load_motion_encoder_weights(args)
            
            # 是否冻结权重
            freeze_encoder = getattr(args, 'freeze_motion_encoder', True)
            if freeze_encoder:
                for param in self.motion_style_extractor.parameters():
                    param.requires_grad = False
                self.motion_style_extractor.eval()
                self.accelerator.print("MotionEncoder权重已冻结")
            else:
                self.accelerator.print("MotionEncoder权重将参与训练")
            
            self.repr_align_enabled = True
            dataset_name = getattr(args, 'dataset_name', 'unknown')
            self.accelerator.print(f"成功初始化MotionEncoder (dataset={dataset_name}, input_dim={input_dim}, embed_dim={embed_dim})")
            
        except Exception as e:
            self.accelerator.print(f"初始化MotionEncoder失败: {e}")
            self.motion_style_extractor = None

    def _load_motion_encoder_weights(self, args):
        """加载MoCLIP的MotionEncoder预训练权重"""
        # moclip_model_path 在 train_options.py 中会根据数据集自动设置
        default_path = './checkpoints/moclip_training/checkpoint_epoch_20.pt' if not hasattr(args, 'dataset_name') or args.dataset_name == 't2m' else './checkpoints/moclip_kit_training/best_model.pt'
        moclip_path = getattr(args, 'moclip_model_path', default_path)
        load_weights = getattr(args, 'load_motion_encoder_weights', True)
        
        if not load_weights:
            self.accelerator.print("跳过加载MotionEncoder预训练权重")
            return
        
        self.accelerator.print(f"准备加载MoCLIP模型: {moclip_path}")
            
        try:
            # 临时设置device
            original_device = GLOBAL_CACHE.get("device", None)
            GLOBAL_CACHE["device"] = self.device
            
            # 初始化MoCLIP模型
            _init_clip_motion_model(moclip_path)
            
            # 提取MotionEncoder权重
            if GLOBAL_CACHE["motion_encoder"] is not None:
                state_dict = GLOBAL_CACHE["motion_encoder"].state_dict()
                self.motion_style_extractor.load_state_dict(state_dict)
                self.accelerator.print("成功加载MoCLIP MotionEncoder预训练权重")
            else:
                self.accelerator.print("警告：未能从MoCLIP提取MotionEncoder")
            
            # 清理GLOBAL_CACHE
            GLOBAL_CACHE["clip_model"] = None
            GLOBAL_CACHE["clip_tokenizer"] = None
            GLOBAL_CACHE["motion_encoder"] = None
            GLOBAL_CACHE["clip_motion_align_model"] = None
            if original_device is not None:
                GLOBAL_CACHE["device"] = original_device
            else:
                GLOBAL_CACHE.pop("device", None)
                
        except Exception as e:
            self.accelerator.print(f"加载MotionEncoder权重失败: {e}")
            self.accelerator.print("将使用随机初始化的权重")

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    def clip_norm(self,network_list):
        for network in network_list:
            self.accelerator.clip_grad_norm_(network.parameters(), self.opt.clip_grad_norm) # 0.5 -> 1

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def forward(self, batch_data):
        caption, motions, m_lens = batch_data
        motions = motions.detach().float()

        x_start = motions
        B, T = x_start.shape[:2]
        cur_len = torch.LongTensor([min(T, m_len) for m_len in  m_lens]).to(self.device)
        self.src_mask = self.generate_src_mask(T, cur_len).to(x_start.device)

        # 保存动作长度信息，用于后续的风格提取
        self.current_m_lens = cur_len

        # 1. Sample noise that we'll add to the motion
        real_noise = torch.randn_like(x_start)

        # 2. Sample a random timestep for each motion
        t = torch.randint(0, self.diffusion_steps, (B,), device=self.device)
        self.timesteps = t

        # 3. Add noise to the motion according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        x_t = self.noise_scheduler.add_noise(x_start, real_noise, t)

        # 4. network prediction
        self.prediction, self.repa_emb, self.style_emb_1 = self.model(x_t, t, text=caption)
        # torch_dct.dct only works on the last dimension, so we need to permute T to the last dim
        self.prediction_dct = torch_dct.dct( self.prediction.permute(0, 2, 1), norm='ortho').permute(0, 2, 1)
        
        # Extract low frequency components (first 1/3 of DCT coefficients)
        # 根据数据集调整lowf_cutoff以匹配repa_emb的维度
        if hasattr(self.opt, 'dataset_name') and self.opt.dataset_name == 'kit':
            lowf_cutoff = 63  # KIT数据集的repa_emb维度
        else:
            lowf_cutoff = T // 3  # 196 // 3 = 65 (T2M数据集)
        #self.prediction_dct_lowf = self.prediction_dct[:, :lowf_cutoff, :]
        
        if self.opt.prediction_type =='sample':
            self.target = x_start
        elif self.opt.prediction_type == 'epsilon':
            self.target = real_noise
        elif self.opt.prediction_type == 'v_prediction':
            self.target = self.noise_scheduler.get_velocity(x_start, real_noise, t)
        self.target_dct = torch_dct.dct(self.target.permute(0, 2, 1), norm='ortho').permute(0, 2, 1)
        
        # Extract low frequency components for target as well
        self.target_dct_lowf = self.target_dct[:, :lowf_cutoff, :]

    def masked_l2(self, a, b, mask, weights):
        
        loss = self.mse_criterion(a, b).mean(dim=-1) # (bath_size, motion_length)
        
        loss = (loss * mask).sum(-1) / mask.sum(-1) # (batch_size, )

        loss = (loss * weights).mean()

        return loss

    def backward_G(self):
        loss_logs = OrderedDict({})
        mse_loss_weights = torch.ones_like(self.timesteps)
        loss_logs['loss_mot_rec']= self.masked_l2(self.prediction, self.target, self.src_mask, mse_loss_weights)
        
        # REPA loss: 计算 repa_emb 和 target_dct_lowf 之间的loss
        lowf_cutoff = self.target_dct_lowf.shape[1]  # 65
        repa_mask = self.src_mask[:, :lowf_cutoff]   # 调整mask到低频维度
        loss_logs['loss_repa'] = self.masked_l2(self.repa_emb, self.target_dct_lowf, repa_mask, mse_loss_weights)
        
        # 新增：表征对齐损失
        loss_logs['loss_repr_align'] = self._compute_repr_align_loss()
        
        # 3万步之后将loss_repa和loss_repr_align的系数设置为0
        if self.current_step >= 50000:
            repa_coeff = 0.0
            repr_align_coeff = 0.0
        else:
            repa_coeff = getattr(self.opt, 'repa_coeff', 0.2)
            repr_align_coeff = getattr(self.opt, 'repr_align_coeff', 0.5)
        
        self.loss = loss_logs['loss_mot_rec'] + loss_logs['loss_repa'] * repa_coeff + repr_align_coeff * loss_logs['loss_repr_align']
        

        return loss_logs

    def _compute_repr_align_loss(self):
        """计算表征对齐损失：使用MoCLIP对真实数据编码后与style_emb_1对齐"""
        if not self.repr_align_enabled or self.motion_style_extractor is None:
            return torch.tensor(0.0, device=self.device)
        
        try:
            # 确保MotionEncoder在正确的设备上
            self.motion_style_extractor.to(self.device)
            
            # 使用MoCLIP对真实数据self.target进行编码
            # self.target: (B, T, D_pose)
            # self.current_m_lens: (B,) 实际动作长度
            with torch.set_grad_enabled(not getattr(self.opt, 'freeze_motion_encoder', True)):
                style_from_target = self.motion_style_extractor(self.target, self.current_m_lens)
            
            # 计算真实数据风格特征与预测风格嵌入的余弦相似度损失
            # style_from_target: (B, D_style) - 从真实动作提取的风格特征
            # self.style_emb_1: (B, D_style) - 模型预测的风格嵌入
            repr_align_loss = 1.0 - F.cosine_similarity(self.style_emb_1, style_from_target, dim=-1).mean()
            
            return repr_align_loss
            
        except Exception as e:
            self.accelerator.print(f"计算表征对齐损失时出错: {e}")
            return torch.tensor(0.0, device=self.device)

    def update(self):
        self.zero_grad([self.optimizer])
        loss_logs = self.backward_G()
        self.accelerator.backward(self.loss)
        
        # 梯度剪裁：包含主模型和可训练的motion_style_extractor
        models_to_clip = [self.model]
        if (self.repr_align_enabled and 
            self.motion_style_extractor is not None and 
            not getattr(self.opt, 'freeze_motion_encoder', True)):
            models_to_clip.append(self.motion_style_extractor)
        
        self.clip_norm(models_to_clip)
        self.step([self.optimizer])

        return loss_logs
    
    def generate_src_mask(self, T, length):
        B = len(length)
        src_mask = torch.ones(B, T)
        for i in range(B):
            for j in range(length[i], T):
                src_mask[i, j] = 0
        return src_mask

    def train_mode(self):
        self.model.train()
        if self.model_ema:
            self.model_ema.train()
        # 只有当motion_style_extractor未被冻结时才设置为训练模式
        if (self.repr_align_enabled and 
            self.motion_style_extractor is not None and 
            not getattr(self.opt, 'freeze_motion_encoder', True)):
            self.motion_style_extractor.train()

    def eval_mode(self):
        self.model.eval()
        if self.model_ema:
            self.model_ema.eval()
        # motion_style_extractor始终设置为评估模式（因为通常被冻结）
        if self.repr_align_enabled and self.motion_style_extractor is not None:
            self.motion_style_extractor.eval()

    def save(self, file_name,total_it):
        state = {
            'opt_encoder': self.optimizer.state_dict(),
            'total_it': total_it,
            'encoder': self.accelerator.unwrap_model(self.model).state_dict(),
        }
        if self.model_ema:
            state["model_ema"] = self.accelerator.unwrap_model(self.model_ema).module.state_dict()
        
        # 保存motion_style_extractor的状态（如果它是可训练的）
        if (self.repr_align_enabled and 
            self.motion_style_extractor is not None and 
            not getattr(self.opt, 'freeze_motion_encoder', True)):
            state["motion_style_extractor"] = self.motion_style_extractor.state_dict()
            
        torch.save(state, file_name)
        return

    def load(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.optimizer.load_state_dict(checkpoint['opt_encoder'])
        if self.model_ema:
            self.model_ema.load_state_dict(checkpoint["model_ema"], strict=True)
        self.model.load_state_dict(checkpoint['encoder'], strict=True)
        
        # 加载motion_style_extractor的状态（如果存在且是可训练的）
        if (self.repr_align_enabled and 
            self.motion_style_extractor is not None and 
            not getattr(self.opt, 'freeze_motion_encoder', True) and
            "motion_style_extractor" in checkpoint):
            self.motion_style_extractor.load_state_dict(checkpoint["motion_style_extractor"], strict=True)
            self.accelerator.print("成功加载motion_style_extractor状态")
       
        return checkpoint.get('total_it', 0)

    def train(self, train_loader):
        
        it = 0
        if self.opt.is_continue:
            model_path = pjoin(self.opt.model_dir, self.opt.continue_ckpt)         
            it = self.load(model_path)
            self.current_step = it  # 从检查点恢复时设置当前步数
            self.accelerator.print(f'continue train from  {it} iters in {model_path}')
        start_time = time.time()

        logs = OrderedDict()
        self.dataset = train_loader.dataset
        self.model,self.mse_criterion,self.optimizer,train_loader, self.model_ema = \
        self.accelerator.prepare(self.model,self.mse_criterion,self.optimizer,train_loader,self.model_ema)

        num_epochs = (self.opt.num_train_steps-it)//len(train_loader)  + 1 
        self.accelerator.print(f'need to train for {num_epochs} epochs....')
        
        for epoch in range(0, num_epochs):
            self.train_mode()
            for i, batch_data in enumerate(train_loader):
                self.forward(batch_data)
                log_dict = self.update()
                it += 1
                self.current_step = it  # 更新当前步数

                if self.model_ema and it % self.opt.model_ema_steps == 0:
                    self.accelerator.unwrap_model(self.model_ema).update_parameters(self.model)

                # update logger
                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v
                
                if it % self.opt.log_every == 0 :                   
                    mean_loss = OrderedDict({})
                    for tag, value in logs.items():
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    print_current_loss(self.accelerator,start_time, it, mean_loss, epoch, inner_iter=i)
                    if self.accelerator.is_main_process:
                        self.writer.add_scalar("loss",mean_loss['loss_mot_rec'],it)
                        if 'loss_repa' in mean_loss:
                            self.writer.add_scalar("loss_repa",mean_loss['loss_repa'],it)
                        if 'loss_repr_align' in mean_loss:
                            self.writer.add_scalar("loss_repr_align",mean_loss['loss_repr_align'],it)
                    self.accelerator.wait_for_everyone()
                
                if it % self.opt.save_interval == 0 and self.accelerator.is_main_process: # 500
                    self.save(pjoin(self.opt.model_dir, 'latest.tar').format(it), it)
                self.accelerator.wait_for_everyone()


                if (self.scheduler is not None) and (it % self.opt.update_lr_steps == 0) :
                    self.scheduler.step()

        # Save the last checkpoint if it wasn't already saved.
        if it % self.opt.save_interval != 0 and self.accelerator.is_main_process:
            self.save(pjoin(self.opt.model_dir, 'latest.tar'), it)

        self.accelerator.wait_for_everyone()
        self.accelerator.print('FINISH')

 