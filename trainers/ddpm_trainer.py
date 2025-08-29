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

        # Physics alignment settings
        self.physics_loss_weight = getattr(args, 'physics_loss_weight', 0.1)
        accelerator.print(f'Physics alignment loss weight: {self.physics_loss_weight}')

        accelerator.print('Diffusion_config:\n',self.noise_scheduler.config)

        if self.accelerator.is_main_process:
            starttime = time.strftime("%Y-%m-%d_%H:%M:%S")
            print("Start experiment:", starttime)
            self.writer = SummaryWriter(log_dir=pjoin(args.save_root,'logs_')+starttime[:16],comment=starttime[:16],flush_secs=60)#以实验时间命名，[:13]可以自定义，我是定义到小时基本能确定是哪个实验了
        self.accelerator.wait_for_everyone()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay)
        self.scheduler = ExponentialLR(self.optimizer, gamma=args.decay_rate) if args.decay_rate>0 else None

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

        # Extract key physics information from x_start for alignment
        self.physics_target = self.extract_physics_info(x_start)

        # 1. Sample noise that we'll add to the motion
        real_noise = torch.randn_like(x_start)

        # 2. Sample a random timestep for each motion
        t = torch.randint(0, self.diffusion_steps, (B,), device=self.device)
        self.timesteps = t

        # 3. Add noise to the motion according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        x_t = self.noise_scheduler.add_noise(x_start, real_noise, t)

        # 4. network prediction with physics features
        self.prediction, self.physics_prediction = self.model(x_t, t, text=caption, return_physics=True)
        
        if self.opt.prediction_type =='sample':
            self.target = x_start
        elif self.opt.prediction_type == 'epsilon':
            self.target = real_noise
        elif self.opt.prediction_type == 'v_prediction':
            self.target = self.noise_scheduler.get_velocity(x_start, real_noise, t)

    def masked_l2(self, a, b, mask, weights):
        
        loss = self.mse_criterion(a, b).mean(dim=-1) # (bath_size, motion_length)
        
        loss = (loss * mask).sum(-1) / mask.sum(-1) # (batch_size, )

        loss = (loss * weights).mean()

        return loss

    def masked_physics_l2(self, pred_physics, target_physics, physics_weights, mask):
        """
        Calculate weighted physics alignment loss with temporal masking
        
        Args:
            pred_physics: [B, T, physics_dim] - predicted physics features
            target_physics: [B, T, physics_dim] - target physics features  
            physics_weights: [physics_dim] - per-dimension weights
            mask: [B, T] - temporal mask for valid frames
        
        Returns:
            physics_loss: scalar - weighted physics alignment loss
        """
        # Ensure physics_weights has the right shape for broadcasting
        physics_weights = physics_weights.view(1, 1, -1)  # [1, 1, physics_dim]
        
        # Calculate per-dimension MSE: [B, T, physics_dim]
        physics_mse = self.mse_criterion(pred_physics, target_physics)
        
        # Apply physics dimension weights: [B, T, physics_dim]
        weighted_physics_mse = physics_mse * physics_weights
        
        # Average across physics dimensions: [B, T]
        physics_loss_per_frame = weighted_physics_mse.mean(dim=-1)
        
        # Apply temporal mask and average: [B] -> scalar
        mask_expanded = mask.unsqueeze(-1)  # [B, T, 1] for broadcasting
        valid_frames = mask.sum(dim=-1)  # [B] - number of valid frames per batch
        
        # Mask and sum over time, then average over valid frames
        masked_loss = (physics_loss_per_frame * mask).sum(dim=-1) / (valid_frames + 1e-8)  # [B]
        
        return masked_loss.mean()  # scalar

    def backward_G(self):
        loss_logs = OrderedDict({})
        mse_loss_weights = torch.ones_like(self.timesteps)
        
        # Main diffusion loss
        loss_logs['loss_mot_rec'] = self.masked_l2(self.prediction, self.target, self.src_mask, mse_loss_weights)
        
        # Physics alignment loss
        if hasattr(self, 'physics_prediction') and self.physics_prediction is not None:
            # Convert physics target to tensor format
            physics_target_tensor, physics_weights = self.get_physics_tensor_and_weights(self.physics_target)
            
            # Handle temporal dimension mismatch due to downsampling
            pred_T = self.physics_prediction.shape[1]
            target_T = physics_target_tensor.shape[1]
            
            if pred_T != target_T:
                # Use interpolation to match dimensions exactly
                if pred_T < target_T:
                    # Downsample target to match prediction using interpolation
                    physics_target_tensor = torch.nn.functional.interpolate(
                        physics_target_tensor.transpose(1, 2),  # [B, 29, T]
                        size=pred_T,
                        mode='linear',
                        align_corners=False
                    ).transpose(1, 2)  # [B, pred_T, 29]
                    
                    # Downsample mask using nearest neighbor
                    downsampled_mask = torch.nn.functional.interpolate(
                        self.src_mask.unsqueeze(1).float(),  # [B, 1, T]
                        size=pred_T,
                        mode='nearest'
                    ).squeeze(1)  # [B, pred_T]
                else:
                    # Upsample prediction to match target (less likely case)
                    self.physics_prediction = torch.nn.functional.interpolate(
                        self.physics_prediction.transpose(1, 2),  # [B, 29, pred_T]
                        size=target_T,
                        mode='linear',
                        align_corners=False
                    ).transpose(1, 2)  # [B, target_T, 29]
                    downsampled_mask = self.src_mask
            else:
                downsampled_mask = self.src_mask
            
            # Calculate physics alignment loss
            loss_logs['loss_physics_align'] = self.masked_physics_l2(
                self.physics_prediction, 
                physics_target_tensor, 
                physics_weights, 
                downsampled_mask
            )
            
            # Combine losses with weighting
            self.loss = loss_logs['loss_mot_rec'] + self.physics_loss_weight * loss_logs['loss_physics_align']
        else:
            self.loss = loss_logs['loss_mot_rec']

        return loss_logs

    def update(self):
        self.zero_grad([self.optimizer])
        loss_logs = self.backward_G()
        self.accelerator.backward(self.loss)
        self.clip_norm([self.model])
        self.step([self.optimizer])

        return loss_logs
    
    def extract_physics_info(self, x_start):
        """
        Extract key physics information from motion data for alignment
        x_start: [B, T, 263] - motion representation
        
        Returns:
            physics_dict: Dictionary with physics components and their weights
        """
        B, T, D = x_start.shape
        
        # Physics dimension indices based on PHYSICS_DIMENSIONS_README.md
        # Motion Representation (263 dimensions):
        # Root Data [0:4]: [root_angular_vel_y, root_linear_vel_x, root_linear_vel_z, root_height_y]
        # RIC Data [4:67]: joint local positions  
        # Rotation Data [67:193]: joint rotations (6D)
        # Local Velocity [193:259]: joint velocities
        # Foot Contact [259:263]: foot contact states
        
        physics_components = {}
        
        # 1. Root motion (weight: 2.0)
        root_angular_vel = x_start[:, :, 0:1]  # root angular velocity Y
        root_linear_vel = x_start[:, :, 1:3]   # root linear velocity XZ
        root_height = x_start[:, :, 3:4]       # root height Y
        
        physics_components['root_motion'] = {
            'data': torch.cat([root_angular_vel, root_linear_vel, root_height], dim=-1),  # [B, T, 4]
            'weight': 2.0
        }
        
        # 2. Foot contact (weight: 3.0)
        foot_contact = x_start[:, :, 259:263]  # [B, T, 4] - left/right foot contact
        physics_components['foot_contact'] = {
            'data': foot_contact,
            'weight': 3.0
        }
        
        # 3. Key joint velocities (weight: 1.5)
        # Root velocity 3D: [193:196]
        # Left foot velocity: [214:217] (assuming foot joint index)
        # Right foot velocity: [217:220] (assuming foot joint index)
        root_velocity_3d = x_start[:, :, 193:196]
        left_foot_vel = x_start[:, :, 214:217] 
        right_foot_vel = x_start[:, :, 217:220]
        
        physics_components['joint_velocities'] = {
            'data': torch.cat([root_velocity_3d, left_foot_vel, right_foot_vel], dim=-1),  # [B, T, 9]
            'weight': 1.5
        }
        
        # 4. Key joint positions (weight: 1.0)
        # Hip positions: [7:13] (left hip: [10:13], right hip: [7:10])
        # Ankle positions: [55:61] (left ankle: [58:61], right ankle: [55:58])
        hip_positions = x_start[:, :, 7:13]    # [B, T, 6]
        ankle_positions = x_start[:, :, 55:61] # [B, T, 6]
        
        physics_components['joint_positions'] = {
            'data': torch.cat([hip_positions, ankle_positions], dim=-1),  # [B, T, 12]
            'weight': 1.0
        }
        
        return physics_components

    def get_physics_tensor_and_weights(self, physics_components):
        """
        Convert physics components dict to tensors for loss calculation
        
        Returns:
            physics_tensor: [B, T, total_physics_dims] - concatenated physics data
            physics_weights: [total_physics_dims] - per-dimension weights
        """
        physics_tensors = []
        physics_weights = []
        
        # Concatenate all physics components in order
        for component_name in ['root_motion', 'foot_contact', 'joint_velocities', 'joint_positions']:
            component = physics_components[component_name]
            data = component['data']  # [B, T, component_dims]
            weight = component['weight']
            
            physics_tensors.append(data)
            # Repeat weight for each dimension in this component
            physics_weights.extend([weight] * data.shape[-1])
        
        # Concatenate all physics data
        physics_tensor = torch.cat(physics_tensors, dim=-1)  # [B, T, total_dims]
        physics_weights = torch.tensor(physics_weights, device=physics_tensor.device, dtype=physics_tensor.dtype)
        
        return physics_tensor, physics_weights

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

    def eval_mode(self):
        self.model.eval()
        if self.model_ema:
            self.model_ema.eval()

    def save(self, file_name,total_it):
        state = {
            'opt_encoder': self.optimizer.state_dict(),
            'total_it': total_it,
            'encoder': self.accelerator.unwrap_model(self.model).state_dict(),
        }
        if self.model_ema:
            state["model_ema"] = self.accelerator.unwrap_model(self.model_ema).module.state_dict()
        torch.save(state, file_name)
        return

    def load(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.optimizer.load_state_dict(checkpoint['opt_encoder'])
        if self.model_ema:
            self.model_ema.load_state_dict(checkpoint["model_ema"], strict=True)
        self.model.load_state_dict(checkpoint['encoder'], strict=True)
       
        return checkpoint.get('total_it', 0)

    def train(self, train_loader):
        
        it = 0
        if self.opt.is_continue:
            model_path = pjoin(self.opt.model_dir, self.opt.continue_ckpt)         
            it = self.load(model_path)
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
                        self.writer.add_scalar("loss/motion_reconstruction", mean_loss['loss_mot_rec'], it)
                        if 'loss_physics_align' in mean_loss:
                            self.writer.add_scalar("loss/physics_alignment", mean_loss['loss_physics_align'], it)
                            self.writer.add_scalar("loss/total", mean_loss['loss_mot_rec'] + 
                                                 self.physics_loss_weight * mean_loss['loss_physics_align'], it)
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

 