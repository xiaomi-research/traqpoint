import argparse
import os
import time
import sys
import glob
from pathlib import Path

from traqpoint.traqpoint_helper import traqpoint_helper

import torch.distributed

def parse_arguments():
    parser = argparse.ArgumentParser(description="TraqPoint training script.")
    parser.add_argument('--megadepth_root_path', type=str, default='/high_perf_store3/evad-autolabeling/lihao_data/lyp-data/megadepth',
                        help='Path to the MegaDepth dataset root directory.')
    parser.add_argument('--test_data_root', type=str, default='/high_perf_store3/world-model/liuyepeng/data/open_source/test_data/megadepth_test_1500',
                        help='Path to the MegaDepth test dataset root directory.')
    parser.add_argument('--ckpt_save_path', type=str, required=True,
                        help='Path to save the checkpoints.')
    parser.add_argument('--model_name', type=str, default='TraqPoint',
                        help='Name of the model to save.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training. Default is 4.')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate. Default is 0.0001.')
    parser.add_argument('--gamma_steplr', type=float, default=0.5,
                        help='Gamma value for StepLR scheduler. Default is 0.5.')
    parser.add_argument('--training_res', type=int,
                        default=800, help='Training resolution as width,height. Default is 800 for training descriptor.')
    parser.add_argument('--save_ckpt_every', type=int, default=1000,
                        help='Save checkpoints every N steps. Default is 500.')
    parser.add_argument('--test_every_iter', type=int, default=2000,
                        help='Save checkpoints every N steps. Default is 2000.')
    parser.add_argument('--weights', type=str, default=None,)
    parser.add_argument('--num_encoder_layers', type=int, default=4)
    parser.add_argument('--enc_n_points', type=int, default=8)
    parser.add_argument('--num_feature_levels', type=int, default=4)
    parser.add_argument('--train_detector', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=65)
    parser.add_argument('--distributed', action='store_true', default=False)
    parser.add_argument('--config_path', type=str, default='./configs/default.yaml')
    parser.add_argument('--n_keypoints', type=int, default=512)
    parser.add_argument('--lambda_entropy', type=float, default=0.01)
    parser.add_argument('--depth_thresh', type=float, default=0.2)
    parser.add_argument('--match_thresh', type=float, default=0.5, help='Cosine similarity threshold for descriptor matching')
    parser.add_argument('--w_track', type=float, default=1.0, help='Weight for tracking reward')
    parser.add_argument('--w_dist', type=float, default=0.1, help='Weight for distribution reward (entropy)')
    parser.add_argument('--grid_size', type=int, default=16, help='Divide image into grid_size x grid_size cells for sampling')
    parser.add_argument('--repeatability_thresh', type=float, default=3.0, help='Pixel distance threshold for repeatability')
    # Expert-guided parameters
    parser.add_argument('--use_supervision', action='store_true', default=True, help='Use OpenCV FAST detector for supervision warmup')
    parser.add_argument('--w_supervision', type=float, default=10.0, help='Initial weight for supervision loss')
    parser.add_argument('--supervision_warmup_epochs', type=float, default=2, help='Number of epochs to apply supervision loss')
    parser.add_argument('--ratio_thresh', type=float, default=0.75, help='Divide image into grid_size x grid_size cells for sampling')
    # Sampling strategy parameters
    parser.add_argument('--sampling_strategy', type=str, default='hybrid', choices=['grid', 'global', 'hybrid'])
    parser.add_argument('--num_global_samples', type=int, default=512, help="Number of points to sample from the global probability map in 'global' or 'hybrid' mode.")

    args = parser.parse_args()

    return args

args = parse_arguments()

import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from traqpoint.traqpoint import build
from benchmarks.mega_1500 import MegaDepthPoseMNNBenchmark
from traqpoint.dataset.megadepth.megadepth_video import MegaDepthSequenceDataset
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from training.losses.detector_rl_loss import ReinforceLoss, calculate_advanced_rewardv2, grid_based_sampling
from training.losses.fast_detect_lossv2 import SupervisionLoss, get_expert_keypoints_and_heatmap
from training.losses.descriptor_loss import DescriptorLoss

from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import timedelta

from traqpoint.utils import read_config

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
class Trainer():

    def __init__(self, rank, args=None):
        config = read_config(args.config_path)
        config['num_encoder_layers'] = args.num_encoder_layers
        config['enc_n_points'] = args.enc_n_points
        config['num_feature_levels'] = args.num_feature_levels
        config['train_detector'] = args.train_detector
        config['weights'] = args.weights
        
        # distributed training
        if args.distributed:
            print(f"Training in distributed mode with {args.n_gpus} GPUs")
            assert torch.cuda.is_available()
            device = rank

            torch.distributed.init_process_group(
                backend="nccl",
                world_size=args.n_gpus,
                rank=device,
                init_method="file://" + str(args.lock_file),
                timeout=timedelta(seconds=2000)
            )
            torch.cuda.set_device(device)

            # adjust batch size and num of workers since these are per GPU
            batch_size = int(args.batch_size / args.n_gpus)
            self.n_gpus = args.n_gpus
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            batch_size = args.batch_size
        print(f"Using device {device}")
        
        self.seed = 0
        self.set_seed(self.seed)
        self.training_res = args.training_res
        self.dev = device
        config['device'] = device
        model = build(config)
        self.reward_baseline = 0.0
        self.rank = rank
        self.ratio_thresh = args.ratio_thresh
        
   
        if args.distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[device], find_unused_parameters=True
            )
            # Save original model reference (DDP-wrapped model requires .module to access inner model)
            self.orig_model = self.model.module
        else:
            self.model = model.to(device)
            self.orig_model = self.model  # unified variable name for reuse in later logic
        
        self.saved_ckpts = []
        self.best = -1.0
        self.best_loss = 1e6
        self.fine_weight = 1.0
        self.dual_softmax_weight = 1.0
        self.heatmaps_weight = 1.0
        self.depth_thresh = args.depth_thresh
        self.match_thresh = args.match_thresh
        self.w_track = args.w_track
        self.w_dist = args.w_dist
        self.grid_size = args.grid_size
        self.repeatability_thresh = args.repeatability_thresh
        self.sampling_strategy = args.sampling_strategy
        self.num_global_samples = args.num_global_samples

        # Parameter grouping with different learning rates
        # Initialize parameter groups (separate backbone from other params)
        backbone_params = []          # DINO backbone trainable params (lr = 0.2*args.lr)
        backbone_param_names = []     # record backbone param names
        other_params = []             # other trainable params (lr = args.lr)
        other_param_names = []        # record other param names

        # Iterate over original model params (avoid DDP module prefix affecting name identification)
        for name, param in self.orig_model.named_parameters():
            if param.requires_grad:  # only consider params requiring gradient updates
                # Identify DINO backbone params (adjust keywords based on actual param names)
                if 'dino' in name:
                    backbone_params.append(param)
                    backbone_param_names.append(name)
                else:
                    other_params.append(param)
                    other_param_names.append(name)

        # Only print on main process (rank=0) to avoid duplicate output in distributed mode
        if self.rank == 0:
            print(f"=== DINO Backbone trainable params ({len(backbone_param_names)} total) ===")
            for name in backbone_param_names:
                print(f"  {name}")

            print(f"\n=== Other trainable params ({len(other_param_names)} total) ===")
            for name in other_param_names:
                print(f"  {name}")


        # rl learning params
        self.n_keypoints = args.n_keypoints
        # setup optimizer 
        self.batch_size = batch_size
        self.epochs = args.epochs
        self.opt = optim.AdamW(filter(lambda x: x.requires_grad, self.model.parameters()) , lr = args.lr, weight_decay=1e-4)
        self.use_supervision = args.use_supervision
        self.final_steps = 55000
        # losses
        if args.train_detector:
            self.ReinforceLoss = ReinforceLoss(lambda_entropy=args.lambda_entropy).to(self.dev)
            # Initialize supervision module
            if self.use_supervision:
                self.SupervisionLoss = SupervisionLoss().to(self.dev)
                # Compute supervision cutoff step
                self.supervision_cutoff_step = self.final_steps // 10 # 10
                self.initial_w_supervision = args.w_supervision
        else:
            self.DescriptorLoss = DescriptorLoss(inv_temp=20)
        
        self.benchmark = MegaDepthPoseMNNBenchmark(data_root=args.test_data_root)

        ##################### MEGADEPTH INIT ##########################
        
        TRAIN_BASE_PATH = f"{args.megadepth_root_path}/megadepth_indices"
        print('Loading MegaDepth dataset from ', TRAIN_BASE_PATH)
        TRAINVAL_DATA_SOURCE = args.megadepth_root_path
        self.TRAINVAL_DATA_SOURCE = TRAINVAL_DATA_SOURCE
        TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/sequence_indices_0.1_0.7_s5"
        self.seq_len = 5
        self.TRAIN_NPZ_ROOT = TRAIN_NPZ_ROOT
        npz_paths = glob.glob(TRAIN_NPZ_ROOT + '/*.npz')[:]
        self.npz_paths = npz_paths
        self.epoch = 0
        self.create_data_loader()
        
        ##################### MEGADEPTH INIT END #######################

        os.makedirs(args.ckpt_save_path, exist_ok=True)
        os.makedirs(args.ckpt_save_path / 'logdir', exist_ok=True)

        self.save_ckpt_every = args.save_ckpt_every
        self.ckpt_save_path = args.ckpt_save_path
        if rank == 0:
            self.writer = SummaryWriter(str(self.ckpt_save_path) + f'/logdir/{args.model_name}_' + time.strftime("%Y_%m_%d-%H_%M_%S"))
        else:
            self.writer = None
        self.model_name = args.model_name
        print("======grid_size=======", self.grid_size, self.ratio_thresh, self.sampling_strategy,self.num_global_samples)
        self.scheduler = CosineAnnealingLR(
            optimizer=self.opt,          # bind optimizer
            T_max=self.final_steps,           # total annealing period
            eta_min=5e-6        # minimum learning rate (default 5e-6)
        )
        #self.scheduler = MultiStepLR(self.opt, milestones=[10, 20, 30, 40, 50], gamma=0.5)
        
    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def create_data_loader(self):
        # Generate seed for current epoch (epoch + 42)
        if args.distributed:
            print(f"[Rank {self.rank}] Creating data_loader for epoch {self.epoch} with seed {self.seed}")
        current_seed = self.seed + 42  # use self.epoch to ensure different seed per epoch
        print("current seed", current_seed)

        # 1. Classify npz files by overlap level
        npz_low = []    # 0.1_0.3 low overlap
        npz_medium = [] # 0.3_0.5 medium overlap
        npz_high = []   # 0.5_0.7 high overlap
        
        for path in self.npz_paths:
            if "0.1_0.3" in path:
                npz_low.append(path)
            elif "0.3_0.5" in path:
                npz_medium.append(path)
            elif "0.5_0.7" in path:
                npz_high.append(path)
        
         # 2. Determine quantity allocation for current stage (total kept at 600, consistent with original)
        total_epochs = self.epochs
        current_epoch = self.epoch
        base_num = 350

        # Adjust quantity ratios for different overlap levels based on stage
        if current_epoch < total_epochs // 3:
            # Stage 1: high overlap at 60%
            high_num = int(base_num * 1.8)  # high overlap count (60%)
            medium_num = int(base_num * 0.6) # medium overlap (30%)
            low_num = int(base_num * 0.6)    # low overlap (10%)
            stage = "high overlap stage"
        elif current_epoch < 2 * (total_epochs // 3):
            # Stage 2: medium overlap at 60%
            high_num = int(base_num * 0.6)
            medium_num = int(base_num * 1.8)
            low_num = int(base_num * 0.6)
            stage = "medium overlap stage"
        else:
            # Stage 3: low overlap at 60%
            high_num = int(base_num * 0.6)
            medium_num = int(base_num * 0.6)
            low_num = int(base_num * 1.8)
            stage = "low overlap stage"

        if self.rank == 0:
            print(f"current: {stage} (epoch {current_epoch}/{total_epochs})")
            print(high_num, medium_num, low_num, high_num+medium_num+low_num)


        # 2) Build scale-mode datasets
        # High overlap scale dataset
        scale_high = torch.utils.data.ConcatDataset([MegaDepthSequenceDataset(
            root=self.TRAINVAL_DATA_SOURCE,
            npz_path=path,
            image_size=self.training_res,
            num_per_scene=high_num,
            gray=False,
            seq_len=self.seq_len,
            crop_or_scale='scale',
            seed=current_seed
        ) for path in npz_high])
        
        # Medium overlap scale dataset
        scale_medium = torch.utils.data.ConcatDataset([MegaDepthSequenceDataset(
            root=self.TRAINVAL_DATA_SOURCE,
            npz_path=path,
            image_size=self.training_res,
            num_per_scene=medium_num,
            gray=False,
            seq_len=self.seq_len,
            crop_or_scale='scale',
            seed=current_seed
        ) for path in npz_medium])
        
        # Low overlap scale dataset
        scale_low = torch.utils.data.ConcatDataset([MegaDepthSequenceDataset(
            root=self.TRAINVAL_DATA_SOURCE,
            npz_path=path,
            image_size=self.training_res,
            num_per_scene=low_num,
            gray=False,
            seq_len=self.seq_len,
            crop_or_scale='scale',
            seed=current_seed
        ) for path in npz_low])
        
        # Concatenate all scale datasets
        mega_scale = torch.utils.data.ConcatDataset([scale_high , scale_medium , scale_low])

        combined_dataset = mega_scale
        if args.distributed:
            #torch.distributed.barrier()  # all processes wait for dataset construction
            print(f"[Rank {self.rank}] Epoch {self.epoch}: Dataset barrier passed")
        # Create sampler
        if args.distributed:
            sampler = DistributedSampler(combined_dataset, rank=self.rank, num_replicas=self.n_gpus, drop_last=True, shuffle=True)
        else:
            # Create sampler
            sampler = RandomSampler(combined_dataset)

        self.data_loader = DataLoader(combined_dataset, 
                                    batch_size=self.batch_size, 
                                    sampler=sampler, 
                                    num_workers=8,
                                    pin_memory=True)
        if args.distributed:
            print(f"[Rank {self.rank}] Epoch {self.epoch}: DataLoader created")
            print(f"[Rank {self.rank}] DataLoader length: {len(self.data_loader)}")

    def validate(self, total_steps):
        # Only main process (rank=0) runs validation, others return directly
        if 1:
            with torch.no_grad():
                if args.train_detector:
                    method = 'sparse'
                else:
                    method = 'aliked'
                
                if args.distributed:
                    self.model.module.eval()
                    model_helper = traqpoint_helper(self.model.module)
                    test_out = self.benchmark.benchmark(model_helper, model_name='experiment', plot_every_iter=1, plot=False, method=method)
                else:
                    self.model.eval()
                    model_helper = traqpoint_helper(self.model)
                    test_out = self.benchmark.benchmark(model_helper, model_name='experiment', plot_every_iter=1, plot=False, method=method)
                    
                auc5 = test_out['auc_5']
                auc10 = test_out['auc_10']
                auc20 = test_out['auc_20']
                print("=============", auc5, auc10, auc20)
                if self.rank == 0:
                    self.writer.add_scalar('Accuracy/auc5', auc5, total_steps)
                    self.writer.add_scalar('Accuracy/auc10', auc10, total_steps)
                    self.writer.add_scalar('Accuracy/auc20', auc20, total_steps)
                    if auc5 > self.best:
                        self.best = auc5
                        if args.distributed:
                            torch.save(self.model.module.state_dict(), str(self.ckpt_save_path) + f'/{self.model_name}_best.pth')
                        else:
                            torch.save(self.model.state_dict(), str(self.ckpt_save_path) + f'/{self.model_name}_best.pth')
        # All processes wait here to ensure main process finishes testing before continuing
        
        self.model.train()
        
    
    def _inference(self, d, total_steps):
        if d is not None:
            for k in d.keys():
                if isinstance(d[k], torch.Tensor):
                    d[k] = d[k].to(self.dev)
        
        if args.train_detector:
            B, N_frames, C, H, W = d['images'].shape
            all_images = d['images'].view(B * N_frames, C, H, W)

            # 1. Forward pass
            all_feats, _, all_policy_maps, all_log_policy_maps, all_logits = self.model(all_images)
            all_policy_maps = all_policy_maps.view(B, N_frames, 1, H, W)
            all_log_policy_maps = all_log_policy_maps.view(B, N_frames, 1, H, W)
            all_logits = all_logits.view(B, N_frames, 1, H, W)

            # 2. Grid-based forced sampling (reference frame only)
            ref_sampled_coords, log_probs = grid_based_sampling(
                all_logits[:, 0], all_log_policy_maps[:, 0],
                grid_cell_size=self.grid_size,
                sampling_strategy=self.sampling_strategy,
                num_global_samples=self.num_global_samples
            )
            
            # 3. Compute reward
            _, D_desc, H_des, W_des = all_feats.shape
            all_desc_maps = all_feats.view(B, N_frames, D_desc, H_des, W_des)
            
            acceptance_mask = 1
            # 4. Compute advanced reward
            is_warmup = total_steps < self.supervision_cutoff_step
            if total_steps < self.supervision_cutoff_step:
                current_warm_weight = 0.1 + (1.0 - 0.1) * (total_steps / self.supervision_cutoff_step)
            else:
                current_warm_weight = 1.0
            
            total_reward, covisibility_count, mask_valid_ref_depth = calculate_advanced_rewardv2(ref_sampled_coords, d['images'], d['depths'], d['intrinsics'], d['rel_poses_to_ref'], \
                all_desc_maps, all_logits, 60, \
                self.depth_thresh, self.match_thresh, self.w_track, acceptance_mask, current_warm_weight, self.ratio_thresh, self.seq_len )
            
            # 1) method used when mask=0.001
            covisible_mask = (covisibility_count > 0)
            final_valid_mask = covisible_mask 
            # per_point_reward[~covisible_mask] = -0.1 * current_warm_weight # penalize inactive points

            
            current_avg_reward_for_log = total_reward.mean().item()
            self.reward_baseline = current_avg_reward_for_log
           
           

            # Compute dynamic lambda_entropy for current step
            decay_progress = min(1.0, total_steps / self.final_steps)
            current_lambda_entropy = 0.015 - (0.015 - 0.005) * decay_progress

            loss_components = self.ReinforceLoss(all_policy_maps[:, 0], log_probs, total_reward, final_valid_mask, current_lambda_entropy)
            
            rl_loss = loss_components['total_loss']
            sv_loss = torch.tensor(0.0, device=self.dev, dtype=rl_loss.dtype)
            # 6. Compute and add supervision loss
            if self.use_supervision and total_steps < self.supervision_cutoff_step:
                current_w_supervision = self.initial_w_supervision * (1.0 - total_steps / self.supervision_cutoff_step)
                current_rl_weight = 0.1 + (1.0 - 0.1) * (total_steps / self.supervision_cutoff_step)
                if current_w_supervision > 0:
                    expert_heatmap = get_expert_keypoints_and_heatmap(d['images'][:, 0])
                    ref_logits = all_logits[:, 0]
                    
                    loss_supervision = self.SupervisionLoss(ref_logits, expert_heatmap)
                    
                    sv_loss =  current_w_supervision * loss_supervision
                    total_loss = current_rl_weight * rl_loss + sv_loss
            else:
                total_loss = rl_loss
            
            return {
                # 'pos_num': (per_point_reward>0).sum(),
                # 'neg_num': (per_point_reward<0).sum(),
                'loss': total_loss,
                'reward_baseline': self.reward_baseline,
                'rl_loss': rl_loss,
                'sv_loss': sv_loss,
                'entropy': loss_components['entropy'],
                'avg_reward': loss_components['avg_reward']
            }
    
    def train(self):

        self.model.train()
        #self.stride = 4 if args.num_feature_levels == 5 else 8
        self.stride = 4
        total_steps = 0
        
        for epoch in range(self.epochs):
            
            if args.distributed:
                self.data_loader.sampler.set_epoch(epoch)
            pbar = tqdm(total=len(self.data_loader), desc=f"Epoch {epoch+1}/{args.epochs}") if self.rank == 0 else None
           
            for i, d in enumerate(self.data_loader):
                
                metrics = self._inference(d, total_steps)
                
                if metrics is None or metrics['loss'] is None:
                    continue
                
                loss = metrics['loss']
                # Compute Backward Pass
                self.opt.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                self.opt.step()
                

                if (total_steps + 1) % self.save_ckpt_every == 0 and self.rank == 0:
                    print('saving iter ', total_steps + 1)
                    if args.distributed:
                        torch.save(self.model.module.state_dict(), str(self.ckpt_save_path) + f'/{self.model_name}_{total_steps + 1}.pth')
                    else:
                        torch.save(self.model.state_dict(), str(self.ckpt_save_path) + f'/{self.model_name}_{total_steps + 1}.pth')
                    self.saved_ckpts.append(total_steps + 1)
                    if len(self.saved_ckpts) > 5:
                        os.remove(str(self.ckpt_save_path) + f'/{self.model_name}_{self.saved_ckpts[0]}.pth')
                        self.saved_ckpts = self.saved_ckpts[1:]
                
                if args.distributed:
                    torch.distributed.barrier()
                    
                if (total_steps+1) % args.test_every_iter == 0:
                    self.validate(total_steps)
                
                if pbar is not None:
                    if args.train_detector:
                        pbar.set_description(f"Loss : {metrics['loss'].item():.4f} | RL : {metrics['rl_loss'].item():.4f}  | Avg Reward: {metrics['avg_reward']:.3f} ")
                    
                    pbar.update(1)

                # Log metrics
                if self.rank == 0:
                    self.writer.add_scalar('Loss/total', loss.item(), total_steps)
                    self.writer.add_scalar('Loss/sv', metrics['sv_loss'].item(), total_steps)
                    self.writer.add_scalar('Loss/rl', metrics['rl_loss'].item(), total_steps)
                    self.writer.add_scalar('Reward/baseline', metrics['avg_reward'], total_steps)
                    self.writer.add_scalar('Reward/really', metrics['reward_baseline'], total_steps)
                    self.writer.add_scalar('Entropy/Entropy', metrics['entropy'], total_steps)
                    
                
                # if not args.distributed:
                self.scheduler.step()
                total_steps = total_steps + 1

            self.seed = self.seed + 1
            self.set_seed(self.seed)
            #self.scheduler.step()
            self.epoch = self.epoch + 1
            self.create_data_loader()

            if args.distributed:
                    torch.distributed.barrier()
             
def main_worker(rank, args):
    trainer = Trainer(
        rank=rank,
        args=args
    )

    # The most fun part
    
    trainer.train()
   

if __name__ == '__main__':
    if args.distributed:
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)  
    
    if not Path(args.ckpt_save_path).exists():
        os.makedirs(args.ckpt_save_path)
    
    args.ckpt_save_path = Path(args.ckpt_save_path).resolve()

    if args.distributed:
        args.n_gpus = torch.cuda.device_count()
        args.lock_file = Path(args.ckpt_save_path) / "distributed_lock"
        if args.lock_file.exists():
            args.lock_file.unlink()
        
        # Each process gets its own rank and dataset
        torch.multiprocessing.spawn(
            main_worker, nprocs=args.n_gpus, args=(args,)
        )
    else:
        main_worker(0, args)
