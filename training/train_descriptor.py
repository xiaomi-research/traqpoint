import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from datetime import timedelta
from tqdm import tqdm
import numpy as np

from traqpoint.traqpoint import build
from traqpoint.traqpoint_helper import traqpoint_helper
from traqpoint.dataset.megadepth.megadepth import MegaDepthDataset
from traqpoint.dataset.megadepth import megadepth_warper
from training.losses.descriptor_loss import DescriptorLoss
from training.utils import check_accuracy
from benchmarks.mega_1500 import MegaDepthPoseMNNBenchmark
from traqpoint.utils import read_config

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

def parse_arguments():
    parser = argparse.ArgumentParser(description="TraqPoint descriptor training (Stage-1)")
    parser.add_argument('--megadepth_root_path', type=str, default='/high_perf_store3/evad-autolabeling/lihao_data/lyp-data/megadepth')
    parser.add_argument('--test_data_root', type=str, default='/high_perf_store3/world-model/liuyepeng/data/open_source/test_data/megadepth_test_1500')
    parser.add_argument('--ckpt_save_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='TraqPoint')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma_steplr', type=float, default=0.5)
    parser.add_argument('--training_res', type=int, default=800)
    parser.add_argument('--save_ckpt_every', type=int, default=1000)
    parser.add_argument('--test_every_iter', type=int, default=6000)
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--num_encoder_layers', type=int, default=4)
    parser.add_argument('--enc_n_points', type=int, default=8)
    parser.add_argument('--num_feature_levels', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--distributed', action='store_true', default=False)
    parser.add_argument('--config_path', type=str, default='./configs/default.yaml')
    args = parser.parse_args()
    return args

args = parse_arguments()

class Trainer:
    def __init__(self, rank, args=None):
        config = read_config(args.config_path)
        config['num_encoder_layers'] = args.num_encoder_layers
        config['enc_n_points'] = args.enc_n_points
        config['num_feature_levels'] = args.num_feature_levels
        config['train_detector'] = False
        config['weights'] = args.weights

        if args.distributed:
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
            batch_size = int(args.batch_size / args.n_gpus)
            self.n_gpus = args.n_gpus
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            batch_size = args.batch_size
        print(f"Using device {device}")

        self.dev = device
        self.training_res = args.training_res
        self.rank = rank
        self.seed = 0
        self.set_seed(self.seed)

        model = build(config)
        if args.distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[device], find_unused_parameters=True
            )
            self.orig_model = self.model.module
        else:
            self.model = model.to(device)
            self.orig_model = self.model

        self.saved_ckpts = []
        self.best = -1.0
        self.batch_size = batch_size
        self.epochs = args.epochs
        self.opt = optim.AdamW(filter(lambda x: x.requires_grad, self.model.parameters()), lr=args.lr, weight_decay=1e-4)
        self.DescriptorLoss = DescriptorLoss(inv_temp=20)
        self.benchmark = MegaDepthPoseMNNBenchmark(data_root=args.test_data_root)

        train_base = f"{args.megadepth_root_path}/megadepth_indices"
        train_npz_root = f"{train_base}/scene_info_0.1_0.7"
        self.TRAINVAL_DATA_SOURCE = args.megadepth_root_path
        self.npz_paths = list(Path(train_npz_root).glob('*.npz'))
        self.epoch = 0
        self.create_data_loader()

        self.ckpt_save_path = Path(args.ckpt_save_path)
        os.makedirs(self.ckpt_save_path, exist_ok=True)
        os.makedirs(self.ckpt_save_path / 'logdir', exist_ok=True)
        self.save_ckpt_every = args.save_ckpt_every
        if rank == 0:
            self.writer = SummaryWriter(str(self.ckpt_save_path) + f'/logdir/{args.model_name}_' + time.strftime("%Y_%m_%d-%H_%M_%S"))
        else:
            self.writer = None
        self.model_name = args.model_name

        if args.distributed:
            self.scheduler = MultiStepLR(self.opt, milestones=[2, 5, 8, 11], gamma=args.gamma_steplr)
        else:
            self.scheduler = StepLR(self.opt, step_size=args.test_every_iter, gamma=args.gamma_steplr)

    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def create_data_loader(self):
        current_seed = self.seed + 42
        mega_crop = torch.utils.data.ConcatDataset([
            MegaDepthDataset(root=self.TRAINVAL_DATA_SOURCE, npz_path=str(path),
                             min_overlap_score=0.1, max_overlap_score=0.7,
                             image_size=self.training_res, num_per_scene=200,
                             gray=False, crop_or_scale='crop', seed=current_seed)
            for path in self.npz_paths
        ])
        mega_scale = torch.utils.data.ConcatDataset([
            MegaDepthDataset(root=self.TRAINVAL_DATA_SOURCE, npz_path=str(path),
                             min_overlap_score=0.1, max_overlap_score=0.7,
                             image_size=self.training_res, num_per_scene=200,
                             gray=False, crop_or_scale='scale', seed=current_seed)
            for path in self.npz_paths
        ])
        combined_dataset = torch.utils.data.ConcatDataset([mega_crop, mega_scale])

        if args.distributed:
            sampler = DistributedSampler(combined_dataset, rank=self.rank, num_replicas=self.n_gpus, drop_last=True)
        else:
            sampler = RandomSampler(combined_dataset)

        self.data_loader = DataLoader(
            combined_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=6,
            pin_memory=True,
        )

    def unpack_stage1_outputs(self, model_out):
        """Adapter to handle Stage-1 legacy (3-tuple) and official (5-tuple) outputs."""
        if isinstance(model_out, (list, tuple)):
            if len(model_out) == 3:
                feats, scoremap_like, hmap_like = model_out
            elif len(model_out) >= 5:
                feats = model_out[0]
                hmap_like = model_out[1]  # matchibility
                scoremap_like = model_out[2]  # policy_map
            else:
                raise ValueError(f"Unexpected model output length: {len(model_out)}")
        else:
            raise ValueError("Model output must be a tuple/list")
        return feats, scoremap_like, hmap_like

    def validate(self, total_steps):
        # only rank0
        if self.rank != 0:
            return
        with torch.no_grad():
            method = 'aliked'
            if args.distributed:
                self.model.module.eval()
                model_helper = traqpoint_helper(self.model.module)
            else:
                self.model.eval()
                model_helper = traqpoint_helper(self.model)
            test_out = self.benchmark.benchmark(model_helper, model_name='experiment', plot_every_iter=1, plot=False, method=method)
            auc5 = test_out['auc_5']
            auc10 = test_out['auc_10']
            auc20 = test_out['auc_20']
            print("Validate AUC:", auc5, auc10, auc20)
            self.writer.add_scalar('Accuracy/auc5', auc5, total_steps)
            self.writer.add_scalar('Accuracy/auc10', auc10, total_steps)
            self.writer.add_scalar('Accuracy/auc20', auc20, total_steps)
            if auc5 > self.best:
                self.best = auc5
                target = self.model.module if args.distributed else self.model
                torch.save(target.state_dict(), str(self.ckpt_save_path / f'{self.model_name}_best.pth'))
        self.model.train()

    def _inference(self, d):
        if d is not None:
            for k in d.keys():
                if isinstance(d[k], torch.Tensor):
                    d[k] = d[k].to(self.dev)
        p1, p2 = d['image0'], d['image1']
        positives_md_coarse = megadepth_warper.spvs_coarse(d, scale=4)

        with torch.no_grad():
            positives_c = positives_md_coarse

        # Sync skip decision across ranks to avoid collective mismatch
        is_corrupted_local = any(len(p) < 30 for p in positives_c)
        if args.distributed:
            skip_flag = torch.tensor([1 if is_corrupted_local else 0], device=self.dev)
            dist.all_reduce(skip_flag, op=dist.ReduceOp.SUM)
            is_corrupted = skip_flag.item() > 0
        else:
            is_corrupted = is_corrupted_local

        if is_corrupted:
            return None, None, None, None

        out1 = self.model(p1)
        out2 = self.model(p2)
        feats1, scores_map1, hmap1 = self.unpack_stage1_outputs(out1)
        feats2, scores_map2, hmap2 = self.unpack_stage1_outputs(out2)

        loss_items = []
        acc_coarse_items = []
        acc_kp_items = []

        for b in range(len(positives_c)):
            if len(positives_c[b]) > 10000:
                positives = positives_c[b][torch.randperm(len(positives_c[b]))[:10000]]
            else:
                positives = positives_c[b]
            pts1, pts2 = positives[:, :2], positives[:, 2:]

            h1 = hmap1[b, :, :, :]
            h2 = hmap2[b, :, :, :]

            m1 = feats1[b, :, pts1[:, 1].long(), pts1[:, 0].long()].permute(1, 0)
            m2 = feats2[b, :, pts2[:, 1].long(), pts2[:, 0].long()].permute(1, 0)
            loss_ds, loss_h, acc_kp = self.DescriptorLoss(m1, m2, h1, h2, pts1, pts2)
            loss_items.append(loss_ds.unsqueeze(0) + loss_h.unsqueeze(0))

            acc_coarse = check_accuracy(m1, m2)
            acc_kp_items.append(acc_kp)
            acc_coarse_items.append(acc_coarse)

        nb_coarse = len(pts1)
        loss = torch.cat(loss_items, -1).mean()
        acc_coarse = sum(acc_coarse_items) / len(acc_coarse_items)
        acc_kp = sum(acc_kp_items) / len(acc_kp_items)
        return loss, acc_coarse, acc_kp, nb_coarse

    def train(self):
        self.model.train()
        self.stride = 4
        total_steps = 0

        for epoch in range(self.epochs):
            if args.distributed:
                self.data_loader.sampler.set_epoch(epoch)
            pbar = tqdm(total=len(self.data_loader), desc=f"Epoch {epoch+1}/{args.epochs}") if self.rank == 0 else None

            for _, d in enumerate(self.data_loader):
                loss, acc_coarse, acc_kp, nb_coarse = self._inference(d)
                if loss is None:
                    if args.distributed:
                        dist.barrier()
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                self.opt.step()
                self.opt.zero_grad()

                if (total_steps + 1) % self.save_ckpt_every == 0 and self.rank == 0:
                    target = self.model.module if args.distributed else self.model
                    torch.save(target.state_dict(), str(self.ckpt_save_path / f'{self.model_name}_{total_steps + 1}.pth'))
                    self.saved_ckpts.append(total_steps + 1)
                    if len(self.saved_ckpts) > 5:
                        old = self.saved_ckpts.pop(0)
                        os.remove(str(self.ckpt_save_path / f'{self.model_name}_{old}.pth'))

                if args.distributed:
                    torch.distributed.barrier()

                if (total_steps + 1) % args.test_every_iter == 0:
                    self.validate(total_steps)

                if pbar is not None:
                    pbar.set_description(
                        'Loss: {:.4f} acc_coarse {:.3f} acc_kp: {:.3f} #matches_c: {:d}'.format(
                            loss.item(), acc_coarse, acc_kp, nb_coarse)
                    )
                    pbar.update(1)

                if self.rank == 0:
                    self.writer.add_scalar('Loss/total', loss.item(), total_steps)
                    self.writer.add_scalar('Accuracy/coarse_mdepth', acc_coarse, total_steps)
                    self.writer.add_scalar('Count/matches_coarse', nb_coarse, total_steps)

                if not args.distributed:
                    self.scheduler.step()
                total_steps += 1

            self.seed += 1
            self.set_seed(self.seed)
            if args.distributed:
                dist.barrier()
            self.scheduler.step()
            self.epoch += 1
            self.create_data_loader()


def main_worker(rank, args):
    trainer = Trainer(rank=rank, args=args)
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
        torch.multiprocessing.spawn(main_worker, nprocs=args.n_gpus, args=(args,))
    else:
        main_worker(0, args)
