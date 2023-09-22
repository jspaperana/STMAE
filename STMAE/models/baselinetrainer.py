import os
import sys
import csv
import math
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from einops import rearrange
import matplotlib.pyplot as plt

sys.path.append('../')
from data_utils.dataset_pems import load_dataset
from utils.log_utils import save_csv_log
from utils.torch_utils import get_scheduler
from utils.loss_utils import masked_mae, masked_mape, masked_rmse, metric

def exists(x):
    return x is not None

class BaselineTrainer(object):
    def __init__(
        self,
        dataset,
        model,
        adj,
        loss,
        cfg,
        batch_size,
        learning_rate,
        weight_decay,
        opt_name='adam',
    ):
        super().__init__()

        self.model = model
        self.device = next(self.model.parameters()).device
        self.adj = adj  # The adjacency matrix is from https://arxiv.org/pdf/2108.11873.pdf: weighted with self_loops
        self.loss = loss

        self.cfg = cfg

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.in_horizon = cfg.in_horizon
        self.out_horizon = cfg.out_horizon
        self.dtype = torch.float32 if self.cfg.dtype == 'float32' else torch.float64

        self.dataset_name = cfg.dataset
        self.opt_name = opt_name

        # datasets
        if self.dataset_name == 'pems_03':
            input_data = './data/pems_03'
        elif self.dataset_name == 'pems_04':
            input_data = './data/pems_04'
        elif self.dataset_name == 'pems_07':
            input_data = './data/pems_07'
        elif self.dataset_name == 'pems_08':
            input_data = './data/pems_08'
        else:
            raise NotImplementedError

        self.dataloader = load_dataset(input_data, self.batch_size, self.batch_size, self.batch_size)
        self.scaler = self.dataloader['scaler'] 

        # optimizer
        if opt_name == 'adam':
            self.opt = Adam(self.model.parameters(), lr=learning_rate, eps=self.cfg.pt_eps, weight_decay=weight_decay)
        elif opt_name == 'adamw':
            self.opt = AdamW(self.model.parameters(), lr=learning_rate, eps=self.cfg.pt_eps, weight_decay=weight_decay)
        else:
            raise NotImplementedError

        self.scheduler = get_scheduler(self.opt, policy=cfg.pt_sched_policy, nepoch_fix=cfg.pt_num_epoch_fix_lr, nepoch=cfg.train_epoch, \
            decay_step=cfg.pt_decay_step, gamma=cfg.pt_gamma, milestones=cfg.pt_milestones)

        self.epoch = 0
        self.train_loss_list = []
        self.valid_loss_list = []
        self.best_valid_loss = float('inf')
        self.batches_seen = None
        print('Baseline Trainer initialization done.')

    def _get_error_info(self, prediction, target):
        mae = masked_mae(prediction, target, 0.0).item()
        rmse = masked_rmse(prediction, target, 0.0).item()
        mape = masked_mape(prediction, target, 0.0).item()
        error_info = {'mae': mae, 'rmse': rmse, 'mape': mape}
        return error_info

    def save(self, to_save_path):
        data = {
            'epoch': self.epoch,
            'train_loss_list': self.train_loss_list,
            'valid_loss_list': self.train_loss_list,
            'best_valid_loss': self.best_valid_loss,
            'batches_seen': self.batches_seen,
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict(),
            'sched': self.scheduler.state_dict() if exists(self.scheduler) else None,
        }
        torch.save(data, to_save_path)
        return

    def load(self, to_load_path):
        if self.cfg.backbone == 'dcrnn':
            self.model(
                torch.zeros(1, self.in_horizon, self.cfg.num_nodes, self.cfg.in_dim).to(self.device).to(self.dtype),
                labels=self.scaler.transform(torch.zeros(1, self.in_horizon, self.cfg.num_nodes, self.cfg.out_dim).to(self.device).to(self.dtype)), 
                batches_seen=1
            )
            optimizer = Adam if self.opt_name == 'adam' else AdamW
            self.opt = optimizer(self.model.parameters(), lr=self.learning_rate, eps=self.cfg.pt_eps, weight_decay=self.weight_decay)
            self.scheduler = get_scheduler(self.opt, policy=self.cfg.pt_sched_policy, nepoch_fix=self.cfg.pt_num_epoch_fix_lr, nepoch=self.cfg.train_epoch, \
                decay_step=self.cfg.pt_decay_step, gamma=self.cfg.pt_gamma, milestones=self.cfg.pt_milestones)
        device = self.device
        data = torch.load(to_load_path, map_location=device)
        self.epoch = data['epoch']
        self.train_loss_list = data['train_loss_list']
        self.valid_loss_list = data['valid_loss_list']
        self.best_valid_loss = data['best_valid_loss']
        self.batches_seen = data['batches_seen']
        self.model.load_state_dict(data['model'])
        self.opt.load_state_dict(data['opt'])
        if exists(data['sched']):
            self.scheduler.load_state_dict(data['sched'])  
        else: self.scheduler = None
        print(">>> finish loading baseline model ckpt from path '{}'".format(to_load_path))
        return

    def train(self):
        """
        baseline model for one epoch
        """ 
        self.model.train()

        t_s = time.time()
        epoch_loss = 0.
        epoch_iter = 0
        epoch_error_info = {}
        self.batches_seen = self.dataloader['train_loader'].num_batch * self.epoch    # this is for dcrnn specific training
        self.dataloader['train_loader'].shuffle()
        for idx, (x, y) in enumerate(self.dataloader['train_loader'].get_iterator()):
            x = torch.Tensor(x).to(self.device)[...,:self.cfg.in_dim]
            y = torch.Tensor(y).to(self.device)[...,:self.cfg.out_dim]
            output = self.model(x, labels=self.scaler.transform(y), batches_seen=self.batches_seen)
            output = self.scaler.inverse_transform(output)

            if self.batches_seen == 0 and self.cfg.backbone == 'dcrnn':   # dcrnn only
                if self.cfg.opt_name == 'adam':
                    self.opt = Adam(self.model.parameters(), lr=self.learning_rate, eps=self.cfg.pt_eps, weight_decay=self.weight_decay)
                elif self.cfg.opt_name == 'adamw':
                    self.opt = AdamW(self.model.parameters(), lr=self.learning_rate, eps=self.cfg.pt_eps, weight_decay=self.weight_decay)
                self.scheduler = get_scheduler(self.opt, policy=self.cfg.pt_sched_policy, nepoch_fix=self.cfg.pt_num_epoch_fix_lr, nepoch=self.cfg.train_epoch, \
                decay_step=self.cfg.pt_decay_step, gamma=self.cfg.pt_gamma, milestones=self.cfg.pt_milestones)

            loss = self.loss(output, y[...,:self.cfg.out_dim], 0.0)
            error_info = self._get_error_info(output, y[...,:self.cfg.out_dim])
            for key, value in error_info.items():
                if key not in epoch_error_info:
                    epoch_error_info[key] = value
                else:
                    epoch_error_info[key] += value
            self.opt.zero_grad()
            loss.backward()
            if exists(self.cfg.pt_clip_grad) and self.cfg.pt_clip_grad != 'None':
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.pt_clip_grad)
            self.opt.step()
            epoch_loss += loss.item()
            self.batches_seen += 1
            epoch_iter += 1
        if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and exists(self.scheduler):
            self.scheduler.step()
        
        self.epoch += 1
        epoch_loss /= epoch_iter
        for key, value in epoch_error_info.items():
            epoch_error_info[key] /= epoch_iter
        lr = self.opt.param_groups[0]['lr']
        dt = time.time() - t_s
        self.train_loss_list.append(epoch_loss)
        return lr, epoch_loss, epoch_error_info, dt
    
    @torch.no_grad()
    def valid(self):
        self.model.eval()
        total_iter = 0
        epoch_error_info = {}
        for idx, (x, y) in enumerate(self.dataloader['val_loader'].get_iterator()):
            x = torch.Tensor(x).to(self.device)[...,:self.cfg.in_dim]
            y = torch.Tensor(y).to(self.device)[...,:self.cfg.out_dim]
            output = self.model(x)
            output = self.scaler.inverse_transform(output)
            error_info = self._get_error_info(output, y[...,:self.cfg.out_dim])
            for key, value in error_info.items():
                if key not in epoch_error_info:
                    epoch_error_info[key] = value
                else:
                    epoch_error_info[key] += value
            total_iter += 1

        for key, value in epoch_error_info.items():
            epoch_error_info[key] /= total_iter
        self.valid_loss_list.append(epoch_error_info['mae'])
        return epoch_error_info

    @torch.no_grad()
    def test(self):
        """
        NOTE: test should only be called once
        """
        # set header and logger
        head = np.array(['metric'])
        for k in range(1, self.out_horizon + 1):
            head = np.append(head, [f'{k}'])
        log = np.zeros([4, self.out_horizon + 1])

        self.model.eval()
        total_iter = 0
        all_preds, all_targets = [], []
        for idx, (x, y) in enumerate(self.dataloader['test_loader'].get_iterator()):
            x = torch.Tensor(x).to(self.device)[...,:self.cfg.in_dim]
            y = torch.Tensor(y).to(self.device)[...,:self.cfg.out_dim]
            output = self.model(x)
            output = self.scaler.inverse_transform(output)  # [B, Tout, N, C=1]
            all_preds.append(output)
            all_targets.append(y[...,:self.cfg.out_dim])
        
        all_preds = torch.cat(all_preds, dim=0).squeeze()           # [all_sample, T_out, N]
        all_targets = torch.cat(all_targets, dim=0).squeeze()       # [all_sample, T_out, N]
        
        # horizon-wise evaluation
        metrics = metric(all_preds, all_targets, dim=(0, 2))        # [T_out]

        head = np.array(['metric'])
        for k in range(1, self.out_horizon + 1):
            head = np.append(head, [f'{k}'])
        head = np.append(head, ['average'])

        log = np.zeros([3, self.out_horizon])
        m_names = []

        for idx, (k, v) in enumerate(metrics.items()):
            m_names.append(k)
            log[idx] = metrics[k]

        m_names = np.expand_dims(m_names, axis=1)
        avg = np.mean(log, axis=1, keepdims=True)
        log = np.concatenate([m_names, log, avg], axis=1)           # [3, 1+T+1]

        print_log = 'Average Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:4f}'
        print_log_specific = 'MAE at 15 min: {:.4f}, MAE at 30 min: {:.4f}, MAE at 60 min: {:4f}'
        print(print_log.format(avg[0,0], avg[1,0], avg[2,0]))
        print(print_log_specific.format(metrics['mae'][2], metrics['mae'][5], metrics['mae'][11]))

        save_csv_log(self.cfg, head, log, is_create=True, file_property='result', file_name='result')

    def plot(self):
        plt.figure()
        plt.plot(self.train_loss_list, 'r', label='Train loss')
        plt.plot(self.valid_loss_list, 'g', label='Val loss')
        plt.legend()
        plt.savefig(os.path.join(self.cfg.vis_dir, self.cfg.id + '_ft.png'))
        plt.close()
