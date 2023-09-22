"""baseline training: currently support agcrn"""
import faulthandler; faulthandler.enable()

import os
import sys
import copy
import numpy as np
import torch
import argparse
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from utils import *

from data_utils.dataset_utils import load_adj
from models.load_models import get_baseline
from models.baselinetrainer import BaselineTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--mode', default='train') # mode can be train and test
    parser.add_argument('--load', action='store_true', default=False)  # for continuous training in pretraining stage
    parser.add_argument('--load_best', action='store_true', default=False)  # for continuous training in pretraining stage
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--d', type=str, default=None)
    args = parser.parse_args()

    """setup"""
    cfg = Config(args.cfg, args.d)
    set_global_seed(args.seed)
    dtype = torch.float32 if cfg.dtype == 'float32' else torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    """data"""
    if cfg.dataset == 'pems_03' or cfg.dataset == 'pems_04' or cfg.dataset == 'pems_07' or cfg.dataset == 'pems_08':
        dataset_name = cfg.dataset
    else:
        # TODO: Extra datasets such as BAY
        raise NotImplementedError
    _, _, raw_adj = load_adj(cfg.dataset, cfg.adj_type)

    """parameter"""
    if cfg.dataset == 'pems_03':
        cfg.num_nodes = 358
        cfg.node_emb_dim = 10
    elif cfg.dataset == 'pems_04':
        cfg.num_nodes = 307
        cfg.node_emb_dim = 10
    elif cfg.dataset == 'pems_07':
        cfg.num_nodes = 883
        cfg.node_emb_dim = 10
    elif cfg.dataset == 'pems_08':
        cfg.num_nodes = 170
        cfg.node_emb_dim = 2

    """model"""
    model = get_baseline(cfg, adj=raw_adj, device=device).to(dtype).to(device)

    """pretrainer setup"""
    assert cfg.pipeline == 'baseline', "Only used for baseline training/evaluation!"

    loss_fn = masked_mae
    trainer = BaselineTrainer(
        dataset=dataset_name,
        model=model,
        adj=raw_adj,
        loss=loss_fn,
        cfg=cfg,
        batch_size=cfg.batch_size,
        learning_rate=cfg.pt_lr,
        weight_decay=cfg.pt_wd,
        opt_name=cfg.opt_name,
    )

    """baseline load && train"""
    start_epoch = 0
    best_valid_loss = float('inf')
    wait = 0
    print(">>> baseline trainer on:", device)
    print(">>> baseline trainer params: {:.2f}M".format(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0))

    if args.load:
        # For continuous baseline training
        if args.load_best:
            file_name = 'ckpt_' + cfg.id + '_bl_best.pth.tar'
        else:
            file_name = 'ckpt_' + cfg.id + '_bl_last.pth.tar'
        trainer.load(os.path.join(cfg.model_dir, file_name))
        start_epoch = trainer.epoch
        best_valid_loss = trainer.best_valid_loss

    # Train baseline
    if args.mode != "test":
        
        for epoch in range(start_epoch, cfg.train_epoch):
            ret_log = np.array([epoch + 1])
            head = np.array(['epoch'])

            lr, epoch_loss, epoch_loss_info, dt = trainer.train()

            ret_log = np.append(ret_log, [lr, dt, epoch_loss])
            head = np.append(head, ['lr', 'dt', 't_l'])

            for key, value in epoch_loss_info.items():
                head = np.append(head, key)
                ret_log = np.append(ret_log, value)

            # validation
            valid_loss_info = trainer.valid()

            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Valid Loss: {:.04f}'
            print(log.format(epoch, epoch_loss, valid_loss_info['mae']))

            # update log file and save checkpoint
            is_create = False
            if not args.load:
                if epoch == 0:
                    is_create = True
            save_csv_log(cfg, head, ret_log, is_create, file_property='log', file_name=cfg.id + '_bl_log')
            
            # patience, checkpoint, info saving
            if valid_loss_info['mae'] < best_valid_loss:
                is_best = True
                best_valid_loss = valid_loss_info['mae']
                trainer.best_valid_loss = best_valid_loss
                wait = 0
            else:
                is_best = False
                wait += 1
            file_name = ['ckpt_' + cfg.id + '_bl_best.pth.tar', 'ckpt_' + cfg.id + '_bl_last.pth.tar']
            save_ckpt(cfg, trainer, is_best=is_best, file_name=file_name)

            if wait == cfg.patience:
                print(">>> early stopping baseline trainer...")
                break

            # plotting
            trainer.plot()


    print('All training end, compute final stats...')
    ## load best
    file_name = 'ckpt_' + cfg.id + '_bl_best.pth.tar'
    trainer.load(os.path.join(cfg.model_dir, file_name))
    trainer.test()

if __name__ == '__main__':
    main()