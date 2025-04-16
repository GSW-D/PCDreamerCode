import logging
import os
import torch
import utils.data_loaders
import utils.helpers
import argparse
from datetime import datetime
from tqdm import tqdm
from time import time
from tensorboardX import SummaryWriter
from core.eval_55_w3d import test_net
from utils.average_meter import AverageMeter
from torch.optim.lr_scheduler import StepLR
from utils.schedular import GradualWarmupScheduler
from utils.loss_utils import *
from utils.helpers import seprate_point_cloud
from models.PCDreamer import PCDreamer_55_W3d


def train_net(cfg):
    torch.backends.cudnn.benchmark = True

    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg).get_dataset(
        utils.data_loaders.DatasetSubset.TRAIN)
    test_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader,
                                                    batch_size=cfg.TRAIN.BATCH_SIZE,
                                                    num_workers=cfg.CONST.NUM_WORKERS,
                                                    collate_fn=utils.data_loaders.collate_fn_55,pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=False)
    val_data_loader = torch.utils.data.DataLoader(dataset=test_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TEST),
                                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                                  num_workers=cfg.CONST.NUM_WORKERS//2,
                                                  collate_fn=utils.data_loaders.collate_fn_55,
                                                  pin_memory=True,
                                                  shuffle=False)

    # Set up folders for logs and checkpoints
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', datetime.now().isoformat())
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    cfg.DIR.LOGS = output_dir % 'logs'
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)

    # Create tensorboard writers
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

    model = PCDreamer_55_W3d(cfg)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    # Create the optimizers
    def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
        decay = []
        no_decay = []
        for name, param in model.module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}]

    param_groups = add_weight_decay(model, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    print('num of no decayed params: ', len(param_groups[0]['params']))

    optimizer = torch.optim.AdamW(param_groups,
                                 lr=cfg.TRAIN.LEARNING_RATE,
                                 weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # lr scheduler
    scheduler_steplr = StepLR(optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)

    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_STEPS,
                                          after_scheduler=scheduler_steplr)


    init_epoch = 0
    best_metrics = float('inf')
    steps = 0
    BestEpoch = 0

    if 'WEIGHTS' in cfg.CONST:
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        steps = cfg.TRAIN.WARMUP_STEPS+1
        lr_scheduler = StepLR(optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
        optimizer.param_groups[0]['lr'] = cfg.TRAIN.LEARNING_RATE

        logging.info('Recover complete.')

    # Training/Testing the network
    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):
        epoch_start_time = time()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        model.train()

        total_cd_pc = 0
        total_cd_p1 = 0
        total_cd_p2 = 0

        batch_end_time = time()
        n_batches = len(train_data_loader)
        print('epoch: ', epoch_idx, 'optimizer: ', optimizer.param_groups[0]['lr'])
        with tqdm(train_data_loader) as t:
            for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(t):
                data_time.update(time() - batch_end_time)
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)
                # partial = data['partial_cloud']
                gt = data['gtcloud']
                batchsize,npoints,_ = gt.size()
                if batchsize%2 != 0:
                    gt = torch.cat([gt,gt],0)
                partial, _ = seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = torch.Tensor([-1, 0, 0]))
                mv_depth_maps = data['views'].reshape(cfg.TRAIN.BATCH_SIZE, -1, 224, 224)
                # partial_depth = torch.unsqueeze(render.get_img(partial),1)
                pcds_pred = model(partial, mv_depth_maps)

                # loss_total, losses = get_loss_PM(pcds_pred, partial, gt, sqrt=False)
                losses = model.module.get_loss(pcds_pred, gt)
                sparse_loss, dense_loss, loss_coarse, loss_fine = losses

                loss_total = sparse_loss + dense_loss

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                cd_pc_item = losses[2].item() * 1e3
                total_cd_pc += cd_pc_item
                cd_p1_item = losses[3].item() * 1e3
                total_cd_p1 += cd_p1_item
                # cd_p2_item = losses[2].item() * 1e3
                # total_cd_p2 += cd_p2_item
                n_itr = (epoch_idx - 1) * n_batches + batch_idx
                train_writer.add_scalar('Loss/Batch/cd_pc', cd_pc_item, n_itr)
                train_writer.add_scalar('Loss/Batch/cd_p1', cd_p1_item, n_itr)
                # train_writer.add_scalar('Loss/Batch/cd_p2', cd_p2_item, n_itr)
                batch_time.update(time() - batch_end_time)
                batch_end_time = time()
                t.set_description('[Epoch %d/%d][Batch %d/%d]' % (epoch_idx, cfg.TRAIN.N_EPOCHS, batch_idx + 1, n_batches))
                t.set_postfix(loss='%s' % ['%.4f' % l for l in [cd_pc_item, cd_p1_item]])

                if steps <= cfg.TRAIN.WARMUP_STEPS:
                    lr_scheduler.step()
                    steps += 1

        avg_cdc = total_cd_pc / n_batches
        avg_cd1 = total_cd_p1 / n_batches
        # avg_cd2 = total_cd_p2 / n_batches

        lr_scheduler.step()
        epoch_end_time = time()
        train_writer.add_scalar('Loss/Epoch/cd_pc', avg_cdc, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_p1', avg_cd1, epoch_idx)
        # train_writer.add_scalar('Loss/Epoch/cd_p2', avg_cd2, epoch_idx)
        logging.info(
            '[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s' %
            (epoch_idx, cfg.TRAIN.N_EPOCHS, epoch_end_time - epoch_start_time, ['%.4f' % l for l in [avg_cdc, avg_cd1]]))

        if epoch_idx % 1 == 0:
            # Validate the current model
            cd_eval = test_net(cfg, epoch_idx, val_data_loader, val_writer, model)
            # Save checkpoints
            if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0 or cd_eval < best_metrics:
                if cd_eval < best_metrics:
                    best_metrics = cd_eval
                    BestEpoch = epoch_idx
                    file_name = 'ckpt-best.pth'

                else:
                    file_name = 'ckpt-epoch-%03d.pth' % epoch_idx
                output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, output_path)

                logging.info('Saved checkpoint to %s ...' % output_path)

        logging.info('Best Performance: Epoch %d -- CD %.4f' % (BestEpoch,best_metrics))

    train_writer.close()
    val_writer.close()
