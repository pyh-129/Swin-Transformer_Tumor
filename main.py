# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import json
import random
import argparse
import datetime
import numpy as np
import sklearn
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor

os.environ['CUDA_VISIBLE_DEVICES'] = '0/'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training 去掉
    parser.add_argument("--local_rank",  required=True, type=int, help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    # dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)
## add k-folds
    num_folds=5
    fold_metrics = {
        "val_loss": {},
        "val_acc": {},
        "val_TP": {},
        "val_TN": {},
        "val_FP": {},
        "val_FN": {},
        "val_prob": {}
    }
    
    for fold_index in range(num_folds):
        dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config,num_folds,current_fold=fold_index)

        logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
        model = build_model(config)
        logger.info(str(model))

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"number of params: {n_parameters}")
        if hasattr(model, 'flops'):
            flops = model.flops()
            logger.info(f"number of GFLOPs: {flops / 1e9}")

        model.cuda()
        model_without_ddp = model

        optimizer = build_optimizer(config, model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
        loss_scaler = NativeScalerWithGradNormCount()

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
        else:
            lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

        if config.AUG.MIXUP > 0.:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif config.MODEL.LABEL_SMOOTHING > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        max_accuracy = 0.0

        if config.TRAIN.AUTO_RESUME:
            resume_file = auto_resume_helper(config.OUTPUT)
            if resume_file:
                if config.MODEL.RESUME:
                    logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
                config.defrost()
                config.MODEL.RESUME = resume_file
                config.freeze()
                logger.info(f'auto resuming from {resume_file}')
            else:
                logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

        train_loss = []
        val_loss = []
        val_acc = []
        val_TP = []
        val_TN = []
        val_FP = []
        val_FN = []
        val_prob = []

        if config.MODEL.RESUME and fold_index == 0:
            max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
            acc1, loss, TP, TN, FP, FN, prob = validate(config, data_loader_val, model)
            val_loss.append(loss)
            val_acc.append(acc1)
            val_TP.append(TP)
            val_TN.append(TN)
            val_FP.append(FP)
            val_FN.append(FN)
            val_prob.append(prob)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
            if config.EVAL_MODE:
                return
        elif fold_index > 0:
            max_accuracy = 0.0 
        if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
            load_pretrained(config, model_without_ddp, logger)
            acc1, loss, _, _, _, _, _= validate(config, data_loader_val, model)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")

        if config.THROUGHPUT_MODE:
            throughput(data_loader_val, model, logger)
            return

        logger.info("Start training")

        start_time = time.time()

       
        for metric in fold_metrics.keys():
             fold_metrics[metric][fold_index] = []

        for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS + 1):
            # data_loader_train.sampler.set_epoch(epoch)

            loss1 = train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                                    loss_scaler)
            train_loss.append(loss1)
            if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
                save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                                logger)
            #自己加print
            # print(loss1)
            acc1, loss, TP, TN, FP, FN, prob= validate(config, data_loader_val, model)
            fold_metrics["val_loss"][fold_index].append(loss)
            fold_metrics["val_acc"][fold_index].append(acc1)
            fold_metrics["val_TP"][fold_index].append(TP)
            fold_metrics["val_TN"][fold_index].append(TN)
            fold_metrics["val_FP"][fold_index].append(FP)
            fold_metrics["val_FN"][fold_index].append(FN)
            fold_metrics["val_prob"][fold_index].append(prob)
            # val_loss.append(loss)
            # val_acc.append(acc1)
            # val_TP.append(TP)
            # val_TN.append(TN)
            # val_FP.append(FP)
            # val_FN.append(FN)
            # val_prob.append(prob)

            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
            max_accuracy = max(max_accuracy, acc1)
            logger.info(f'Max accuracy: {max_accuracy:.2f}%')

        path = 'D:\Learning\Grad_0\Project\Swin-Transformer_Tumor\Swin-Transformer_Tumor\data'
        file_name = f'output_fold_{fold_index}'
        with open(fr"{path}\\{file_name}_train_loss.txt", 'w') as train_los:
            train_los.write(str(fold_metrics["train_loss"][fold_index]))
        with open(fr"{path}\\{file_name}_val_loss.txt", 'w') as val_los:
            val_los.write(str(fold_metrics["val_loss"][fold_index]))
        with open(fr"{path}\\{file_name}_val_acc.txt", 'w') as train_ac:
            train_ac.write(str(fold_metrics["val_acc"][fold_index]))
        with open(fr"{path}\\{file_name}_val_TP.txt", 'w') as val_record_tp:
            val_record_tp.write(str(fold_metrics["val_TP"][fold_index]))
        with open(fr"{path}\\{file_name}_val_TN.txt", 'w') as val_record_tn:
            val_record_tn.write(str(fold_metrics["val_TN"][fold_index]))
        with open(fr"{path}\\{file_name}_val_FP.txt", 'w') as val_record_fp:
            val_record_fp.write(str(fold_metrics["val_FP"][fold_index]))
        with open(fr"{path}\\{file_name}_val_FN.txt", 'w') as val_record_fn:
            val_record_fn.write(str(fold_metrics["val_FN"][fold_index]))
        with open(fr"{path}\\{file_name}_val_prob.txt", 'w') as val_record_prob:
            val_record_prob.write(str(fold_metrics["val_prob"][fold_index]))

    total_time = time.time() - start_time

    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    loss_list = []

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)
        loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
        loss_list.append(loss_meter.avg)
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    return sum(loss_list[1:]) / (len(loss_list)-1)


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()

    # 良性benign = 0(阳性positive), 恶性malignant = 1(阴性negative)
    TP = 0  # 真实为恶性，预测为恶性
    TN = 0  # 真实为良性，预测为良性
    FP = 0  # 真实为良性，预测为恶性
    FN = 0  # 真实为恶性，预测为良性
    prob = []

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.long().cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        for num in range(len(output)):
            if output[num][0] > output[num][1]:
                # 预测为良性
                if target[num] == 0:
                    TN += 1
                elif target[num] == 1:
                    FN += 1
            elif output[num][0] < output[num][1]:
                # 预测为恶性
                if target[num] == 0:
                    FP += 1
                elif target[num] == 1:
                    TP += 1
            array = output[num].data.cpu().numpy()
            array1 = [array[0], array[1]]
            prob.append(array1)
        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1 = accuracy(output, target, topk=(1,))

        acc1 = reduce_tensor(acc1)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(
        f' * Acc@1 {acc1_meter.avg:.3f}\t'
    )
    return acc1_meter.avg, loss_meter.avg, TP, TN, FP, FN, prob


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='gloo', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config)
