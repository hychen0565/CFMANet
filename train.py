import os
import random
import time
import cv2
import numpy as np
import argparse
import os.path as osp
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from model.mfanet import mfanet
from util.dataset_fss import SemData as dataset
from util import transform, config
from util.util import *


cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

def get_parser():
    parser = argparse.ArgumentParser(
        description='PyTorch Few-Shot Semantic Segmentation'
    )

    parser.add_argument('--arch', type=str, default='MFANet')

    # None 表示命令行没有指定，继续使用 YAML 配置
    parser.add_argument(
        '--eval_only',
        dest='eval_only',
        action='store_true',
        default=None
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/fold0_resnet50.yaml',
        help='config file'
    )

    cli_args = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(cli_args.config)

    # 只有命令行显式传入时才覆盖 YAML
    if cli_args.eval_only is not None:
        cfg.eval_only = cli_args.eval_only
    elif not hasattr(cfg, 'eval_only'):
        cfg.eval_only = False

    return cfg


def get_model(args):
    model = mfanet(args)
    optimizer = model.get_optim(args, args.base_lr)
    freeze_modules(model)
    model = model.cuda()

    # Resume
    get_save_path(args)
    check_makedirs(args.snapshot_path)
    check_makedirs(args.result_path)

    if args.resume:
        resume_path = osp.join(args.snapshot_path, args.resume)
        if os.path.isfile(resume_path):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            new_param = checkpoint['state_dict']
            try:
                model.load_state_dict(new_param)
            except RuntimeError:
                # 1GPU loads mGPU model
                for key in list(new_param.keys()):
                    new_param[key[7:]] = new_param.pop(key)
                model.load_state_dict(new_param)
            optimizer.load_state_dict(checkpoint['optimizer'])
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(resume_path))
    return model, optimizer


def main_process():
    return True


def build_val_loader(args):
    """构建验证集 DataLoader（训练与验证共用同一套归一化参数）。"""
    value_scale = 255
    mean = [item * value_scale for item in (0.485, 0.456, 0.406)]
    std = [item * value_scale for item in (0.229, 0.224, 0.225)]
    if args.resized_val:
        val_transform = transform.Compose([
            transform.Resize(h=args.val_size, w=args.val_size),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])
    else:
        val_transform = transform.Compose([
            transform.test_Resize(size=args.val_size),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])
    val_data = dataset(split=args.split, shot=args.shot, data_root=args.data_root, data_list=args.val_list,
                       transform=val_transform, mode='val', data_set=args.data_set)
    return torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False,
                                       num_workers=args.workers, pin_memory=True)


def main():
    global args, logger, writer
    args = get_parser()
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)

    logger.info(" -------------------- creating model-------------------")
    model, optimizer = get_model(args)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Model created: %s | total params %.2fM | trainable params %.2fM',
                model.__class__.__name__, total_params / 1e6, trainable_params / 1e6)

    # ---------------------- DATASET ----------------------
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    # Train
    train_transform = [
        transform.Resize(args.train_h, args.train_w),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.padding_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)]
    train_transform = transform.Compose(train_transform)
    train_data = dataset(split=args.split, shot=args.shot, data_root=args.data_root, \
                         data_list=args.train_list, transform=train_transform, mode='train', data_set=args.data_set)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                               num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                                               drop_last=True)

    if args.evaluate:
        val_loader = build_val_loader(args)

    # 仅评估模式：需要 YAML 中 evaluate=True，并通过 resume 指定权重。
    if getattr(args, 'eval_only', False):
        if not args.evaluate:
            raise ValueError('--eval_only requires evaluate: True in the YAML configuration')
        if not args.resume:
            raise ValueError('--eval_only requires a checkpoint filename in YAML field TRAIN.resume')
        validate(val_loader, model)
        if main_process():
            writer.flush()
            writer.close()
        return

    # ---------------------- TRAINVAL ----------------------
    global best_miou, best_FBiou, best_piou, best_epoch, keep_epoch, val_num
    global best_miou_m
    best_miou = 0.
    best_FBiou = 0.
    best_piou = 0.
    best_epoch = 0
    val_num = 0
    max_iou = 0.
    max_fbiou = 0
    best_epoch = 0
    filename = 'SD-AANet.pth'

    for epoch in range(args.start_epoch, args.epochs):
        if args.fix_random_seed_val:
            torch.cuda.manual_seed(args.manual_seed + epoch)
            np.random.seed(args.manual_seed + epoch)
            torch.manual_seed(args.manual_seed + epoch)
            torch.cuda.manual_seed_all(args.manual_seed + epoch)
            random.seed(args.manual_seed + epoch)

        epoch_log = epoch + 1
        # loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, optimizer, epoch)
        loss_train, mIoU_train, mAcc_train, allAcc_train, mani_loss_train = train(train_loader, model, optimizer, epoch)
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)
            writer.add_scalar('mani_loss_train', mani_loss_train, epoch_log)

        if args.evaluate:
            loss_val, mIoU_val, mAcc_val, allAcc_val, class_miou, class_iou_class = validate(val_loader, model)
            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('class_miou_val', class_miou, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
                if class_miou > max_iou:
                    max_iou = class_miou
                    best_epoch = epoch
                    if os.path.exists(filename):
                        os.remove(filename)
                    filename = args.save_path + '/train_epoch_' + str(epoch) + '_' + str(max_iou) + '.pth'
                    logger.info('Saving checkpoint to: ' + filename)
                    torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict()}, filename)
                if mIoU_val > max_fbiou:
                    max_fbiou = mIoU_val
                logger.info(
                    'Best Epoch {:.1f}, Best IoU {:.4f} Best FB-IoU {:4F}'.format(best_epoch, max_iou, max_fbiou))

                #写入带轮数标记的log文件
                log_file_path = os.path.join(args.save_path, 'val_results_log.txt')
                prefix = "[{}/{}] ".format(epoch, args.epochs - 1)
                log_line = "meanIoU---Val result: mIoU {:.4f}. ".format(class_miou)
                for i in range(len(class_iou_class)):
                    log_line += "Class_{} Result: iou {:.4f}. ".format(i + 1, class_iou_class[i])
                log_line += "FBIoU---Val result: mIoU {:.4f}".format(mIoU_val)
                with open(log_file_path, 'a') as f:
                    f.write(prefix + log_line + '\n')

        if main_process():
            writer.flush()

    if main_process():
        writer.flush()
        writer.close()


def train(train_loader, model, optimizer, epoch):
    logger.info('>>>>>>>>>>>>>>>> Start Train <<<<<<<<<<<<<<<<<<')
    batch_time = AverageMeter()
    data_time = AverageMeter()

    manifold_loss_meter = AverageMeter()

    main_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    val_time = 0.
    max_iter = args.epochs * len(train_loader)
    print('Warmup: {}'.format(args.warmup))

    for i, (input, target, s_input, s_mask, subcls) in enumerate(train_loader):
        data_time.update(time.time() - end)
        current_iter = epoch * len(train_loader) + i + 1
        index_split = -1
        if args.base_lr > 1e-6:
            poly_learning_rate(optimizer, args.base_lr, current_iter, max_iter, power=args.power,
                               index_split=index_split, warmup=args.warmup, warmup_step=len(train_loader) // 2)

        s_input = s_input.cuda(non_blocking=True)
        s_mask = s_mask.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output, total_loss, seg_loss, m_loss = model(s_x=s_input, s_y=s_mask, x=input, y=target)
        # output, main_loss = model(s_x=s_input, s_y=s_mask, x=input, y=target)

        loss = total_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = input.size(0)
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)

        #更新Manifold Loss 监控器
        manifold_loss_meter.update(m_loss.item(), n)
        main_loss_meter.update(seg_loss.item(), n)  # 分割损失
        loss_meter.update(loss.item(), n)  # 总损失
        batch_time.update(time.time() - end)
        end = time.time()

        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        # if (i + 1) % args.print_freq == 0 and main_process():
        #     logger.info('Epoch: [{}/{}][{}/{}] '
        #                 'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
        #                 'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
        #                 'Remain {remain_time} '
        #                 'MainLoss {main_loss_meter.val:.4f} '
        #                 'Loss {loss_meter.val:.4f} '
        #                 'Accuracy {accuracy:.4f}.'.format(epoch, args.epochs - 1, i + 1, len(train_loader),
        #                                                   batch_time=batch_time, data_time=data_time,
        #                                                   remain_time=remain_time, main_loss_meter=main_loss_meter,
        #                                                   loss_meter=loss_meter, accuracy=accuracy))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'SegLoss {main_loss_meter.val:.4f} '
                        'ManiLoss {manifold_loss_meter.val:.4f} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch, args.epochs - 1, i + 1, len(train_loader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          main_loss_meter=main_loss_meter,
                                                          manifold_loss_meter=manifold_loss_meter,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

        if main_process():
            writer.add_scalar('loss_train_batch', main_loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    if main_process():
        logger.info(
            'Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch, args.epochs, mIoU,
                                                                                           mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>end train<<<<<<<<<<<<<<<<<<<<<<<<<<')
    #return main_loss_meter.avg, mIoU, mAcc, allAcc
    return main_loss_meter.avg, mIoU, mAcc, allAcc, manifold_loss_meter.avg


def extract_subcls_indices(subcls, batch_size):
    """将 DataLoader 整理后的 subcls 转成长度为 batch_size 的一维整数数组。

    Dataset 每个 episode 只返回一个类别索引，default_collate 后为 Tensor[B]。
    这里同时兼容 Tensor、numpy.ndarray、list 与标量，避免 batch_size 变化时报错。
    """
    if batch_size <= 0:
        return np.empty((0,), dtype=np.int64)

    if torch.is_tensor(subcls):
        indices = subcls.detach().cpu().reshape(-1).numpy()
    elif isinstance(subcls, np.ndarray):
        indices = np.asarray(subcls).reshape(-1)
    elif isinstance(subcls, (list, tuple)):
        if len(subcls) > 0 and torch.is_tensor(subcls[0]):
            indices = subcls[0].detach().cpu().reshape(-1).numpy()
        else:
            indices = np.asarray(subcls).reshape(-1)
    else:
        indices = np.asarray([subcls])

    indices = indices.astype(np.int64, copy=False)
    if indices.size == 1 and batch_size > 1:
        indices = np.repeat(indices, batch_size)
    if indices.size < batch_size:
        raise ValueError(
            f'Cannot extract {batch_size} class indices from subcls; '
            f'type={type(subcls).__name__}, extracted={indices.tolist()}'
        )
    return indices[:batch_size]

def validate(val_loader, model):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    split_gap = len(val_loader.dataset.val_class)
    class_intersection_meter = [0.0] * split_gap
    class_union_meter = [0.0] * split_gap

    if args.manual_seed is not None and args.fix_random_seed_val:
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    model.eval()
    end = time.time()
    val_start = end
    test_batches = len(val_loader)

    for batch_idx, (input, target, s_input, s_mask, subcls, ori_label) in enumerate(val_loader):
        data_time.update(time.time() - end)

        s_input = s_input.cuda(non_blocking=True)
        s_mask = s_mask.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        ori_label = ori_label.cuda(non_blocking=True)

        start_time = time.time()
        with torch.inference_mode():
            output_logits = model(s_x=s_input, s_y=s_mask, x=input, y=target)
        model_time.update(time.time() - start_time)

        if args.ori_resize:
            # 将同一 batch 中不同原始尺寸统一填充到公共方形尺寸。
            original_sizes = [tuple(ori_label[idx].shape[-2:]) for idx in range(ori_label.size(0))]
            common_side = max(max(h_i, w_i) for h_i, w_i in original_sizes)
            resized_targets = []
            for sample_idx, (ori_h, ori_w) in enumerate(original_sizes):
                canvas = torch.full(
                    (common_side, common_side),
                    fill_value=args.ignore_label,
                    dtype=ori_label.dtype,
                    device=ori_label.device
                )
                canvas[:ori_h, :ori_w] = ori_label[sample_idx, :ori_h, :ori_w]
                resized_targets.append(canvas)
            target = torch.stack(resized_targets, dim=0).long()
            output_logits = F.interpolate(
                output_logits,
                size=(common_side, common_side),
                mode='bilinear',
                align_corners=True
            )

        loss = criterion(output_logits, target).mean()
        pred_mask = output_logits.argmax(dim=1)

        # 指标计算
        intersection_t, union_t, new_target_t = intersectionAndUnionGPU(
            pred_mask, target, args.classes, args.ignore_label
        )
        intersection = intersection_t.cpu().numpy()
        union = union_t.cpu().numpy()
        new_target = new_target_t.cpu().numpy()

        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(new_target)

        # 逐样本记录目标类别 IoU。
        subcls_indices = extract_subcls_indices(subcls, pred_mask.size(0))
        for sample_idx, cls_value in enumerate(subcls_indices):
            cls_idx = int(cls_value)
            if 0 <= cls_idx < split_gap:
                sample_intersection_t, sample_union_t, _ = intersectionAndUnionGPU(
                    pred_mask[sample_idx:sample_idx + 1],
                    target[sample_idx:sample_idx + 1],
                    args.classes,
                    args.ignore_label
                )
                sample_intersection = sample_intersection_t.cpu().numpy()
                sample_union = sample_union_t.cpu().numpy()
                class_intersection_meter[cls_idx] += float(sample_intersection[1])
                class_union_meter[cls_idx] += float(sample_union[1])
            else:
                logger.warning('subcls %s is outside [0, %d]', cls_idx, split_gap - 1)

        accuracy = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
        loss_meter.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if main_process() and ((batch_idx + 1) % max(1, args.print_freq) == 0 or batch_idx + 1 == test_batches):
            logger.info(
                'Test: [%d/%d] Data %.3f (%.3f) Batch %.3f (%.3f) '
                'Loss %.4f (%.4f) Accuracy %.4f',
                batch_idx + 1,
                test_batches,
                data_time.val,
                data_time.avg,
                batch_time.val,
                batch_time.avg,
                loss_meter.val,
                loss_meter.avg,
                accuracy
            )

    val_time = time.time() - val_start
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    class_iou_class = []
    for cls_idx in range(split_gap):
        cls_iou = class_intersection_meter[cls_idx] / (class_union_meter[cls_idx] + 1e-10)
        class_iou_class.append(cls_iou)
    class_miou = float(np.mean(class_iou_class)) if class_iou_class else 0.0

    logger.info('meanIoU---Val result: mIoU %.4f.', class_miou)
    for cls_idx, cls_iou in enumerate(class_iou_class):
        logger.info('Class_%d Result: iou %.4f.', cls_idx + 1, cls_iou)
    if main_process():
        logger.info('FBIoU---Val result: mIoU/mAcc/allAcc %.4f/%.4f/%.4f.', mIoU, mAcc, allAcc)
        for cls_idx in range(args.classes):
            logger.info(
                'Class_%d Result: iou/accuracy %.4f/%.4f.',
                cls_idx,
                iou_class[cls_idx],
                accuracy_class[cls_idx]
            )
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    logger.info('Total validation time: %.4f s, average inference time: %.4f s/batch',
                val_time, model_time.avg)

    return loss_meter.avg, mIoU, mAcc, allAcc, class_miou, class_iou_class

if __name__ == '__main__':
    main()
