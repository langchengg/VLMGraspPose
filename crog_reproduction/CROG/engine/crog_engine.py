import os
import time
from contextlib import nullcontext
from tqdm import tqdm

import cv2
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
from loguru import logger
from utils.device import move_to_device, record_memory_sample
from utils.misc import (AverageMeter, ProgressMeter, concat_all_gather, trainMetricGPU)
from utils.grasp_eval import (detect_grasps, calculate_jacquard_index, visualization)
from utils.grasp_metrics import binary_mask_iou, load_raw_binary_target_mask


def _freeze_batch_norm_1d(model):
    """Use running statistics for singleton micro-batches without freezing affine weights."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()


def _distributed_mean(tensor):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor)
        tensor /= dist.get_world_size()
    return tensor


def _model_device(model, args):
    configured = getattr(args, "device", None)
    if configured is not None:
        return configured if isinstance(configured, torch.device) else torch.device(configured)
    return next(model.parameters()).device


def _as_metric_tensor(values, device):
    return torch.from_numpy(values).float().to(device)


def _move_crog_inputs(data, device):
    values = move_to_device(
        (
            data["img"], data["word_vec"], data["mask"],
            data["grasp_masks"]["qua"], data["grasp_masks"]["sin"],
            data["grasp_masks"]["cos"], data["grasp_masks"]["wid"],
        ),
        device,
    )
    image, text, ins_mask, qua, sin, cos, wid = values
    return (
        image, text, ins_mask.unsqueeze(1), qua.unsqueeze(1), sin.unsqueeze(1),
        cos.unsqueeze(1), wid.unsqueeze(1),
    )


def _record_training_memory(args, device, step, total_batches):
    samples = getattr(args, "memory_samples", None)
    if samples is None:
        return
    interval = max(1, int(getattr(args, "memory_sample_interval", 1000)))
    if step == 1 or step == total_batches or step % interval == 0:
        sample = record_memory_sample(samples, device, "training", step)
        if sample:
            logger.info(
                "MPS memory step={}/{} allocated={:.1f}MB driver={:.1f}MB",
                step,
                total_batches,
                sample["allocated_bytes"] / (1024 ** 2),
                sample["driver_bytes"] / (1024 ** 2),
            )

def train_with_grasp(train_loader, model, optimizer, scheduler, scaler, epoch, args):
    batch_time = AverageMeter('Batch', ':2.2f')
    data_time = AverageMeter('Data', ':2.2f')
    lr = AverageMeter('Lr', ':1.6f')
    loss_meter = AverageMeter('Loss', ':2.4f')
    qua_loss_metter = AverageMeter('Loss_qua', ':2.4f')
    sin_loss_metter = AverageMeter('Loss_sin', ':2.4f')
    cos_loss_metter = AverageMeter('Loss_cos', ':2.4f')
    wid_loss_metter = AverageMeter('Loss_wid', ':2.4f')
    iou_meter = AverageMeter('IoU', ':2.2f')
    pr_meter = AverageMeter('Prec@50', ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [
            batch_time, data_time, lr, loss_meter, 
            qua_loss_metter, sin_loss_metter, cos_loss_metter, wid_loss_metter, 
            iou_meter, pr_meter
        ],
        prefix="Training: Epoch=[{}/{}] ".format(epoch, args.epochs))

    model.train()
    if getattr(args, "freeze_bn1d_stats", False):
        _freeze_batch_norm_1d(model)

    device = _model_device(model, args)
    accumulation_steps = max(1, int(getattr(args, "accumulation_steps", 1)))
    total_batches = len(train_loader)
    optimizer.zero_grad()
    end = time.time()

    # size_list = [320, 352, 384, 416, 448, 480, 512]
    # idx = np.random.choice(len(size_list))
    # new_size = size_list[idx]

    for i, data in enumerate(train_loader):
        # image, target, text = data
        # ins_mask, grasp_quality_mask, grasp_sin_mask, grasp_cos_mask, grasp_width_mask = target
        
        (image, text, ins_mask, grasp_qua_mask, grasp_sin_mask,
         grasp_cos_mask, grasp_wid_mask) = move_to_device(
            (
                data["img"], data["word_vec"], data["mask"],
                data["grasp_masks"]["qua"], data["grasp_masks"]["sin"],
                data["grasp_masks"]["cos"], data["grasp_masks"]["wid"],
            ),
            device,
        )
        
        
        data_time.update(time.time() - end)
        ins_mask = ins_mask.unsqueeze(1)
        grasp_qua_mask = grasp_qua_mask.unsqueeze(1)
        grasp_sin_mask = grasp_sin_mask.unsqueeze(1)
        grasp_cos_mask = grasp_cos_mask.unsqueeze(1)
        grasp_wid_mask = grasp_wid_mask.unsqueeze(1)

        # # multi-scale training
        # image = F.interpolate(image, size=(new_size, new_size), mode='bilinear')

        group_start = (i // accumulation_steps) * accumulation_steps
        group_size = min(accumulation_steps, total_batches - group_start)
        should_step = (i + 1) % accumulation_steps == 0 or (i + 1) == total_batches
        sync_context = model.no_sync if hasattr(model, "no_sync") and not should_step else nullcontext
        autocast_context = amp.autocast if scaler is not None else nullcontext

        with sync_context():
            with autocast_context():
                pred, target, loss, loss_dict = model(
                    image, text, ins_mask, grasp_qua_mask, grasp_sin_mask,
                    grasp_cos_mask, grasp_wid_mask)
            if not torch.isfinite(loss).all():
                raise FloatingPointError(
                    "Non-finite loss at epoch={} iteration={}".format(epoch, i + 1)
                )
            loss_value = float(loss.detach().item())
            args.loss_finite = True
            args.loss_min = min(getattr(args, "loss_min", loss_value), loss_value)
            args.loss_max = max(getattr(args, "loss_max", loss_value), loss_value)
            backward_loss = loss / group_size
            if scaler is not None:
                scaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()
        
        ins_mask_pred = pred[0]
        ins_mask_target = target[0]

        if should_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
            if args.max_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            checkpoint_callback = getattr(args, "mid_epoch_checkpoint_callback", None)
            if checkpoint_callback is not None:
                checkpoint_callback(epoch, i + 1, total_batches)

        _record_training_memory(args, device, i + 1, total_batches)

        # metric
        iou, pr5 = trainMetricGPU(ins_mask_pred, ins_mask_target, 0.35, 0.5)
        reduced_loss = _distributed_mean(loss.detach().clone())
        iou = _distributed_mean(iou)
        pr5 = _distributed_mean(pr5)

        loss_meter.update(reduced_loss.item(), image.size(0))
        qua_loss_metter.update(loss_dict["m_qua"], image.size(0))
        sin_loss_metter.update(loss_dict["m_sin"], image.size(0))
        cos_loss_metter.update(loss_dict["m_cos"], image.size(0))
        wid_loss_metter.update(loss_dict["m_wid"], image.size(0))
        iou_meter.update(iou.item(), image.size(0))
        pr_meter.update(pr5.item(), image.size(0))
        lr.update(scheduler.get_last_lr()[-1])
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i + 1)
            # if dist.get_rank() in [-1, 0]:
            #     wandb.log(
            #         {
            #             "time/batch": batch_time.val,
            #             "time/data": data_time.val,
            #             "training/lr": lr.val,
            #             "training/loss": loss_meter.val,
            #             "training/loss_qua": qua_loss_metter.val,
            #             "training/loss_sin": sin_loss_metter.val,
            #             "training/loss_cos": cos_loss_metter.val,
            #             "training/loss_wid": wid_loss_metter.val,
            #             "training/iou": iou_meter.val,
            #             "training/prec@50": pr_meter.val,
            #         },
            #         step=epoch * len(train_loader) + (i + 1))


@torch.no_grad()
def validate_with_grasp(val_loader, model, epoch, args):
    def inverse(img, mat, w, h):
        inv_img = cv2.warpAffine(img, mat, (w, h),
                                    flags=cv2.INTER_CUBIC,
                                    borderValue=0.)
        return inv_img

    iou_list = []
    num_correct_grasps = 0
    num_total_grasps = 0
    model.eval()
    time.sleep(2)
    device = _model_device(model, args)

    num_grasps = [1,5]
    num_correct_grasps = [0, 0]
    num_total_grasps = [0, 0]

    pbar = tqdm(val_loader)
    for data in pbar:
        # data
        image = data["img"]
        text = data["word_vec"]
        ins_mask = data["mask"]
        grasp_qua_mask = data["grasp_masks"]["qua"]
        grasp_sin_mask = data["grasp_masks"]["sin"]
        grasp_cos_mask = data["grasp_masks"]["cos"]
        grasp_wid_mask = data["grasp_masks"]["wid"]
        inverse_matrix = data["inverse"]
        ori_sizes = data["ori_size"]
        grasp_targets = data["grasps"]
        mask_paths = data["mask_path"]
        object_ids = data["objID"]
        
        (image, text, ins_mask, grasp_qua_mask, grasp_sin_mask,
         grasp_cos_mask, grasp_wid_mask) = _move_crog_inputs(data, device)
        
        # inference & get predictions from model
        pred, target = model(image, text, ins_mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask)
        
        # predictions
        ins_mask_preds = pred[0]
        grasp_qua_mask_preds = pred[1]
        grasp_sin_mask_preds = pred[2]
        grasp_cos_mask_preds = pred[3]
        grasp_wid_mask_preds = pred[4]
        
        # targets
        ins_mask_targets = target[0]
        grasp_qua_mask_targets = target[1]
        grasp_sin_mask_targets = target[2]
        grasp_cos_mask_targets = target[3]
        grasp_wid_mask_targets = target[4]
        
        # Interpolate the predicted ins mask to the same size of input image
        ins_mask_preds = torch.sigmoid(ins_mask_preds)
        grasp_qua_mask_preds = torch.sigmoid(grasp_qua_mask_preds)
        grasp_wid_mask_preds = torch.sigmoid(grasp_wid_mask_preds)
        
        if ins_mask_preds.shape[-2:] != image.shape[-2:]:
            ins_mask_preds = F.interpolate(ins_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)

            grasp_qua_mask_preds = F.interpolate(grasp_qua_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)
            
            grasp_sin_mask_preds = F.interpolate(grasp_sin_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)
            
            grasp_cos_mask_preds = F.interpolate(grasp_cos_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)
            
            grasp_wid_mask_preds = F.interpolate(grasp_wid_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)
        
        # iterate over the whole batch
        for idx in range(ins_mask_preds.shape[0]):
            inv_mat = inverse_matrix[idx]
            ori_size = ori_sizes[idx]
            h, w = ori_size
            
            ins_mask_pred = ins_mask_preds[idx].cpu().numpy()
            grasp_qua_mask_pred = grasp_qua_mask_preds[idx].squeeze().cpu().numpy()
            grasp_sin_mask_pred = grasp_sin_mask_preds[idx].squeeze().cpu().numpy()
            grasp_cos_mask_pred = grasp_cos_mask_preds[idx].squeeze().cpu().numpy()
            grasp_wid_mask_pred = grasp_wid_mask_preds[idx].squeeze().cpu().numpy()
            
            grasp_target = grasp_targets[idx]
            grasp_qua_mask_target = grasp_qua_mask_targets[idx].squeeze().cpu().numpy()
            grasp_sin_mask_target = grasp_sin_mask_targets[idx].squeeze().cpu().numpy()
            grasp_cos_mask_target = grasp_cos_mask_targets[idx].squeeze().cpu().numpy()
            grasp_wid_mask_target = grasp_wid_mask_targets[idx].squeeze().cpu().numpy()
            
            # Inverse to original size
            ins_mask_pred = inverse(ins_mask_pred, inv_mat, w, h)
            ins_mask_pred = (ins_mask_pred > 0.35)
            grasp_qua_mask_pred = inverse(grasp_qua_mask_pred, inv_mat, w, h)
            grasp_sin_mask_pred = inverse(grasp_sin_mask_pred, inv_mat, w, h)
            grasp_cos_mask_pred = inverse(grasp_cos_mask_pred, inv_mat, w, h)
            grasp_wid_mask_pred = inverse(grasp_wid_mask_pred, inv_mat, w, h)
            
            grasp_qua_mask_target = inverse(grasp_qua_mask_target, inv_mat, w, h)
            grasp_sin_mask_target = inverse(grasp_sin_mask_target, inv_mat, w, h)
            grasp_cos_mask_target = inverse(grasp_cos_mask_target, inv_mat, w, h)
            grasp_wid_mask_target = inverse(grasp_wid_mask_target, inv_mat, w, h)
            
            # Evaluate against the original-resolution binary instance mask.
            raw_gt_mask = load_raw_binary_target_mask(mask_paths[idx], object_ids[idx])
            iou = binary_mask_iou(ins_mask_pred, raw_gt_mask)
            iou_list.append(iou)
            
            # Calculate grasp configurations
            for i in range(len(num_grasps)):
                num_g = num_grasps[i]
                grasp_preds, _ = detect_grasps(grasp_qua_mask_pred, grasp_sin_mask_pred, grasp_cos_mask_pred, grasp_wid_mask_pred, num_g)

                j_index = calculate_jacquard_index(grasp_preds, grasp_target)
                
                num_correct_grasps[i] += j_index
                num_total_grasps[i] += 1
    
    J_index = [0, 0]
    for i in range(len(num_grasps)):
        J_index[i] = num_correct_grasps[i]/num_total_grasps[i]
            
    iou_list = np.stack(iou_list)
    iou_list = _as_metric_tensor(iou_list, image.device)
    iou_list = concat_all_gather(iou_list)
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    iou = iou_list.mean()
    prec = {}
    temp = '  '
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres * 10)
        value = prec_list[i].item()
        prec[key] = value
        temp += "{}: {:.2f}  ".format(key, 100. * value)
    head = 'Evaluation: Epoch=[{}/{}]  IoU={:.2f}  J_index@1: {:.2f}  J_index@5: {:.2f}'.format(
        epoch, args.epochs, 100. * iou.item(), 100. * J_index[0], 100. * J_index[1])
    logger.info(head + temp)
    return iou.item(), prec, J_index


@torch.no_grad()
def validate_without_grasp(val_loader, model, epoch, args):
    def inverse(img, mat, w, h):
        inv_img = cv2.warpAffine(img, mat, (w, h),
                                    flags=cv2.INTER_CUBIC,
                                    borderValue=0.)
        return inv_img

    iou_list = []
    num_correct_grasps = 0
    num_total_grasps = 0
    model.eval()
    time.sleep(2)
    device = _model_device(model, args)

    num_grasps = [1,5]
    num_correct_grasps = [0, 0]
    num_total_grasps = [0, 0]

    pbar = tqdm(val_loader)
    for data in pbar:
        # data
        image = data["img"]
        text = data["word_vec"]
        ins_mask = data["mask"]
        grasp_qua_mask = data["grasp_masks"]["qua"]
        grasp_sin_mask = data["grasp_masks"]["sin"]
        grasp_cos_mask = data["grasp_masks"]["cos"]
        grasp_wid_mask = data["grasp_masks"]["wid"]
        inverse_matrix = data["inverse"]
        ori_sizes = data["ori_size"]
        grasp_targets = data["grasps"]
        mask_paths = data["mask_path"]
        object_ids = data["objID"]
        
        (image, text, ins_mask, grasp_qua_mask, grasp_sin_mask,
         grasp_cos_mask, grasp_wid_mask) = _move_crog_inputs(data, device)
        
        # inference & get predictions from model
        pred, ins_mask_targets = model(image, text, ins_mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask)
        
        # Interpolate the predicted ins mask to the same size of input image
        ins_mask_preds = torch.sigmoid(pred)
        if ins_mask_preds.shape[-2:] != image.shape[-2:]:
            ins_mask_preds = F.interpolate(ins_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)
        
        # iterate over the whole batch
        for idx in range(ins_mask_preds.shape[0]):
            inv_mat = inverse_matrix[idx]
            ori_size = ori_sizes[idx]
            h, w = ori_size
            
            ins_mask_pred = ins_mask_preds[idx].squeeze().cpu().numpy()
            # Inverse to original size
            ins_mask_pred = inverse(ins_mask_pred, inv_mat, w, h)
            ins_mask_pred = (ins_mask_pred > 0.35)

            raw_gt_mask = load_raw_binary_target_mask(mask_paths[idx], object_ids[idx])
            iou = binary_mask_iou(ins_mask_pred, raw_gt_mask)
            iou_list.append(iou)
    
    J_index = [0, 0]
    
    iou_list = np.stack(iou_list)
    iou_list = _as_metric_tensor(iou_list, image.device)
    iou_list = concat_all_gather(iou_list)
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    iou = iou_list.mean()
    prec = {}
    temp = '  '
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres * 10)
        value = prec_list[i].item()
        prec[key] = value
        temp += "{}: {:.2f}  ".format(key, 100. * value)
    head = 'Evaluation: Epoch=[{}/{}]  IoU={:.2f}  J_index@1: {:.2f}  J_index@5: {:.2f}'.format(
        epoch, args.epochs, 100. * iou.item(), 100. * J_index[0], 100. * J_index[1])
    logger.info(head + temp)
    return iou.item(), prec, J_index



@torch.no_grad()
def inference_with_grasp(test_loader, model, args):
    def inverse(img, mat, w, h):
        inv_img = cv2.warpAffine(img, mat, (w, h),
                                    flags=cv2.INTER_CUBIC,
                                    borderValue=0.)
        return inv_img

    iou_list = []
    num_correct_grasps = 0
    num_total_grasps = 0
    model.eval()
    time.sleep(2)
    device = _model_device(model, args)
    
    num_grasps = [1,5]
    num_correct_grasps = [0, 0]
    num_total_grasps = [0, 0]
    
    tbar = tqdm(test_loader, desc='Inference:', ncols=100)
    for cnt, data in enumerate(tbar):
        
        # data
        image = data["img"]
        text = data["word_vec"]
        ins_mask = data["mask"]
        grasp_qua_mask = data["grasp_masks"]["qua"]
        grasp_sin_mask = data["grasp_masks"]["sin"]
        grasp_cos_mask = data["grasp_masks"]["cos"]
        grasp_wid_mask = data["grasp_masks"]["wid"]
        inverse_matrix = data["inverse"]
        ori_sizes = data["ori_size"]
        grasp_targets = data["grasps"]
        sentences = data["sentence"]
        img_paths = data["img_path"]
        mask_paths = data["mask_path"]
        object_ids = data["objID"]
        
        (image, text, ins_mask, grasp_qua_mask, grasp_sin_mask,
         grasp_cos_mask, grasp_wid_mask) = _move_crog_inputs(data, device)
        
        # inference & get predictions from model
        pred, target = model(image, text, ins_mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask)
        
        # predictions
        ins_mask_preds = pred[0]
        grasp_qua_mask_preds = pred[1]
        grasp_sin_mask_preds = pred[2]
        grasp_cos_mask_preds = pred[3]
        grasp_wid_mask_preds = pred[4]
        
        # targets
        ins_mask_targets = target[0]
        grasp_qua_mask_targets = target[1]
        grasp_sin_mask_targets = target[2]
        grasp_cos_mask_targets = target[3]
        grasp_wid_mask_targets = target[4]
        
        # Interpolate the predicted ins mask to the same size of input image
        ins_mask_preds = torch.sigmoid(ins_mask_preds)
        grasp_qua_mask_preds = torch.sigmoid(grasp_qua_mask_preds)
        grasp_wid_mask_preds = torch.sigmoid(grasp_wid_mask_preds)
        
        if ins_mask_preds.shape[-2:] != image.shape[-2:]:
            ins_mask_preds = F.interpolate(ins_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)

            grasp_qua_mask_preds = F.interpolate(grasp_qua_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)
            
            grasp_sin_mask_preds = F.interpolate(grasp_sin_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)
            
            grasp_cos_mask_preds = F.interpolate(grasp_cos_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)
            
            grasp_wid_mask_preds = F.interpolate(grasp_wid_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)
        

        # iterate over the whole batch
        for idx in range(ins_mask_preds.shape[0]):
            inv_mat = inverse_matrix[idx]
            ori_size = ori_sizes[idx]
            h, w = ori_size
            sent = sentences[idx]
            img_path = img_paths[idx]
            
            ins_mask_pred = ins_mask_preds[idx].cpu().numpy()
            grasp_qua_mask_pred = grasp_qua_mask_preds[idx].squeeze().cpu().numpy()
            grasp_sin_mask_pred = grasp_sin_mask_preds[idx].squeeze().cpu().numpy()
            grasp_cos_mask_pred = grasp_cos_mask_preds[idx].squeeze().cpu().numpy()
            grasp_wid_mask_pred = grasp_wid_mask_preds[idx].squeeze().cpu().numpy()
            
            grasp_target = grasp_targets[idx]
            grasp_qua_mask_target = grasp_qua_mask_targets[idx].squeeze().cpu().numpy()
            grasp_sin_mask_target = grasp_sin_mask_targets[idx].squeeze().cpu().numpy()
            grasp_cos_mask_target = grasp_cos_mask_targets[idx].squeeze().cpu().numpy()
            grasp_wid_mask_target = grasp_wid_mask_targets[idx].squeeze().cpu().numpy()
            
            # Inverse to original size
            ins_mask_pred = inverse(ins_mask_pred, inv_mat, w, h)
            ins_mask_pred = (ins_mask_pred > 0.35)
            grasp_qua_mask_pred = inverse(grasp_qua_mask_pred, inv_mat, w, h)
            grasp_sin_mask_pred = inverse(grasp_sin_mask_pred, inv_mat, w, h)
            grasp_cos_mask_pred = inverse(grasp_cos_mask_pred, inv_mat, w, h)
            grasp_wid_mask_pred = inverse(grasp_wid_mask_pred, inv_mat, w, h)
            
            grasp_qua_mask_target = inverse(grasp_qua_mask_target, inv_mat, w, h)
            grasp_sin_mask_target = inverse(grasp_sin_mask_target, inv_mat, w, h)
            grasp_cos_mask_target = inverse(grasp_cos_mask_target, inv_mat, w, h)
            grasp_wid_mask_target = inverse(grasp_wid_mask_target, inv_mat, w, h)
            
            # Calculate IoU against raw binary GT (no GT interpolation halo).
            raw_gt_mask = load_raw_binary_target_mask(mask_paths[idx], object_ids[idx])
            iou = binary_mask_iou(ins_mask_pred, raw_gt_mask)
            iou_list.append(iou)
            
            # Calculate grasp configurations
            for i in range(len(num_grasps)):
                num_g = num_grasps[i]
                grasp_preds, grasp_ang_mask_pred = detect_grasps(grasp_qua_mask_pred, grasp_sin_mask_pred, grasp_cos_mask_pred, grasp_wid_mask_pred, num_g)

                j_index = calculate_jacquard_index(grasp_preds, grasp_target)
                
                num_correct_grasps[i] += j_index
                num_total_grasps[i] += 1
                
                # Visualization
                if args.visualize:
                    img_bgr = cv2.imread(img_path)
                    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    visualization(img, ins_mask_pred, (grasp_qua_mask_pred, grasp_ang_mask_pred, grasp_wid_mask_pred), grasp_preds, sent, save_path=os.path.join("./results", args.exp_name, f"results_{cnt}_{num_g}_grasps.png"))
                
    J_index = [0, 0]
    for i in range(len(num_grasps)):
        J_index[i] = num_correct_grasps[i]/num_total_grasps[i]
            
    iou_list = np.stack(iou_list)
    iou_list = _as_metric_tensor(iou_list, image.device)
    # print(iou_list)
    # iou_list = concat_all_gather(iou_list)
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    iou = iou_list.mean()
    prec = {}
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres*10)
        value = prec_list[i].item()
        prec[key] = value
    logger.info('IoU={:.2f}'.format(100.*iou.item()))
    for k, v in prec.items():
        logger.info('{}: {:.2f}.'.format(k, 100.*v))
    logger.info("J@1: {:.2f}, J@5: {:.2f}".format(100. * J_index[0], 100. * J_index[1]))

    return iou.item(), prec, J_index
