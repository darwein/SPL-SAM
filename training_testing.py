import torch
import numpy as np
import os
from utils.visualization import show_anns, show_mask
from utils.dice_calculate import dice_calculate
import gc
from medpy import metric
from utils.utils import Logger
from mysam import SamAutomaticMaskGenerator
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import time
from utils.utils import poly_lr
from torch.nn import functional as F
from medpy.metric.binary import hd,hd95
import torch.nn as nn
import cv2
import random

def calculate_metrics(predict_image, gt_image, evaluate):
    # 将图像转换为二进制数组
    predict_image = np.array(predict_image, dtype=bool)
    gt_image = np.array(gt_image, dtype=bool)
    # 计算True Positive（TP）
    tp = np.sum(np.logical_and(predict_image, gt_image))
    # 计算True Negative（TN）
    tn = np.sum(np.logical_and(np.logical_not(predict_image), np.logical_not(gt_image)))
    # 计算False Positive（FP）
    fp = np.sum(np.logical_and(predict_image, np.logical_not(gt_image)))
    # 计算False Negative（FN）
    fn = np.sum(np.logical_and(np.logical_not(predict_image), gt_image))
    # 计算IOU（Intersection over Union）
    iou = tp / (tp + fn + fp + 1e-7)
    # 计算Dice Coefficient（Dice系数）
    dice_coefficient = 2 * tp / (2 * tp + fn + fp + 1e-7)
    # 计算Accuracy（准确率）
    accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-7)
    # 计算precision（精确率）
    precision = tp / (tp + fp + 1e-7)
    # 计算recall（召回率）
    recall = tp / (tp + fn + 1e-7)
    # 计算Sensitivity（敏感度）
    sensitivity = tp / (tp + fn + 1e-7)
    # 计算F1-score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    # 计算Specificity（特异度）
    specificity = tn / (tn + fp + 1e-7)
    if evaluate == "iou":
        return iou
    if evaluate == "dice_coefficient":
        return dice_coefficient
    if evaluate == "accuracy":
        return accuracy
    if evaluate == "precision":
        return precision
    if evaluate == "recall":
        return recall
    if evaluate == "sensitivity":
        return sensitivity
    if evaluate == "f1":
        return f1
    if evaluate == "specificity":
        return specificity

def save_test_result(save_path,name,predict):
    cur_name=name[0]
    predict = (predict * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_path,cur_name.replace('.png','_mask.png')), predict)

def visual(image, groundtruth, predict):
    B=image.size(0)
    for i in range(B):
        img = image[i].permute(1,2,0).cpu().numpy().squeeze()
        img = img.astype(np.uint8)
        gt = groundtruth[i].cpu().numpy().squeeze()
        gt = (gt * 255).astype(np.uint8)
        pre = predict[i].squeeze()
        pre = (pre * 255).astype(np.uint8)
        #同时显示三张图
        gt = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
        pre = cv2.cvtColor(pre, cv2.COLOR_GRAY2BGR)
        combined_image = np.hstack((img, gt, pre))
        cv2.imshow('Combined Images', combined_image)

        # cv2.imshow("img", img)
        # cv2.imshow("gt", gt)
        # cv2.imshow("predict", pre)
        cv2.waitKey(0)

def run_training_wholebox(img_size, mysam_model,medsam, max_num_epochs, model_save_path, optimizer, initial_lr,
        train_dataloader, val_dataloader, scaler, criterion1, criterion2,iou_loss,device):
    logger = Logger(output_folder=model_save_path)
    train_losses = []
    val_losses = []
    best_loss = 1e10
    best_dice=0

    for epoch in range(max_num_epochs):
        # update lr
        mysam_model.train()
        logger.print_to_log_file("\nepoch: ", epoch)
        optimizer.param_groups[0]['lr'] = poly_lr(epoch, max_num_epochs, initial_lr, 0.9)

        epoch_diceceloss = 0
        # train
        for step, (name,image, gt, box) in enumerate(train_dataloader):
            image = image.to(torch.float32).permute(0, 3, 1, 2).to(device)
            box = box.to(device)
            with torch.no_grad():
                image_embedding = mysam_model.image_encode(image, mask_ratio=0)
                B, _, H, W = gt.shape
                boxes_torch = torch.from_numpy(np.array([[0, 0, H, W]] * B)).float().to(device)
                sparse_embeddings, dense_embeddings = mysam_model.prompt_encoder(
                    points=None,
                    boxes=boxes_torch,
                    masks=None,)

            # predicted masks
            low_res_masks, iou_predictions = mysam_model.mask_decoder(
                image_embeddings=image_embedding,  # (B, 256, 64, 64)
                image_pe=mysam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False,)
            mask_pred = F.interpolate(low_res_masks, (img_size, img_size), mode="bilinear",
                                      align_corners=False,)
            optimizer.zero_grad()
            loss1 = criterion1(mask_pred, gt.to(device))
            loss2 = criterion2(mask_pred, gt.to(device))
            #loss3 = criterion3(mask_pred, gt.to(device))
            loss = 1 * loss1 + 1 * loss2 #+ 1 * loss3

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_diceceloss += loss.item()

        epoch_diceceloss /= step
        train_epoch_loss = epoch_diceceloss
        train_losses.append(epoch_diceceloss)

        val_epoch_loss, val_epoch_dice = run_validation_wholebox(img_size,mysam_model, val_dataloader, device, criterion1, criterion2)
        val_losses.append(val_epoch_loss)

        # if val_epoch_loss < best_loss:
        #     best_loss = val_epoch_loss
        #     torch.save(mysam_model.state_dict(), os.path.join(model_save_path, 'mysam_model_best.pth'))
        if val_epoch_dice>best_dice:
            best_dice=val_epoch_dice
            torch.save(mysam_model.state_dict(), os.path.join(model_save_path, 'mysam_model_best.pth'))

        logger.print_to_log_file("train loss : %.8f" % train_epoch_loss)
        logger.print_to_log_file("validation loss: %.8f" % val_epoch_loss)
        logger.print_to_log_file("Average global foreground Dice: %.8f" % val_epoch_dice)

    return train_losses, val_losses

def run_validation_wholebox(img_size,mysam_model, val_dataloader, device, criterion1,criterion2):
    mysam_model.eval()
    val_epoch_diceceloss = 0
    dice_list = []

    for step, (name,image, gt, box) in enumerate(val_dataloader):
        image = image.to(torch.float32).permute(0, 3, 1, 2).to(device)
        box = box.to(device)
        with torch.no_grad():
            image_embedding = mysam_model.image_encode(image, mask_ratio=0)
            B, _, H, W = gt.shape
            box_torch = torch.from_numpy(np.array([[0, 0, H, W]])).float().to(device)
            sparse_embeddings, dense_embeddings = mysam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,)
            # predicted masks
            low_res_masks, _ = mysam_model.mask_decoder(
                image_embeddings=image_embedding,  # (B, 256, 64, 64)
                image_pe=mysam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False,)
            mask_pred = F.interpolate(low_res_masks, (img_size, img_size), mode="bilinear",
                                      align_corners=False, )

            loss1 = criterion1(mask_pred, gt.to(device))
            loss2 = criterion2(mask_pred, gt.to(device))
            loss = loss1 + loss2
            val_epoch_diceceloss += loss.item()

            mask_pred = torch.sigmoid(mask_pred)
            mask_pred = mask_pred.cpu().numpy().squeeze()
            mask_pred = (mask_pred > 0.5).astype(np.uint8)

            #visual(image, gt, mask_pred)
            dice_list.append(metric.dc(mask_pred, gt.cpu().numpy().squeeze()))

    val_epoch_diceceloss /= step
    val_epoch_loss = val_epoch_diceceloss

    return val_epoch_loss, np.mean(dice_list)

def run_testing_wholebox(img_size,model,medsam, val_dataloader, device,criterion1, criterion2,save_path):  #用val数据集
    model.eval()

    iou_list = []
    dice_list = []
    accuracy_list = []
    precision_list = []
    recall_list = []
    sensitivity_list = []
    specificity_list = []
    hausdorff_distance_list = []

    benign_iou_list = []
    benign_dice_list = []
    benign_accuracy_list = []
    benign_precision_list = []
    benign_recall_list = []
    benign_sensitivity_list = []
    benign_specificity_list = []
    benign_hausdorff_distance_list = []

    malignant_iou_list = []
    malignant_dice_list = []
    malignant_accuracy_list = []
    malignant_precision_list = []
    malignant_recall_list = []
    malignant_sensitivity_list = []
    malignant_specificity_list = []
    malignant_hausdorff_distance_list = []

    small_iou_list = []
    small_dice_list = []
    small_accuracy_list = []
    small_precision_list = []
    small_recall_list = []
    small_sensitivity_list = []
    small_specificity_list = []
    small_hausdorff_distance_list = []

    medium_iou_list = []
    medium_dice_list = []
    medium_accuracy_list = []
    medium_precision_list = []
    medium_recall_list = []
    medium_sensitivity_list = []
    medium_specificity_list = []
    medium_hausdorff_distance_list = []

    large_iou_list = []
    large_dice_list = []
    large_accuracy_list = []
    large_precision_list = []
    large_recall_list = []
    large_sensitivity_list = []
    large_specificity_list = []
    large_hausdorff_distance_list = []

    for step, (name,image, gt, box) in tqdm(enumerate(val_dataloader)):
        image = image.to(torch.float32).permute(0, 3, 1, 2).to(device)
        box = box.to(device)
        with torch.no_grad():
            image_embedding = model.image_encode(image, mask_ratio=0)
            B, _, H, W = gt.shape
            box_torch = torch.from_numpy(np.array([[0, 0, H, W]])).float().to(device)
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,)
            # predicted masks
            low_res_masks, _ = model.mask_decoder(
                image_embeddings=image_embedding,  # (B, 256, 64, 64)
                image_pe=model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False,)
            mask_pred = F.interpolate(low_res_masks, (img_size, img_size), mode="bilinear",
                                      align_corners=False, )

            mask_pred = torch.sigmoid(mask_pred)
            mask_pred = mask_pred.cpu().numpy().squeeze()
            mask_pred = (mask_pred > 0.5).astype(np.uint8)
            # visual(image, gt, mask_pred)
            #save_test_result(save_path, name, mask_pred)
            #dice_list.append(metric.dc(mask_pred, gt.cpu().numpy().squeeze()))
            iou_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'iou'))
            dice_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'dice_coefficient'))
            accuracy_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'accuracy'))
            precision_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'precision'))
            recall_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'recall'))
            sensitivity_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'sensitivity'))
            specificity_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'specificity'))
            if (np.count_nonzero(mask_pred) == 0):
                mask_pred[127, 127] = 1
            hausdorff_distance_list.append(hd95(mask_pred, gt.cpu().numpy().squeeze()))

            if('benign' in name[0]):
                benign_iou_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'iou'))
                benign_dice_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'dice_coefficient'))
                benign_accuracy_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'accuracy'))
                benign_precision_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'precision'))
                benign_recall_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'recall'))
                benign_sensitivity_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'sensitivity'))
                benign_specificity_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'specificity'))
                benign_hausdorff_distance_list.append(hd95(mask_pred, gt.cpu().numpy().squeeze()))
            elif('malignant' in name[0]):
                malignant_iou_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'iou'))
                malignant_dice_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'dice_coefficient'))
                malignant_accuracy_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'accuracy'))
                malignant_precision_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'precision'))
                malignant_recall_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'recall'))
                malignant_sensitivity_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'sensitivity'))
                malignant_specificity_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'specificity'))
                malignant_hausdorff_distance_list.append(hd95(mask_pred, gt.cpu().numpy().squeeze()))

            size=torch.sum(gt[0] > 0).item()
            if(size/(256*256)<=0.05):
                small_iou_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'iou'))
                small_dice_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'dice_coefficient'))
                small_accuracy_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'accuracy'))
                small_precision_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'precision'))
                small_recall_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'recall'))
                small_sensitivity_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'sensitivity'))
                small_specificity_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'specificity'))
                small_hausdorff_distance_list.append(hd95(mask_pred, gt.cpu().numpy().squeeze()))
            elif(size/(256*256)>0.05 and size/(256*256)<=0.2):
                medium_iou_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'iou'))
                medium_dice_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'dice_coefficient'))
                medium_accuracy_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'accuracy'))
                medium_precision_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'precision'))
                medium_recall_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'recall'))
                medium_sensitivity_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'sensitivity'))
                medium_specificity_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'specificity'))
                medium_hausdorff_distance_list.append(hd95(mask_pred, gt.cpu().numpy().squeeze()))
            elif(size/(256*256)>0.2):
                large_iou_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'iou'))
                large_dice_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'dice_coefficient'))
                large_accuracy_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'accuracy'))
                large_precision_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'precision'))
                large_recall_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'recall'))
                large_sensitivity_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'sensitivity'))
                large_specificity_list.append(calculate_metrics(mask_pred, gt.cpu().numpy().squeeze(), 'specificity'))
                large_hausdorff_distance_list.append(hd95(mask_pred, gt.cpu().numpy().squeeze()))


    print("IoU: %.8f" % np.mean(iou_list))
    print("Dice: %.8f" % np.mean(dice_list))
    print("Accuracy: %.8f" % np.mean(accuracy_list))
    print("Precision: %.8f" % np.mean(precision_list))
    print("Recall: %.8f" % np.mean(recall_list))
    print("Sensitivity: %.8f" % np.mean(sensitivity_list))
    print("Specificity: %.8f" % np.mean(specificity_list))
    print("Hausdorff Distance: %.8f" % np.mean(hausdorff_distance_list))

    print('良性数：',len(benign_iou_list))
    print('恶性数：',len(malignant_iou_list))

    print("IoU: %.8f" % np.mean(benign_iou_list))
    print("Dice: %.8f" % np.mean(benign_dice_list))
    print("Accuracy: %.8f" % np.mean(benign_accuracy_list))
    print("Precision: %.8f" % np.mean(benign_precision_list))
    print("Recall: %.8f" % np.mean(benign_recall_list))
    print("Sensitivity: %.8f" % np.mean(benign_sensitivity_list))
    print("Specificity: %.8f" % np.mean(benign_specificity_list))
    print("Hausdorff Distance: %.8f" % np.mean(benign_hausdorff_distance_list))

    print("IoU: %.8f" % np.mean(malignant_iou_list))
    print("Dice: %.8f" % np.mean(malignant_dice_list))
    print("Accuracy: %.8f" % np.mean(malignant_accuracy_list))
    print("Precision: %.8f" % np.mean(malignant_precision_list))
    print("Recall: %.8f" % np.mean(malignant_recall_list))
    print("Sensitivity: %.8f" % np.mean(malignant_sensitivity_list))
    print("Specificity: %.8f" % np.mean(malignant_specificity_list))
    print("Hausdorff Distance: %.8f" % np.mean(malignant_hausdorff_distance_list))

    print('small数：',len(small_iou_list),'medium数：',len(medium_iou_list),'large数：',len(large_iou_list))
    print("IoU: %.8f" % np.mean(small_iou_list))
    print("Dice: %.8f" % np.mean(small_dice_list))
    print("Accuracy: %.8f" % np.mean(small_accuracy_list))
    print("Precision: %.8f" % np.mean(small_precision_list))
    print("Recall: %.8f" % np.mean(small_recall_list))
    print("Sensitivity: %.8f" % np.mean(small_sensitivity_list))
    print("Specificity: %.8f" % np.mean(small_specificity_list))
    print("Hausdorff Distance: %.8f" % np.mean(small_hausdorff_distance_list))
    print('-----------------------------------')
    print("IoU: %.8f" % np.mean(medium_iou_list))
    print("Dice: %.8f" % np.mean(medium_dice_list))
    print("Accuracy: %.8f" % np.mean(medium_accuracy_list))
    print("Precision: %.8f" % np.mean(medium_precision_list))
    print("Recall: %.8f" % np.mean(medium_recall_list))
    print("Sensitivity: %.8f" % np.mean(medium_sensitivity_list))
    print("Specificity: %.8f" % np.mean(medium_specificity_list))
    print("Hausdorff Distance: %.8f" % np.mean(medium_hausdorff_distance_list))
    print('-----------------------------------')
    print("IoU: %.8f" % np.mean(large_iou_list))
    print("Dice: %.8f" % np.mean(large_dice_list))
    print("Accuracy: %.8f" % np.mean(large_accuracy_list))
    print("Precision: %.8f" % np.mean(large_precision_list))
    print("Recall: %.8f" % np.mean(large_recall_list))
    print("Sensitivity: %.8f" % np.mean(large_sensitivity_list))
    print("Specificity: %.8f" % np.mean(large_specificity_list))
    print("Hausdorff Distance: %.8f" % np.mean(large_hausdorff_distance_list))




