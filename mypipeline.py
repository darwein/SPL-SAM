import os
from torch.utils.data import Subset
from mysam import sam_model_registry
from training_testing import run_training_wholebox, run_testing_wholebox
import argparse
import monai
import numpy as np
import cv2
from torch.utils.data import DataLoader
from utils.datasets import ProstateDataset
import torch.cuda.amp

def save_test_result(save_path,name,predict):
    cur_name=name[0]
    predict = (predict * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_path,cur_name.replace('.png','_mask.png')), predict)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--model_type', type=str, default='student')
parser.add_argument('--distill_checkpoints', type=str, default='Place your weights here')
parser.add_argument('--SAMmed_checkpoint', type=str, default="./mysam/sam-med2d_b.pth")

parser.add_argument('--image_size', type=int, default=256, help='image size')
parser.add_argument('--dataset_path', type=str, default='')
parser.add_argument('--model_save_path', type=str, default='')
parser.add_argument('--test_result_save_path', type=str, default='')

parser.add_argument('--epoch', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.0001)

parser.add_argument('--mse', type=bool, default=False)
parser.add_argument('--sgd', type=bool, default=False)
parser.add_argument('--train_test', type=str, default='train')
parser.add_argument('--random_validation', type=bool, default=False)
parser.add_argument('--mixprecision', type=bool, default=False)
args = parser.parse_args()

SAMmed_weight=torch.load(args.SAMmed_checkpoint)['model']
promptencoder_weights = {k: v for k, v in SAMmed_weight.items() if 'prompt_encoder' in k}
model = sam_model_registry[args.model_type](checkpoint=args.distill_checkpoints)
model.to(args.device)
model.load_state_dict(promptencoder_weights, strict=False)
medsam = sam_model_registry['teacher'](checkpoint=args.SAMmed_checkpoint).to(args.device)


dataset_path=args.dataset_path
all_dataset = ProstateDataset(data_root=dataset_path,img_size=args.image_size,is_train=False)
# 分为测试集和训练集-------------------
indices = list(range(len(all_dataset)))
split = int(len(all_dataset) * 0.2)  # 假设80%的数据用于验证
np.random.seed(args.seed)
np.random.shuffle(indices)
train_indices, val_indices = indices[:split], indices[split:]
train_dataset = Subset(ProstateDataset(data_root=dataset_path, img_size=args.image_size, is_train=True), train_indices)
val_dataset = Subset(ProstateDataset(data_root=dataset_path, img_size=args.image_size, is_train=False), val_indices)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)


if args.sgd:
    optimizer = torch.optim.SGD(model.mask_decoder.parameters(), lr=args.lr, weight_decay=0, momentum=0.99)
else:
    optimizer = torch.optim.Adam(model.mask_decoder.parameters(), lr=args.lr, weight_decay=0.001)

if args.mse:
    iou_loss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
else:
    iou_loss = torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')

criterion1 = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean',lambda_dice= 1.0,lambda_ce = 1.5)
criterion2 = monai.losses.DiceFocalLoss(sigmoid=True, squared_pred=True, reduction='mean',lambda_dice= 0.0,lambda_focal=1.0)
scaler = torch.cuda.amp.GradScaler(enabled=args.mixprecision)

# train, validation and test
if args.train_test_compare == 'test':
    print('start testing')
    if not os.path.exists(os.path.join(args.model_save_path, 'mysam_model_best.pth')):
        raise FileNotFoundError("The model file does not exist. Train the model before testing.")
    model = sam_model_registry[args.model_type](checkpoint=args.model_save_path+'mysam_model_best.pth')
    model.to(args.device)

    run_testing_wholebox(img_size=args.image_size,model=model,medsam=medsam,val_dataloader=val_dataloader, device=args.device,
                         criterion1=criterion1,criterion2=criterion2,save_path=args.test_result_save_path)

elif args.train_test_compare == 'train':
    print('start training')
    run_training_wholebox(
        img_size=args.image_size,mysam_model=model,medsam=medsam, max_num_epochs=args.epoch, model_save_path=args.model_save_path,
        optimizer=optimizer, initial_lr=args.lr, train_dataloader=train_dataloader,
        val_dataloader=val_dataloader, scaler=scaler, criterion1=criterion1,criterion2=criterion2,
        iou_loss=iou_loss, device=args.device)








