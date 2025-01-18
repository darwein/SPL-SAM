import numpy as np
import os
from torch.nn import functional as F
import cv2
import torch
from utils.utils import poly_lr
import monai
from utils.utils import Logger
from mysam import sam_model_registry
from skimage.measure import label, regionprops
from torch.utils.data import DataLoader,Subset
from torch.utils.data import Dataset
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--save_distill_checkpoints', type=str, default="")
parser.add_argument('--teacher_model', type=str, default='teacher')
parser.add_argument('--student_model', type=str, default='student')

parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--SAMmed_checkpoint', type=str, default="./mysam/sam-med2d_b.pth")
parser.add_argument('--image_size', type=int, default=256, help='image size')
parser.add_argument('--distillation_dataset', type=str, default='')

parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.001)
args = parser.parse_args()

def get_boxes_from_mask(mask, box_num=1, std=0.1, max_pixel=30):#20
    """
    Args:
        mask: Mask, can be a torch.Tensor or a numpy array of binary mask.
        box_num: Number of bounding boxes, default is 1.
        std: Standard deviation of the noise, default is 0.1.
        max_pixel: Maximum noise pixel value, default is 5.
    Returns:
        noise_boxes: Bounding boxes after noise perturbation, returned as a torch.Tensor.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()

    label_img = label(mask)
    regions = regionprops(label_img)

    # Iterate through all regions and get the bounding box coordinates
    boxes = [tuple(region.bbox) for region in regions]

    # If the generated number of boxes is greater than the number of categories,
    # sort them by region area and select the top n regions
    if len(boxes) >= box_num:
        sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)[:box_num]
        boxes = [tuple(region.bbox) for region in sorted_regions]

    # If the generated number of boxes is less than the number of categories,
    # duplicate the existing boxes
    elif len(boxes) < box_num:
        num_duplicates = box_num - len(boxes)
        boxes += [boxes[i % len(boxes)] for i in range(num_duplicates)]

    # Perturb each bounding box with noise
    noise_boxes = []
    for box in boxes:
        y0, x0, y1, x1 = box
        width, height = abs(x1 - x0), abs(y1 - y0)
        # Calculate the standard deviation and maximum noise value
        noise_std = min(width, height) * std
        max_noise = min(max_pixel, int(noise_std * 5))
        # Add random noise to each coordinate
        noise_x = np.random.randint(0, max_noise)
        noise_y = np.random.randint(0, max_noise)
        x0, y0 = x0 - noise_x, y0 - noise_y
        x1, y1 = x1 + noise_x, y1 + noise_y
        noise_boxes.append((x0, y0, x1, y1))
    return torch.as_tensor(noise_boxes, dtype=torch.float)
class ProstateDataset(Dataset):
    def __init__(self, data_root,img_size=256,is_train=False):
        self.data_root_image = data_root+'/'+'images/'
        self.data_root_mask = data_root + '/' + 'masks/'
        self.img_size = img_size
        self.image_filenames = os.listdir(self.data_root_image)
        self.transform = is_train


    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_root_image, self.image_filenames[index])
        mask_path = os.path.join(self.data_root_mask,self.image_filenames[index].replace(').png', ')_mask.png'))
        filename=self.image_filenames[index]
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, (self.img_size, self.img_size))[:, :, 0].squeeze()
        mask = np.where(mask > 50, 1, 0)

        boxes = get_boxes_from_mask(mask)
        return filename, image, torch.tensor(mask[None, :, :]), boxes

distill_dataset = ProstateDataset(data_root=args.distillation_dataset,img_size=args.image_size)

indices = list(range(len(distill_dataset)))
split = int(len(distill_dataset) * 0.8)  # 假设20%的数据用于验证
np.random.seed(args.seed)
np.random.shuffle(indices)
train_indices, test_indices = indices[:split], indices[split:]
train_dataset = Subset(distill_dataset, train_indices)
val_dataset = Subset(distill_dataset, test_indices)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# 开始蒸馏-------------------------------------------------
def distill():
    teacher_model = sam_model_registry[args.teacher_model](checkpoint=args.SAMmed_checkpoint).to(args.device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    student_model = sam_model_registry[args.student_model]().to(args.device)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    DTloss = torch.nn.MSELoss()
    dicece_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    train_loss_log = {'epoch': [], 'loss': []}
    val_loss_log = {'epoch': [], 'loss': []}

    logger = Logger(output_folder=args.save_distill_checkpoints)
    print('start distillation')
    for epoch in range(args.num_epochs):
        optimizer.param_groups[0]['lr'] = poly_lr(epoch, args.num_epochs, args.learning_rate, 0.9)

        student_model.train()
        student_train_loss = 0
        for name,img, gt, box in train_dataloader:
            B = img.shape[0]
            img = img.to(torch.float32).permute(0, 3, 1, 2).to(args.device)
            box = box.to(args.device)
            # 学生模型预测
            student_image_imbeddings = student_model.image_encode(img, mask_ratio=0)  # (B,C=256,16,16)
            # 教师模型预测
            with torch.no_grad():
                teacher_image_imbeddings = teacher_model.image_encode(img, mask_ratio=0)  # (8,C=256,16,16)
                sparse_embeddings, dense_embeddings = teacher_model.prompt_encoder(
                    points=None,
                    boxes=box,
                    masks=None, )
            low_res_masks, iou_predictions = teacher_model.mask_decoder(
                image_embeddings=student_image_imbeddings[-1],
                image_pe=teacher_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False,
            )
            mask_pred = F.interpolate(low_res_masks, (args.image_size, args.image_size), mode="bilinear",
                                      align_corners=False, )

            optimizer.zero_grad()
            # 计算蒸馏后的预测结果及soft_loss
            soft_loss = DTloss(student_image_imbeddings[-1], teacher_image_imbeddings)
            hard_loss=dicece_loss(mask_pred, gt.to(args.device))
            loss=soft_loss+0.5*hard_loss
            # 反向传播,优化权重
            loss.backward()
            student_train_loss += loss.item()
            optimizer.step()

        student_model.eval()
        student_val_loss = 0
        for name,img, gt, box in val_dataloader:
            B = img.shape[0]
            img = img.to(torch.float32).permute(0, 3, 1, 2).to(args.device)
            box = box.to(args.device)
            with torch.no_grad():
                student_image_imbeddings = student_model.image_encode(img, mask_ratio=0)  # (B,C=256,16,16)
                teacher_image_imbeddings = teacher_model.image_encode(img, mask_ratio=0)  # (8,C=256,16,16)
                sparse_embeddings, dense_embeddings = teacher_model.prompt_encoder(
                    points=None,
                    boxes=box,
                    masks=None, )
                low_res_masks, iou_predictions = teacher_model.mask_decoder(
                    image_embeddings=student_image_imbeddings[-1],  # (B, 256, 64, 64)
                    image_pe=teacher_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                    sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                    dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                    multimask_output=False,
                )
                mask_pred = F.interpolate(low_res_masks, (args.image_size, args.image_size), mode="bilinear",
                                          align_corners=False, )
                soft_loss = DTloss(student_image_imbeddings[-1], teacher_image_imbeddings)
                hard_loss = dicece_loss(mask_pred, gt.to(args.device))
                student_val_loss += (soft_loss+0.5*hard_loss).item()

        checkpoint = student_model.state_dict()
        torch.save(checkpoint, args.save_distill_checkpoints + f'epoch{epoch}.pth')
        logger.print_to_log_file(f'epoch:{epoch}    student_train_loss:{student_train_loss / len(train_dataloader):.8f}')
        logger.print_to_log_file(f'epoch:{epoch}    student_val_loss:{student_val_loss / len(val_dataloader):.8f}')

        train_loss_log['epoch'].append(epoch)
        train_loss_log['loss'].append(student_train_loss / len(train_dataloader))
        val_loss_log['epoch'].append(epoch)
        val_loss_log['loss'].append(student_val_loss / len(val_dataloader))
    torch.save(train_loss_log,args.save_distill_checkpoints+ 'train_loss_log.pth')
    torch.save(val_loss_log,args.save_distill_checkpoints+ 'val_loss_log.pth')

if __name__ == '__main__':
    distill()



