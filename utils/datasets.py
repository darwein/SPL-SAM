import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from skimage.measure import label, regionprops
import random
import albumentations as A


def get_transforms1():
    return A.Compose([
        A.HorizontalFlip(p=0.5),                  # 随机水平翻转
        A.VerticalFlip(p=0.5),                    # 随机垂直翻转
        # A.RandomBrightnessContrast(p=0.2),       # 随机亮度和对比度调整
        # A.GaussianBlur(blur_limit=(3,7), p=0.2),     # 模糊
        # A.GaussianNoise(var_limit=(10.0, 30.0),p=0.2),                  # 高斯噪声
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),  # 随机平移、缩放、旋转
    ])

def get_transforms2():
    return A.Compose([
        A.RandomBrightnessContrast(p=0.2),       # 随机亮度和对比度调整
        A.GaussianBlur(blur_limit=(3,7), p=0.2),     # 模糊
        A.GaussNoise(var_limit=(10.0, 30.0),p=0.2),                  # 高斯噪声
    ])

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

        if self.transform:
            transform1 = get_transforms1()
            augmented1 = transform1(image=image, mask=mask)
            image,mask = augmented1['image'], augmented1['mask']

            transform2 = get_transforms2()
            augmented2 = transform2(image=image, mask=mask)
            image = augmented2['image']

        #boxes = find_largest_region_rectangle(mask)
        #boxes = get_boxes_from_mask(mask)
        boxes=image
        #points,points_labels=get_points_from_mask(mask)


        return filename, image, torch.tensor(mask[None, :, :]), boxes#,points,points_labels



        # is_background = np.random.randint(0, 2)
        # if is_background:
        #     y_indices, x_indices = np.where(mask == 0)
        #     random_idx = np.random.randint(0, len(y_indices))
        #     prompt_points = np.array((x_indices[random_idx], y_indices[random_idx]))
        #     iou_label = torch.tensor([0]).float()
        #
        # else:
        #     y_indices, x_indices = np.where(mask > 0)
        #     random_idx = np.random.randint(0, len(y_indices))
        #     prompt_points = np.array((x_indices[random_idx], y_indices[random_idx]))
        #     iou_label = torch.tensor([1]).float()
        #
        # in_points = torch.as_tensor(prompt_points)
        # in_labels = 1
        #return filename,image, torch.tensor(mask[None, :,:]), in_points, in_labels, iou_label

def find_largest_region_rectangle(matrix):
    rows, cols = len(matrix), len(matrix[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    def dfs(x, y):
        stack = [(x, y)]
        min_x, min_y = x, y
        max_x, max_y = x, y

        while stack:
            cx, cy = stack.pop()
            if 0 <= cx < rows and 0 <= cy < cols and not visited[cx][cy] and matrix[cx][cy] == 1:
                visited[cx][cy] = True
                # Update boundary coordinates
                min_x, min_y = min(min_x, cx), min(min_y, cy)
                max_x, max_y = max(max_x, cx), max(max_y, cy)
                # Explore 8-connected neighbors
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                    stack.append((cx + dx, cy + dy))

        return (min_x, min_y), (max_x, max_y)

    max_area = 0
    top_left = (0, 0)
    bottom_right = (0, 0)
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 1 and not visited[i][j]:
                tl, br = dfs(i, j)
                area = (br[0] - tl[0] + 1) * (br[1] - tl[1] + 1)  # Calculate the rectangular area
                if area > max_area:
                    max_area = area
                    top_left = tl
                    bottom_right = br
    noise_boxes=[]
    y0, x0, y1, x1 = top_left[0],top_left[1], bottom_right[0], bottom_right[1]
    width, height = abs(x1 - x0), abs(y1 - y0)
    # Calculate the standard deviation and maximum noise value
    noise_std = min(width, height) * 0.1
    max_noise = min(5, int(noise_std * 5))
    # Add random noise to each coordinate
    noise_x = np.random.randint(-max_noise, max_noise)
    noise_y = np.random.randint(-max_noise, max_noise)
    x0, y0 = x0 + noise_x, y0 + noise_y
    x1, y1 = x1 + noise_x, y1 + noise_y
    noise_boxes.append((x0, y0, x1, y1))

    return torch.as_tensor(noise_boxes, dtype=torch.float)


def get_boxes_from_mask(mask, box_num=1, std=0.1, max_pixel=20):
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

def get_points_from_mask(mask, point_num=1):
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    ones_indices = np.argwhere(mask == 1)
    # 计算几何中心
    center = np.mean(ones_indices, axis=0)
    # 计算每个点到中心的距离
    distances = np.linalg.norm(ones_indices - center, axis=1)
    # 计算权重（距离越小权重越高）
    weights = 1 / (distances + 1e-5)  # 添加一个小值避免除零
    weights /= weights.sum()  # 归一化权重
    # 通过加权随机选择一个点
    chosen_index = random.choices(ones_indices, weights=weights, k=1)[0]
    chosen_point = tuple(chosen_index)

    return torch.as_tensor(chosen_point, dtype=torch.float),torch.as_tensor([1], dtype=torch.int)



