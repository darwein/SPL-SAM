import os
from mysam import sam_model_registry
import argparse
import numpy as np
from torch.nn import functional as F
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.cuda.amp
from scipy.ndimage import binary_fill_holes

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--model_type', type=str, default='student')

parser.add_argument('--image_size', type=int, default=256, help='image size')
parser.add_argument('--dataset_path', type=str, default='./for_segment/dataset/')
parser.add_argument('--save_pre_path', type=str, default='./for_segment/result/')
parser.add_argument('--model_save_path', type=str, default='./checkpoints_mysam/')
args = parser.parse_args()
device=args.device

def extract_largest_connected_component_opencv(binary_image):
    """
    使用 OpenCV 提取二值图像中的最大连通区域
    :param binary_image: 输入二值图像 (numpy array)
    :return: 包含最大连通区域的二值图像
    """
    # 确保输入是二值图像
    binary_image = binary_image.astype(np.uint8)
    # 连通区域分析
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    # 排除背景，找到最大连通区域
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    # 提取最大连通区域
    largest_connected_component = (labels == largest_label).astype(np.uint8)
    return largest_connected_component

def fill_holes(binary_image):
    """
    填充二值图像中的孔洞
    :param binary_image: 输入二值图像 (numpy array)
    :return: 填充孔洞后的图像
    """
    filled_image = binary_fill_holes(binary_image).astype(np.uint8)
    return filled_image

def smooth_boundaries(binary_image):
    """
    平滑二值图像的边界
    :param binary_image: 输入二值图像 (numpy array)
    :return: 平滑后的图像
    """
    # 应用高斯模糊
    blurred_image = cv2.GaussianBlur(binary_image.astype(np.float32), (5, 5), 0)
    # 二值化
    smoothed_image = (blurred_image > 0).astype(np.uint8)
    return smoothed_image

def smooth_segmentation(binary_image, kernel_size=5):
    """
    使用膨胀和腐蚀操作平滑分割结果
    :param binary_image: 输入二值分割图像 (numpy array)
    :param kernel_size: 核大小，用于控制平滑程度
    :return: 平滑后的二值分割图像
    """
    # 创建形态学核
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # 开运算（去除小噪声）
    opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    # 闭运算（填补孔洞）
    smoothed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return smoothed

def postprocess(predict):
    #predict = smooth_segmentation(predict, kernel_size=5)
    predict = fill_holes(predict)
    #predict = smooth_boundaries(predict)
    predict = extract_largest_connected_component_opencv(predict)
    predict = smooth_boundaries(predict)
    return predict

def save_test_result(save_path,name,image,predict):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    cur_name = name[0]
    # image=image.permute(0, 2, 3, 1).cpu().numpy().squeeze()
    # cv2.imwrite(os.path.join(save_path,cur_name), image)

    predict = postprocess(predict)
    predict = (predict * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_path,cur_name.replace('.png','_mask.png')), predict)

class ProstateDataset(Dataset):
    def __init__(self, data_root,img_size=256,is_train=False):
        self.data_root_image = data_root+'/'
        self.img_size = img_size
        self.image_filenames = os.listdir(self.data_root_image)
        self.transform = is_train

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_root_image, self.image_filenames[index])
        filename=self.image_filenames[index]
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.img_size, self.img_size))
        return filename, image

mysam_model = sam_model_registry[args.model_type](checkpoint=args.model_save_path+'mysam_model_best.pth')
mysam_model.to(args.device)

dataset_path=args.dataset_path
infer_dataset = ProstateDataset(data_root=dataset_path,img_size=args.image_size,is_train=False)
infer_dataloader = DataLoader(infer_dataset, batch_size=1, shuffle=False)

mysam_model.eval()

for step, (name, image) in enumerate(infer_dataloader):
    image = image.to(torch.float32).permute(0, 3, 1, 2).to(device)
    with torch.no_grad():
        image_embedding = mysam_model.image_encode(image, mask_ratio=0)
        B, _, H, W = image.shape
        box_torch = torch.from_numpy(np.array([[0, 0, H, W]])).float().to(device)
        sparse_embeddings, dense_embeddings = mysam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None, )
        # predicted masks
        low_res_masks, _ = mysam_model.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=mysam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False, )
        mask_pred = F.interpolate(low_res_masks, (256, 256), mode="bilinear",
                                  align_corners=False, )
        mask_pred = torch.sigmoid(mask_pred)
        mask_pred = mask_pred.cpu().numpy().squeeze()
        mask_pred = (mask_pred > 0.5).astype(np.uint8)

        save_test_result(args.save_pre_path, name, image,mask_pred)




