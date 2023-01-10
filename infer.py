import os
from enum import Enum

from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from data.base_dataset import Normalize_image
from utils.saving_utils import load_checkpoint_mgpu

from networks import U2NET

device = "cuda"

image_dir = "50pageraw"
output_dir = "./50page_cropped"
mask_dir = output_dir + "masks"
img_dir = output_dir + "imgs"
# image_dir = "subset"
# checkpoint_path = os.path.join("trained_checkpoint", "cloth_segm_u2net_latest.pth")
checkpoint_path = "/home/ubuntu/seg/cloth-segmentation/trained_checkpoint/cloth_segm_u2net_latest.pth"
do_palette = True


def get_palette(num_cls):
    """Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
    return palette
class Categories(Enum):
    UPPER_BODY = 1
    LOWER_BODY = 2
    FULL_BODY = 3
def scale_bounding_box(top_left, bottom_right, full_shape, image_shape):
    height_scale = (full_shape[0] // image_shape[0])
    width_scale = (full_shape[1] // image_shape[1])

    new_top_left = (top_left[0] * height_scale, top_left[1] * width_scale)
    new_bottom_right = (bottom_right[0] * full_shape[0] // image_shape[0], bottom_right[1] * full_shape[1] // image_shape[1])
    return (new_top_left, new_bottom_right)
    
def get_cropped_img(image, mask, full_shape):
    # Find the bounding box of the masked region 
    image = transforms.functional.resize(image, list(reversed(full_shape)))
    image = image.numpy().transpose(1, 2, 0).astype(np.uint8)
    test = Image.fromarray(image)
    test.save("test.jpg")
    mask = mask.squeeze().numpy().astype(np.uint8)
    x, y, w, h = cv2.boundingRect(mask)

    # Calculate the center point of the bounding box
    center_x, center_y = x + w // 2, y + h // 2

    # Calculate the top-left and bottom-right coordinates of the rectangular crop
    top_left = (x, y)
    bottom_right = (x + w, y + h)
    # Check if the crop coordinates are within the bounds of the image
    # if top_left[0] < 0:
    #     top_left = (0, top_left[1])
    # if top_left[1] < 0:
    #     top_left = (top_left[0], 0)
    # if bottom_right[0] > image.shape[1]:
    #     bottom_right = (image.shape[1], bottom_right[1])
    # if bottom_right[1] > image.shape[0]:
    #     bottom_right = (bottom_right[0], image.shape[0])
    scaled_top_left, scaled_bottom_right = scale_bounding_box(top_left, bottom_right, full_shape, image.shape)
    # Crop the image using the calculated coordinates
    # scaled_top_left = top_left
    # scaled_bottom_right = bottom_right
    cropped_image = image[scaled_top_left[1]:scaled_bottom_right[1], \
                            scaled_top_left[0]:scaled_bottom_right[0]]

    cropped_pil = Image.fromarray(cropped_image)
    return cropped_pil


def get_masked_img(image, mask_tensor, full_shape):
    masked_img = image.cpu() * mask_tensor
    masked_img[mask_tensor.expand(3, -1, -1) == 0] = 255 
    masked_img = masked_img.squeeze().detach().movedim(0, 2).numpy()
    pil_img = Image.fromarray(masked_img.astype("uint8"))
    return pil_img.resize(full_shape)
    
def mask_img(input_img, 
             mask_tensor, 
             full_shape,
             get_masked=False, get_cropped=True, get_category=None):
    if get_category:
        raise NotImplementedError

    input_img = (input_img + 1) * 127.5

    # mask_tensor = transforms.functional.resize(mask_tensor.unsqueeze(0), full_shape)
    # mask_tensor = mask_tensor.unsqueeze(0)
    upper_body = mask_tensor[mask_tensor == 1]
    lower_body = mask_tensor[mask_tensor == 2]
    full_body = mask_tensor[mask_tensor == 3]
    if upper_body.numel() > lower_body.numel():
        mask_tensor[mask_tensor == 2] = 0
    else:
        mask_tensor[mask_tensor == 1] = 0
    mask_tensor = torch.clip(mask_tensor, 0, 1).cpu()
    if len(input_img.shape) == 4:
        input_img = input_img.squeeze(0)
    # mask_tensor = torch.ceil(mask_tensor)
    if get_masked:
        return get_masked_img(input_img, mask_tensor, full_shape)
    if get_cropped:
        return get_cropped_img(input_img, mask_tensor, full_shape)


transforms_list = []
transforms_list += [transforms.ToTensor()]
transforms_list += [Normalize_image(0.5, 0.5)]
transform_rgb = transforms.Compose(transforms_list)
resize = transforms.Resize((768, 768))

net = U2NET(in_ch=3, out_ch=4)
net = load_checkpoint_mgpu(net, checkpoint_path)
net = net.to(device)
net = net.eval()

palette = get_palette(4)

images_list = sorted(os.listdir(image_dir))
pbar = tqdm(total=len(images_list))

dirs = [output_dir, mask_dir, img_dir]
for dir in dirs:
    if not os.path.exists(dir):
        os.mkdir(dir)

for image_name in images_list:
    img = Image.open(os.path.join(image_dir, image_name)).convert("RGB")
    fullscale_image_tensor = transform_rgb(img)
    
    full_shape = list(reversed(fullscale_image_tensor.shape[1:]))
    image_tensor = resize(fullscale_image_tensor)
    image_tensor = torch.unsqueeze(image_tensor, 0)
    resized_image = image_tensor

    output_tensor = net(image_tensor.to(device))
    output_tensor = F.log_softmax(output_tensor[0], dim=1)
    output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_arr = output_tensor.cpu().numpy()

    output_img = Image.fromarray(output_arr.astype("uint8"), mode="L")
    if do_palette:
        output_img.putpalette(palette)
    output_img.save(os.path.join(mask_dir, image_name[:-3] + "png"))
    # masked_img = mask_img(resized_image, output_tensor, full_shape, get_masked=True, get_cropped=False)
    masked_img = mask_img(resized_image, output_tensor, full_shape, )
    masked_img.save(os.path.join(img_dir, image_name[:-3] + "_masked.png"))
    # print(masked.shape)

    pbar.update(1)

pbar.close()