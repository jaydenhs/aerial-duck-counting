from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
import cv2
from torchvision import transforms
import random
from PIL import Image
import glob
import h5py
import math
import torchvision.transforms.functional as F


def random_crop(im_w, im_h, crop_w, crop_h):
    res_w = im_w - crop_w
    res_h = im_h - crop_h
    i = random.randint(0, res_w)
    j = random.randint(0, res_h)
    return i, j, crop_w, crop_h


class Crowd(Dataset):
    def __init__(self, root, crop_size, d_ratio, method='train'):
        self.imlist = sorted(glob.glob(os.path.join(root, 'imgs/*.png')))
        self.c_size = crop_size
        self.d_ratio = d_ratio
        assert self.c_size % self.d_ratio == 0, f"crop size {crop_size} should be divided by downsampling ratio {d_ratio}. "
        if method not in ['train', 'val']:
            raise Exception('Method not implemented!')
        self.method = method
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.452016860247, 0.447249650955, 0.431981861591],
                                 [0.23242045939, 0.224925786257, 0.221840232611])
        ])

    def __len__(self):
        return len(self.imlist)

    def __getitem__(self, item):
        im_path = self.imlist[item]
        im = Image.open(im_path).convert("RGB")

        if self.method == 'train':
            d_map_path = im_path.replace("imgs", "gt_den").replace("png", "h5").replace("Bas", "GT_Bas")
            d_map = h5py.File(d_map_path, 'r')['density_map']
            img, d_map, b_map = self.train_transform(im, d_map)
            return self.transform(img), d_map, b_map
        else:
            w, h = im.size
            new_w = math.ceil(w / 16) * 16
            new_h = math.ceil(h / 16) * 16
            img = im.resize((new_w, new_h), Image.BICUBIC)
            points = np.load(im_path.replace("imgs", 'points').replace('png', 'npy'))
            name = os.path.basename(im_path).split(".png")[0]
            return self.transform(img), len(points), name

    def train_transform(self, img, density_map):
        w, h = img.size
        i, j, c_w, c_h = random_crop(w, h, self.c_size, self.c_size)
        img = F.crop(img, j, i, c_h, c_w)
        density_map = density_map[j: (j + c_h), i: (i + c_w)]
        if random.random() > 0.5:
            img = F.hflip(img)
            density_map = density_map[:, ::-1]
        d_map = cv2.resize(density_map, (int(density_map.shape[1] / self.d_ratio),
                                         int(density_map.shape[0] / self.d_ratio)),
                           interpolation=cv2.INTER_AREA) * (self.d_ratio ** 2)
        b_map = (d_map > 1e-3).astype(np.float32)
        d_map = torch.from_numpy(d_map).float()
        b_map = torch.from_numpy(b_map).float()

        return img, d_map.unsqueeze(0), b_map.unsqueeze(0)
