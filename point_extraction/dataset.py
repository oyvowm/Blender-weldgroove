import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import cv2
import os
import numpy as np

class LaserPointDataset(Dataset):
    def __init__(self, root):
        self.root = root

    def __getitem__(self, index):
        render = str((index // 20) + 1)
        idx = str((index % 20) + 1)
        while len(idx) < 4:
            idx = '0' + idx
        
        path = os.path.join(self.root, render, 'processed_images' ,'points_' + idx)
        print(f'path: {path}')
        est = np.load(path + '/' + idx + '_EST.npy')
        gt = np.load(path + '/' + idx + '_GT.npy')

        return est, gt


    def __len__(self):
        renders = os.listdir(self.root)
        renders = [i for i in renders if (i[-3:] != 'npy' and i[-3:] != 'exr')]
        return len(renders) * 20


if __name__ == "__main__":
    dataset = LaserPointDataset('/home/oyvind/Blender-weldgroove/render')
    #print(dataset[17])
    print(len(dataset))
    e, g = dataset[2]

    print(e.shape)
    print(g[2])
