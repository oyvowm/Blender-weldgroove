from shutil import move
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from point_extraction import rot_matrix

def normalize(arr):
    """
    Converts np array into tensor and performs normalization.
    """
    #print(arr.shape)
    arr = torch.from_numpy(arr)
    arr = arr.type(torch.float32)
    #t = transforms.Normalize((0.0016, 0.2542), (0.0318, 0.0260)) # old
    t = transforms.Normalize((0.0012, 0.2145), (0.0263, 0.0183))
    arr = arr.unsqueeze(0)
    arr = arr.permute(1, 0, 2)
    arr = t(arr)
    arr = arr.squeeze()
    return arr

def cloud_normalization(arr):
    """
    performs normalization only taking the relevant point cloud into account
    (as oppposed to using the statistics of the whole dataset).
    """

    new_x = (arr[0] - np.average(arr[0])) / (np.amax(arr[0]) - np.amin(arr[0]))
    arr[0] = new_x
    
    new_y = (arr[1] - np.average(arr[1])) / (np.amax(arr[1]) - np.amin(arr[1]))
    arr[1] = new_y
    assert np.amax(new_y) < 1, 'fail'


    assert abs(np.average(new_x)) < 0.0000001, f'avg fail {np.average(new_x)}'
    assert abs(np.average(new_y)) < 0.0000001, f'avg fail {np.average(new_y)}'

    arr = torch.from_numpy(arr)
    arr = arr.type(torch.float32)

    return arr

def move_points(arr, labels):
    # moves the entire point cloud by random amounts along the x- and y-axis
    to_move_x = np.random.uniform(-0.004, 0.004)
    to_move_y = np.random.uniform(-0.02, 0.02)

    arr[-2] = arr[-2] + to_move_x
    arr[-1] = arr[-1] + to_move_y

    labels[0] = labels[0] + to_move_x
    labels[1] = labels[1] + to_move_y

    return arr, labels

def rotate_points(arr, labels):
    angle = np.random.uniform(-7, 7)
    rot_x = rot_matrix('x', angle)
    labels = np.vstack((np.ones((1, labels.shape[1])) ,labels))
    labels = rot_x @ labels
    return rot_x @ arr, labels[1:]

 
def change_x_range(arr, labels):
    x_scale = np.random.uniform(0.8, 1.2)
    #y_scale = np.random.uniform(0.9, 1.1)
    arr[0] = arr[0] * x_scale
    #arr[1] = arr[1] * y_scale
    labels[0] = labels[0] * x_scale
    #labels[1] = labels[1] * y_scale

    return arr, labels

def add_noise(arr, noise_segment_length, max_y_index):
    start = np.random.randint(0, 640)
    #print(arr[-1][max_y_index])
    if start < max_y_index - 50 or start > max_y_index + 50:
        noise_segment_length *= np.random.randint(2,5) # (3,8) før
    
    end = start + noise_segment_length
    while end > 639:
        end -= 1

    segment = arr[1, start:end]
    noise = np.random.uniform(-0.07, 0.07)
    gaussian_noise = np.random.normal(0, 0.0005, segment.shape)

    
    
    if len(segment) > 1:
        avg = np.average(segment)
    else:
        avg = 0

    # 70% chance of giving the noise a sinusoidal sape
    if np.random.rand() < 0.3:
        #print('yaafd')
        segment = 0.5 * avg + segment * 0.5 + noise + gaussian_noise
    else:
        #print('sine')
        wave = np.linspace(start, end, end - start)
        wave_frequency = np.random.uniform(0.1, 0.6)
        segment = 0.5 * avg + segment * 0.5 + noise  + 0.001 * np.sin(wave_frequency * wave) + gaussian_noise

    arr[1, start:end] = segment

    return arr 







class LaserPointDataset(Dataset):
    def __init__(self,
                 root,
                 noise=True,
                 return_gt=False, 
                 corrected=False,
                 normalization='dataset'
                 ):

        self.root = root
        self.noise = noise
        self.return_gt = return_gt
        self.corrected = corrected
        self.normalization_type = normalization
    def __getitem__(self, index):
        render = str((index // 20) + 1)
        idx = str((index % 20) + 1)
        while len(idx) < 4:
            idx = '0' + idx
        
        path = os.path.join(self.root, render, 'processed_images' ,'points_' + idx)
        #print(f'path: {path}')
        if self.corrected:
            est = np.load(path + '/' + idx + '_EST_fixed.npy')
            gt = np.load(path + '/' + idx + '_GT_fixed.npy')
            corner_points = np.load(path + '/' + idx + '_labels_corrected.npy')
        else:
            est = np.load(path + '/' + idx + '_EST.npy')
            gt = np.load(path + '/' + idx + '_GT.npy')
            corner_points = np.load(path + '/' + idx + '_labels.npy')
        corner_points = corner_points.T


        
        if self.noise:

            idx = np.round(np.linspace(0, len(gt[0]) - 1, 640)).astype(int)
            ## adds noise at the indices where the x-value of the estimate and gt differ
            h = np.where((abs(gt[0][idx] - est[0]) < 0.19) & (abs(gt[0][idx] - est[0]) > 0.01))
            ##print((h))
            ## adds the difference mulitplied with a random scalar
            est[2][h] = est[2][h] + np.random.uniform(0.5, 2) * (gt[0][h] - est[0][h])
            
            # TRANSLATION       
            est, corner_points = move_points(est, corner_points)
            # ROTATION
            est, corner_points = rotate_points(est, corner_points)
            
            est = est[1:] 
            
            ## manually added segment noise
            max_y_index = np.argmax(est[-1])
            num_noise_segments = np.random.randint(0,5)
            for _ in range(num_noise_segments):
                noise_segment_length = np.random.randint(1,20)
                est = add_noise(est, noise_segment_length, max_y_index)
            
            est, corner_points = change_x_range(est, corner_points)
        else:
            est = est[1:]   
        if self.normalization_type == 'dataset':
            est = normalize(est)
        elif self.normalization_type == 'cloud':
            est = cloud_normalization(est)
        else:
            est = torch.from_numpy(est)
            est = est.type(torch.float32)
        corner_points = torch.from_numpy(corner_points)
        corner_points = corner_points.type(torch.float32)
        if self.return_gt:
            return est, gt, corner_points
        else:
            return est, corner_points

    def __len__(self):
        renders = os.listdir(self.root)
        renders = [i for i in renders if (i[-3:] != 'npy' and i[-3:] != 'exr')]
        
        renders = [int(i) for i in renders]
        renders.sort()
        #print(renders)
        #renders = renders[:70]
        #renders.pop()
        renders = [str(i) for i in renders]  
        
        return len(renders) * 20


if __name__ == "__main__":
    dataset = LaserPointDataset('/home/oyvind/Blender-weldgroove/render', noise=True, return_gt=True, corrected=True, normalization='')
    dataset2 = LaserPointDataset('/home/oyvind/Blender-weldgroove/render', noise=False, return_gt=True, corrected=False, normalization='')
    #print(dataset[17])
    #print(len(dataset))
    #e, g, p = dataset[45]
    for i in range(len(dataset)):
        e, g, p = dataset[i]
        if (len(p[1])) != 5:
            print(p.shape)
            print(i)
       # print(p.shape)

    #e2, _, _ = dataset2[590]
  #  e = e[1:]
    #n = normalize(e)



    #print(g.shape)
    #print(g[2][:10])
    #print(g[0][:10])
    #print()
    #print(g[1][:10])

    #e[2] = np.sqrt(e[0]**2 + e[2]**2)
    #condlist = [abs(g[0] - e[0]) > 0.005 and abs(g[0] - e[0]) < 0.05, ]
   # h = np.where((abs(g[0] - e[0]) < 0.003) & (abs(g[0] - e[0]) > 0.0001))
    #h = np.where(abs(g[0] - e[0]) > 0.005)
    #print('h', len(h[0]))
    #e[2][h] = e[2][h] + np.random.uniform(0.5, 5) * (g[0][h] - e[0][h])

    #print(e[2][:10])

    #e = e[1:] * 1000
    #g = g[1:] * 1000
    #print(e[-1][:10])
    
    
    #e = add_noise(e, 10)
    #e = change_x_y_range(e)
    #print(len(dataset))
    e = e * 1000
    #e2 = e2*1000
    p = p * 1000
    #p = p.T
    plt.scatter(e[0], e[1], s=1, color='g')
    g = g[1:] * 1000
    plt.scatter(g[0], g[1], s=1, color='r')
    #plt.scatter(e2[0], e2[1], s=1, color='r')
    plt.scatter(p[0], p[1], s = 20)
    plt.show()
    
    #print(e[1][:10])
    #print(g[1][:10])

   