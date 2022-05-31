import cv2 
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import math
import torch
import dataset
from torch.utils.data import DataLoader
from ground_truth_extraction import GTPointExtractionDataset




#print(torch.cuda.is_available())


#print(torch.cuda.get_device_name(0))

def calculate_dataset_mean_and_std():
    """
    Calculates the mean and standard decation of the entire dataset
    """
    data = dataset.LaserPointDataset('/home/oyvind/Blender-weldgroove/render',noise=False, return_gt=True, corrected=True, normalization='')
    loader = DataLoader(data)

    mean_x = 0
    mean_y = 0
    std_x = 0
    std_y = 0
    i = 0
    for est, _, _ in loader:
        e = est.squeeze()
        if e[1].mean() > 100:
            print('asdfg')
            breakpoint()
        
        mean_x += e[0].mean()
        mean_y += e[1].mean()
        std_x += e[0].std()
        std_y += e[1].std()
        i+=1
        #print(i)
    
    print('meanx before division',mean_y)
    mean_x = mean_x / len(data)
    mean_y = mean_y / len(data)
    std_x = std_x / len(data)
    std_y = std_y / len(data)

    return mean_x, mean_y, std_x, std_y

def remove_outliers():
    """
    removes outliers from the estimated point cloud and replaces them with 
    the corresponding ground truth points, then takes the x-value of the point cloud into account to update the depth value.
    """

    root = '/home/oyvind/Blender-weldgroove/render'
    dataset = GTPointExtractionDataset(root, corrected=False)
    print('length dataset:', len(dataset))
    img = 6000 # 1720 first images ok
    h = img
    # extract only ground-truth from the datataset
    while img < len(dataset):
        
        print('i = ',img)
        g, est = dataset[img]

        render = str((img // 20) + 1)
        idx = str((img % 20) + 1)
        while len(idx) < 4:
            idx = '0' + idx

        # the 'real' depth value is equal to the length of the vector spanned by the x- and z-value in the laser frame
        g[2] = np.sqrt(g[0]**2 + g[2]**2)
        est[2] = np.sqrt(est[0]**2 + est[2]**2)

        g_outliers_y = np.where(g[1] - np.average(g[1]) > 5 * np.std(g[1]))[0]
        if len(g_outliers_y) > 0:
            print(f'outliers y (GT): {g_outliers_y}')
            g[1][g_outliers_y] = np.average(g[1])
            break
        g_outliers_z = np.where(g[2] - np.average(g[2]) > 5 * np.std(g[2]))[0]
        if len(g_outliers_z) > 0:
            print(f'outliers z (GT): {g_outliers_z}')
            g[2][g_outliers_z] = np.average(g[2])
            break
        
        shape_diff = g[1].shape[0] / est[1].shape[0]

        s = 3

        small_s = [125, 131, 182, 187, 190, 194, 226, 239, 244, 262, 285] # render - 1

        if img // 20 in small_s:
            print('     asdfasddafsfsafsdfadsfsdf         ')
            s = 2

        est_outliers_y = np.where(abs(est[1] - np.average(est[1])) > s * np.std(est[1]))[0]
        #print(np.average(est[1]))
        #print(np.std(est[1]))
        if len(est_outliers_y) > 0:
            print(f'outliers y (EST): {est_outliers_y}')
            print(f'avg = { np.average(est[1])}')
            print(f'{est[1][est_outliers_y]}')
            #time.sleep(2)
            g_indices_y = np.ceil(est_outliers_y * shape_diff).astype(int)
            est[1][est_outliers_y] = g[1][g_indices_y]
            #est[2][est_outliers_y] = g[2][g_indices_y]

        est_outliers_z = np.where(abs(est[2] - np.average(est[2])) > s * np.std(est[2]))[0]
        #print()
        print(np.average(est[2]))
        print(np.std(est[2]))
        if len(est_outliers_z) > 0:
            print(f'outliers z (EST): {est_outliers_z}')
            print(f'avg = { np.average(est[2])}')
            print(f'std = {np.std(est[2])}')
            print(f'{est[2][est_outliers_z]}')
            #time.sleep(2)
            g_indices_z = np.ceil(est_outliers_z * shape_diff).astype(int)
            est[2][est_outliers_z] = g[2][g_indices_z]
            #est[1][est_outliers_z] = g[1][g_indices_z]
            print('mnew', g[2][g_indices_z])

        

        path = os.path.join(root, render, 'processed_images', 'points_' + idx)
        np.save(path + '/' + idx + '_GT_fixed.npy', g)
        np.save(path + '/' + idx + '_EST_fixed.npy', est)

        img+=1

if __name__ == "__main__":
    mx, my, sx, sy =  calculate_dataset_mean_and_std()
    dataset_statistics = {"mean_x": mx,
                          "mean_y": my,
                          "std_x": sx,
                          "std_y": sy,
    }
    print(dataset_statistics)

    #remove_outliers()