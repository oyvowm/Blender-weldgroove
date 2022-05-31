from logging import root
from shutil import move
from time import time
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
    try:
        arr = torch.from_numpy(arr)
    except:
        pass
    arr = arr.type(torch.float32)
    #t = transforms.Normalize((0.0016, 0.2542), (0.0318, 0.0260)) # v old
    #t = transforms.Normalize((0.0012, 0.2145), (0.0263, 0.0183)) # old
    #t = transforms.Normalize((0.0013, 0.2211), (0.0281, 0.0194)) #
    #t = transforms.Normalize((0.0015, 0.2292), (0.0305, 0.0192)) #
    #t = transforms.Normalize((0.0017, 0.2310), (0.0308, 0.0195)) #
    #t = transforms.Normalize((0.0017, 0.2328), (0.0310, 0.0198))
    #t = transforms.Normalize((0.0061, 0.2474), (0.0307, 0.0298)) # with noise
    t = transforms.Normalize((0.0018, 0.2333), (0.0310, 0.0205))
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
    to_move_x = np.random.uniform(-0.018, 0.018)
    to_move_y = np.random.uniform(-0.048, 0.048)

    arr[-2] = arr[-2] + to_move_x
    arr[-1] = arr[-1] + to_move_y

    labels[0] = labels[0] + to_move_x
    labels[1] = labels[1] + to_move_y

    return arr, labels

def interpolate_groove(arr):
    """
    After tranformations such as e.g. rotating the groove, the resolution of points w.r.t 
    their angle to the laser scanner might not be realistic, hence the need to update the grooove
    through interpolation
    """

    x = np.linspace(arr[0][0], arr[0][-1], len(arr[0]))
    y = np.interp(x, arr[0], arr[1])

    arr[0] = x
    arr[1] = y

    return arr

def affine_transformation(arr, labels):
    angle = np.random.uniform(0, 10) # -30, 30
    #angle = 0
    rot_x = rot_matrix('x', angle)
    rot_x = rot_x[1:,1:]
    mean = np.mean(arr, axis=1)
    #streching = np.array([[np.random.uniform(0.85, 1.15), 0], [0, np.random.uniform(0.9, 1.3)]])
    stretch_x = np.random.normal(1, 0.05) # 0.1 før, 0.5
    stretch_y = np.random.normal(1, 0.15) # 0.3 før, 0.15
    #streching = np.array([[np.random.uniform(0.7, 1.3), 0], [0, np.random.uniform(0.6, 1.7)]])
    streching = np.array([[stretch_x, 0], [0, stretch_y]])

    arr[0] = arr[0] - mean[0]
    arr[1] = arr[1] - mean[1]
    arr = rot_x @ streching @ arr
    #arr = streching @ arr
    arr[0] = arr[0] + mean[0]
    arr[1] = arr[1] + mean[1]

    labels[0] = labels[0] - mean[0]
    labels[1] = labels[1] - mean[1]
    labels = rot_x @ streching @ labels
    #labels = streching @ labels
    labels[0] = labels[0] + mean[0]
    labels[1] = labels[1] + mean[1]

    #arr = interpolate_groove(arr)

    #labels = np.vstack((np.ones((1, labels.shape[1])) ,labels))
    #labels = 
    return arr, labels

def move_corner(arr, corner_points, idx):
    dist = (np.random.normal(0, 0.0005))
    arr[1, idx[2]:] = arr[1, idx[2]:] + dist
    corner_points[1, 2:] = corner_points[1, 2:] + dist
    x = np.linspace(corner_points[0, 1], corner_points[0, 2], len(arr[0, idx[1]+1:idx[2]]))
    y = np.linspace(corner_points[1, 1], corner_points[1, 2], len(arr[1, idx[1]+1:idx[2]]))
    #x = np.linspace(arr[0, idx[1]], arr[0, idx[2]], len(arr[0, idx[1]:idx[2]+1]))
    #y = np.linspace(arr[1, idx[1]], arr[1, idx[2]], len(arr[1, idx[1]:idx[2]+1]))
    assert len(arr[0, idx[1]+1:idx[2]]) == len(x), 'wrong length'
    arr[0, idx[1]+1:idx[2]] = x
    arr[1, idx[1]+1:idx[2]] = y

    #arr[0, idx[1]:idx[2]+1] = x
    #rr[1, idx[1]:idx[2]+1] = y

    return arr, corner_points


def reflect_groove(arr, idx):
    idx[0] = idx[0] + 3
    idx[1] = idx[1] - 3
    root_corners = arr[:, idx]
    points_to_reflect = arr[:, idx[0] + 1:idx[1]].T

    ab = root_corners[:,1] - root_corners[:,0]
    ap = points_to_reflect - root_corners[:,0]

    ab_stacked  = np.dstack(([ab] * ap.shape[0]))[0].T

    ap_dot_ab = np.dot(ap, ab.T)
    ap_dot_ab = np.vstack([ap_dot_ab] * 2).T

    points_to_line = ab_stacked * ap_dot_ab / np.dot(ab, ab) - ap

    reflected_points = points_to_reflect + points_to_line * 2

    arr[:, idx[0] + 1:idx[1]] = reflected_points.T


    return arr

def line_noise(arr):
    gaussian_noise = np.random.normal(0, 0.00003, arr[1].shape) # 5 før
    arr = arr + gaussian_noise

    return arr

def add_noise(arr, noise_segment_length, idx):
    start = np.random.randint(0, 640)
    #print(arr[-1][max_y_index])
    if (start < int(idx[1]) - 140 or start > int(idx[2]) + 90) and (start < int(idx[-1]) - 80):
        noise_segment_length *= np.random.randint(6,8) # (3,8) før
    while noise_segment_length > 150:
        noise_segment_length =- 15
    end = start + noise_segment_length
    while end > 639:
        end -= 1

    segment = arr[1, start:end]
    if len(segment) > 20:
        mini_segment_length = int(np.random.uniform(len(segment) / 15, len(segment) / 3))
        first_idx = 0

        for i in range((len(segment)) // mini_segment_length):
            if np.random.rand() < 0.5: # adding gaussian noise to mini segment
                segment[first_idx:mini_segment_length*(i+1)] += np.random.normal(0, 0.01) + np.random.normal(0, (np.random.rand() * 0.005), segment[first_idx:mini_segment_length*(i+1)].shape)
            else:
                segment[first_idx:mini_segment_length*(i+1)] += np.random.normal(0, 0.01)

            first_idx = mini_segment_length * i+1


    #noise = np.random.uniform(-0.05, 0.05)
    noise = np.random.normal(0, 0.05)
    gaussian_noise = np.random.normal(0, 0.0001, segment.shape)

    
    
    if len(segment) > 1:
        avg = np.average(segment)
    else:
        avg = 0

    # 50% chance of giving the noise a sinusoidal sape
    if np.random.rand() < 0.5:
        #print('yaafd')
        segment = 0.5 * avg + segment * 0.5 + noise + gaussian_noise
    else:
        #print('sine')
        wave = np.linspace(start, end, end - start)
        wave_frequency = np.random.uniform(0.1, 0.6)
        segment = 0.5 * avg + segment * 0.5 + noise  + 0.0006 * np.sin(wave_frequency * wave) + gaussian_noise

    arr[1, start:end] = segment

    return arr 






class LaserPointDataset(Dataset):
    def __init__(self,
                 root,
                 noise=True,
                 return_gt=False, 
                 corrected=False,
                 normalization='dataset',
                 test = False,
                 shuffle_gt_and_est = False,
                 autoencoder = False,
                 ):

        self.root = root
        self.noise = noise
        self.return_gt = return_gt
        self.corrected = corrected
        self.normalization_type = normalization
        self.shuffle_gt_and_est = shuffle_gt_and_est
        self.autoencoder = autoencoder
        
        renders = os.listdir(self.root)
        renders = [i for i in renders if (i[-3:] != 'npy' and i[-3:] != 'exr')]
        
        renders = [int(i) for i in renders]
        renders.sort()
        renders = renders[:312]
        test_set = [i for i in renders if i % 20 == 0]
        training_set = [i for i in renders if i not in test_set]

        if test:
            self.renders = test_set
        else:
            self.renders = training_set

    
        #print(self.renders)
        #renders = renders[:70]
        #renders.pop()
        #renders = [str(i) for i in renders]

    def __getitem__(self, index):
        #st = time()
        if self.shuffle_gt_and_est:
            if np.random.rand() > 0.5: # 0.5 før 
                use_gt = True
                #print('ø')
            else: 
                use_gt = False 
        else:
            use_gt = False

        render = (index // 20)
        render = str(self.renders[render])
        #print(render, 'render')
        #print(index)
        idx = str((index % 20) + 1)
        while len(idx) < 4:
            idx = '0' + idx
        
        path = os.path.join(self.root, render, 'processed_images' ,'points_' + idx)
        #print(f'path: {path}')
        #print(os.path.exists(path))
        
        while os.path.exists(path + '/' + idx + '_GT.npy') == False:
            #print(path)
            #print('path doesnt exist -- sampling a random index to use instead') 
            i = np.random.randint(1,21)
            i = str(i)
            if len(i) == 1:
                idx = idx[:-1] + i
            else:
                idx = idx[:-2] + i
            path = os.path.join(self.root, render, 'processed_images' ,'points_' + idx)
            #print('new path: ', path)
        

        if self.corrected:
            est = np.load(path + '/' + idx + '_EST_fixed.npy')
            gt = np.load(path + '/' + idx + '_GT_fixed.npy')
            corner_points = np.load(path + '/' + idx + '_labels_corrected.npy')
            if use_gt:
                est = gt
                idx = np.round(np.linspace(0, len(gt[0]) - 1, 640)).astype(int)
                est = est[:, idx]


        else:
            est = np.load(path + '/' + idx + '_EST.npy')
            gt = np.load(path + '/' + idx + '_GT.npy')
            corner_points = np.load(path + '/' + idx + '_labels.npy')
        corner_points = corner_points.T
 

        if self.noise:
            if not use_gt:
                if np.random.rand() > 0.3: # 0.3 før
                    #print('æ')
                    i = np.round(np.linspace(0, len(gt[0]) - 1, 640)).astype(int)
                    ## adds noise at the indices where the x-value of the estimate and gt differ
                    h = np.where((abs(gt[0][i] - est[0]) < 0.39) & (abs(gt[0][i] - est[0]) > 0.003))
                    ##print((h))
                    ## adds the difference mulitplied with a random scalar
                    est[2][h] = est[2][h] + np.random.uniform(0.1, 1.) * (gt[0][h] - est[0][h])
            
            # TRANSLATION     
            if not self.autoencoder:
                est, corner_points = move_points(est, corner_points)
            # ROTATION
            #est, corner_points = rotate_points(est, corner_points)
            
            est = est[1:] 
            idx = np.searchsorted(est[0], corner_points[0], side="left") 


            # AFFINE 
            if not self.autoencoder:
            
                est, corner_points = affine_transformation(est, corner_points)
                #idx = np.searchsorted(est[0], corner_points[0], side="left") 

            # REFLECTION OF GROOVE
            if not self.autoencoder:
                if np.random.rand() > 0.6:
                    try:
                        est = reflect_groove(est, idx[1:-2])
                        #print('reflect')
                    except:
                        #print(f"reflecting grooves failed for index {index}")
                        pass
                        #idx = np.searchsorted(est[0], corner_points[0][1:-2], side="left")
            

            est = line_noise(est)


            # manually added segment noise
            num_noise_segments = int(abs(np.random.normal(0, 3)))
            #print(num_noise_segments, 'noisy segments')
            #num_noise_segments = np.random.randint(0,3) # 6 før
            for _ in range(num_noise_segments):
                noise_segment_length = np.random.randint(15,19) # 8, 20 før
                est = add_noise(est, noise_segment_length, idx)

            #if np.random.rand() > 0.6:
            #    est, corner_points = move_corner(est, corner_points, idx)
#
            #    est = interpolate_groove(est)
            #
            #est, corner_points = change_x_range(est, corner_points)
        else:
            est = est[1:] 
            idx = np.searchsorted(est[0], corner_points[0], side="left") 
            #print(idx)
        if self.normalization_type == 'dataset':
            est = normalize(est)
        elif self.normalization_type == 'cloud':
            est = cloud_normalization(est)
        
        else:
            est = torch.from_numpy(est)
            est = est.type(torch.float32)
        
        corner_points = torch.from_numpy(corner_points)
        corner_points = corner_points.type(torch.float32)
        #if self.alternative_corner:
        #    return est, gt, corner_points, alt_points
        #print(time()-st, 'end time')
        

        if self.return_gt:
            i = np.round(np.linspace(0, len(gt[0]) - 1, 640)).astype(int)
            gt = gt[:, i]
            return est, gt, corner_points
        else:
            return est, corner_points


    def __len__(self):
        #renders = os.listdir(self.root)
        #renders = [i for i in renders if (i[-3:] != 'npy' and i[-3:] != 'exr')]
        #
        #renders = [int(i) for i in renders]
        #renders.sort()
        ##print(renders)
        #renders = renders[50:108]
        ##renders.pop()
        #renders = [str(i) for i in renders]  
        
        return len(self.renders) * 20


if __name__ == "__main__":
    dataset = LaserPointDataset('/home/oyvind/Blender-weldgroove/render', noise=True, return_gt=True, corrected=True, normalization='', test=False, shuffle_gt_and_est=True)
    #dataset2 = LaserPointDataset('/home/oyvind/Blender-weldgroove/render', noise=False, return_gt=True, corrected=False, normalization='')
    #print(dataset[17])
    print(len(dataset), 'dataset len')



    num = np.random.randint(0, len(dataset))
    for i in range(5, len(dataset), 20): # 138
        #num = 2646
        print(i)

        e, g, p = dataset[i]

        #print(p)
        e = e * 1000
        #e2 = e2*1000
        p = p * 1000
        #p = p.T
        plt.scatter(e[0], e[1], s=1, color='g')
        g = g[1:] * 1000
        print(g.shape)
        plt.scatter(g[0], g[1], s=1, color='r')
        #plt.scatter(e2[0], e2[1], s=1, color='r')
        plt.scatter(p[0], p[1], s = 20)
        #plt.scatter(e[0][i], e[1][i], s= 40)
        #plt.scatter(g[0][i], g[1][i], s= 40)
        plt.show()
        
        #print(e[1][:10])
        #print(g[1][:10])

    