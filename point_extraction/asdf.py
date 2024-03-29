from numpy import core, short
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

class GTPointExtractionDataset(Dataset):
    def __init__(self, root, corrected=False, remove_outliers=False):
        self.root = root
        self.corrected = corrected
        self.remove_outliers = remove_outliers

    def __getitem__(self, index):
        render = str((index // 20) + 1)
        idx = str((index % 20) + 1)
        while len(idx) < 4:
            idx = '0' + idx
        
        path = os.path.join(self.root, render, 'processed_images' ,'points_' + idx)

        #if not self.remove_outliers:
        while os.path.exists(path + '/' + idx + '_GT.npy') == False:
            print('path doesnt exist -- sampling a random index to use instead') 
            i = np.random.randint(1,21)
            i = str(i)
            if len(i) == 1:
                idx = idx[:-1] + i
            else:
                idx = idx[:-2] + i
            path = os.path.join(self.root, render, 'processed_images' ,'points_' + idx)
            print('new path: ', path)
        

        if self.corrected:
            #print("corrected")
            gt = np.load(path + '/' + idx + '_GT_fixed.npy')
            est = np.load(path + '/' + idx + '_EST_fixed.npy')
        else:
            gt = np.load(path + '/' + idx + '_GT.npy')
            est = np.load(path + '/' + idx + '_EST.npy')
        return gt, est


    def __len__(self):
        renders = os.listdir(self.root)
        renders = [i for i in renders if (i[-3:] != 'npy' and i[-3:] != 'exr')]

        renders = [int(i) for i in renders]
        renders.sort()
        #renders.pop()
        renders = [str(i) for i in renders]  

        return len(renders) * 20


if __name__ == "__main__":
    """
    New, improved method of ground truth extraction.
    """
    corrected = True

    root = '/home/oyvind/Blender-weldgroove/render'
    dataset = GTPointExtractionDataset(root, corrected)
    #print('length dataset:', len(dataset))
    dataset[5]
    img = 5860#2209 # 3560 first images ok
    h = img
    # extract only ground-truth from the datataset
    while img < len(dataset):
        print('\n','i = ',img, '\n')
        g, est = dataset[img]

        # converts to [mm]
        g = g[1:] * 1000
        # finite differences
        d = []
        sd = []
        for i in range(len(g[0])):
            if i == 0:
                d.append((g[1][i+1] - g[1][i]) / (g[0][i+1] - g[0][i]))
                sd.append(g[1][i+2] - 2 * g[1][i+1] + g[1][i])
            elif i == len(g[0]) - 1:
                d.append((g[1][i] - g[1][i-1]) / (g[0][i] - g[0][i-1]))
                sd.append(g[1][i] - 2 * g[1][i - 1] + g[1][i - 2])
            else:
                d.append((g[1][i+1] - g[1][i-1]) / ((g[0][i+1] - g[0][i]) + (g[0][i] - g[0][i-1])))
                sd.append(g[1][i + 1] - 2 * g[1][i] + g[1][i-1])
        #print(len(d))

        sd = np.array(sd)
        mean_sd = np.mean(sd)  
        std = np.std(sd)
        #print(mean_sd)
        #print(std)

        render = (img // 20) + 1
        print('render:', render)

        scalar = 3.5 # vanligvis 3.5; spesielle tilfeller ---- render 77: 2.8; render 103: 2.5;
        mas = [103, 104, 108, 109, 110, 114, 117, 118, 123, 128, 129, 133,
               146, 159, 160, 161, 170, 174 ,175, 179, 183, 189, 200, 222, 
               228, 236, 239, 251, 252, 259, 278, 280, 281, 284]
        if render in mas:
            print('special case - reducing scalar value')
            scalar = 2.5
        elif render == 115 or render == 140 or render == 221 or render == 273:
            scalar = 1.5
        elif render == 77:
            scalar = 2.8
        elif render == 149:
            scalar = 4.5
        elif render == 193 or render == 202 or render == 205 or render == 275:
            scalar = 4.0
        elif render == 285:
            scalar = 2.
        
        
        h = np.where(abs(sd) > mean_sd + scalar * std)
        #print(sd[h])

        min_index = 100
        if render == 94:
            min_index = 300
        h = h[0]
        h = [i for i in h if i > min_index] # fails if early points are falsely identified as candidates

        h_len = 10 # vanligvis 10

        extra_len_img = [1583, 1709, 2178, 2209, 3093]
        longer_len_render = [273, 284, 285]
        extra_len_render = [114, 128, 146, 153, 160, 282, 292, 293]
        shorter_len_render = [143]

        if img in extra_len_img or render in extra_len_render:
            print('asdfost')
            h_len = 30
        elif img == 2120 or img == 2137 or img == 3085 or render in shorter_len_render:
            h_len = 7
        elif render in longer_len_render:
            h_len = 17
        elif render == 294:
            h_len = 4
        
        while len(h) > h_len: # spesielle tilfeller ---- img 1583: 30; img 1709: 30;
            print(h, 'asdf')
            print('an excessive amount of corner points - increasing requirements...')
            if render > 174:
                scalar += 0.1
            else:
                scalar += 0.3
            h = np.where(abs(sd) > mean_sd + scalar * std)
            h = h[0]
            h = [i for i in h if i > min_index]
        h_std = np.std(h)
        print(h_std)
        old_h = h

        reduced_scalar = False

        high_std = [110, 112]
        low_std = [158, 196, 236, 237, 264]
        lower_std = [170, 171, 174, 176, 195, 215, 227, 245, 254]
        min_standard_deviation = 23 # 23
        if render in high_std:
            min_standard_deviation = 30
        elif render in low_std:
            min_standard_deviation = 20
        elif render in lower_std:
            min_standard_deviation = 15
        
        while h_std < min_standard_deviation: # 23
            print('too few potential corner points - reducing requirements...')
            if render > 174:
                scalar -= 0.1
            else:
                scalar -= 0.5
            h = np.where(abs(sd) > mean_sd + scalar * std)
            h = h[0]
            h = [i for i in h if i > min_index]
            new_points = [i for i in h if i not in old_h]
            h_std = np.std(h)
            print(h_std)
            reduced_scalar = True

        if reduced_scalar:
            new_h = [i for i in old_h]
            for point in new_points:
                different_indices = np.array(old_h)[np.where(abs(old_h - point) > 5)[0]] 
                if len(different_indices) > len(old_h) * 0.9:
                    new_h.append(point)
            h = sorted(new_h)
        #else:
        #   h = h.tolist()


        corner_points = []
        print((h))
        
        sd_diff = 1.47
        lower_sd_diff = [108, 109, 110, 111, 112, 113, 114, 115, 117, 118] #120, 121, 122, 123, 124]
        special_case_sd_diff = [137, 176, 177, 179, 180, 183, 187, 195, 196, 
                                198, 199, 201, 214, 215, 218, 222, 227, 234, 241, 250,
                                254, 255, 261, 263, 264, 270, 272, 275, 289]
        if render in special_case_sd_diff:
            sd_diff = 1.
        elif render in lower_sd_diff or render > 120:
            sd_diff = 0.35

        
        print(sd_diff)
        
        while len(corner_points) < 3:
            #print('asdf')
            if len(corner_points) == 0:
                #print('hhh')
                i = 0
                while h[i+1] - h[i] < 3:
                    if g[1][h[i+1]] > g[1][h[i]]:
                        h.pop(i)
                    else:
                        h.pop(i+1)
                corner_points.append(h[0])
                print(h)
            elif len(corner_points) == 1:
                #print(h)
                i = 1
                removed = 0
                while h[i+1] - h[i] < h_std + 10 and h[-1] - h[i+1] > 5:
                    #print(h[i+1])
                    #print(h[i])
                    #print(removed)
                    if h[i+1] - h[i] < 127 + removed: #33 første 179
                        #print(abs(g[1][h[i] + 1] - g[1][h[i]]))
                        #print(abs(g[1][h[i+1] + 1] - g[1][h[i+1]]))
                        #print(removed / len(h), 'øøøøøø')
                        if len(h) > 5 and removed / len(h) < 0.55: # if most of the candidates already are removed the scalar s is increased
                            s = 1.
                        else:
                            s = 3
                        if g[1][h[i+1]] > g[1][h[i]] or (abs(g[1][h[i] + 1] - g[1][h[i]]) * s < abs(g[1][h[i+1] + 1] - g[1][h[i+1]])):
                            print('asdfasdf') 
                            #print(abs(g[1][h[i+1]] - g[1][h[i]]) * 3)
                            #print(abs(g[1][h[i+2]] - g[1][h[i+1]]))
                            print('removing', h[i])
                            h.pop(i)
                            removed += 1
                        elif abs(sd[h[i+1]]) > abs(sd[h[i]]) * sd_diff: #1.47
                            print(abs(sd[h[i+1]]))
                            print(abs(sd[h[i]]))
                            print('jalla')
                            print('removing:', h[i])
                            h.pop(i)
                            removed += 1
                        else:
                            #print(abs(sd[h[i+1]]))
                            #print(abs(sd[h[i]]))
                            print('asf')
                            print('removing', h[i+1])
                            h.pop(i+1)
                            removed += 1
                    else:
                        #print('removing: ', h[i] )
                        h.pop(i)
                        if removed > 0:
                            removed -= 1
                    print(h)
                corner_points.append(h[1])
            
            else:
                corner_points.append(h[-1])
            
        
        print(corner_points)
                

         # finds the corner points cooordinates, these are then used to define two lines
        # whose intersection defines the final fourth corner point.
        groove_corners = g.T[corner_points]
        edge_point_brace = g[:,-5]
        direction_vector_brace = groove_corners[-1] - edge_point_brace

        direction_vector_brace = (direction_vector_brace / np.linalg.norm(direction_vector_brace))
        norm_vector_brace = np.array([-direction_vector_brace[1], direction_vector_brace[0]])
        #print(norm_vector_brace.shape)

        c = norm_vector_brace @ groove_corners[-1]

        #print(c)
        line_brace = np.append(norm_vector_brace, c)
        #print(line_brace)

        edge_point_leg = g[:,corner_points[0] - 30]
        direction_vector_leg = groove_corners[0] - edge_point_leg
        direction_vector_leg = (direction_vector_leg / np.linalg.norm(direction_vector_leg))
        norm_vector_leg = np.array([-direction_vector_leg[1], direction_vector_leg[0]])

        c2 = norm_vector_leg @ groove_corners[0]

        line_leg = np.append(norm_vector_leg, c2)

        #print(line_leg)

        cross_point = (np.cross(line_brace, line_leg))
        cross_point = -cross_point / cross_point[-1]

        # find point in gt closest to cross point
        cp = np.repeat(cross_point[:2], len(g[0]))
        cp = cp.reshape(2,-1)
        gt_to_cross_point = g - cp
        gt_to_cross_point = np.linalg.norm(gt_to_cross_point, axis=0)
        closest = np.argmin(gt_to_cross_point)

        cross_point = g[:,closest]
        
        groove_corners = np.vstack((cross_point, groove_corners))
        groove_corners = np.vstack((groove_corners, g[:,-1]))
        #print(groove_corners)
    # groove_corners = groove_corners.reshape(2,-1)
        #print(groove_corners.shape)
        
        #plt.scatter(cross_point[0], cross_point[1], s=20, color='r')

        
        #if img % 1 == 0:
        #    plt.scatter(g[0][corner_points[0]-30:corner_points[-1] + 10], g[1][corner_points[0]-30:corner_points[-1] + 10], s=1)
        #    plt.scatter(groove_corners[:,0][1:-1], groove_corners[:,1][1:-1], s=20, color='g')
        #    plt.show()
        #break
        
        est = est[1:] * 1000
        
        
        if img % 1 == 0:
            plt.scatter(g[0], g[1], s=1, color="r")
            plt.scatter(est[0], est[1], s=1)
            plt.scatter(groove_corners[:,0], groove_corners[:,1], s=20, color='g')
            plt.show()
        #
        
       
       # np.save()

        render = str((img // 20) + 1)
        idx = str((img % 20) + 1)
        while len(idx) < 4:
            idx = '0' + idx
        
        # returns the units back to [m]
        groove_corners = groove_corners / 1000
        path = os.path.join(root, render, 'processed_images' ,'points_' + idx)

        if corrected:
            np.save(path + '/' + idx + '_labels_corrected.npy', groove_corners)
        else:
            np.save(path + '/' + idx + '_labels.npy', groove_corners)
        img += 1
    #plt.show()
