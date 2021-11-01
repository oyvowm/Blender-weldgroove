import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

class GTPointExtractionDataset(Dataset):
    def __init__(self, root):
        self.root = root

    def __getitem__(self, index):
        render = str((index // 20) + 1)
        idx = str((index % 20) + 1)
        while len(idx) < 4:
            idx = '0' + idx
        
        path = os.path.join(self.root, render, 'processed_images' ,'points_' + idx)
        gt = np.load(path + '/' + idx + '_GT.npy')

        return gt


    def __len__(self):
        renders = os.listdir(self.root)
        renders = [i for i in renders if (i[-3:] != 'npy' and i[-3:] != 'exr')]
        return len(renders) * 20


if __name__ == "__main__":
    root = '/home/oyvind/Blender-weldgroove/render'
    dataset = GTPointExtractionDataset(root)
    img = 0
    # extract only ground-truth from the datataset
    while img < len(dataset):
        print('i = ',img)
        g = dataset[img]

        g = g[1:] * 1000

        # finite differences
        d = []
        for i in range(len(g[0])):
            if i == 0:
                d.append((g[1][i+1] - g[1][i]) / (g[0][i+1] - g[0][i]))
            elif i == 639:
                d.append((g[1][i] - g[1][i-1]) / (g[0][i] - g[0][i-1]))
            else:
                d.append((g[1][i+1] - g[1][i-1]) / ((g[0][i+1] - g[0][i]) + (g[0][i] - g[0][i-1])))


        #plt.scatter(g[0],d, s=1, color='b')

        # find the points where the derivatives changes most
        
        
        num_corner_points = 130
        ite = 0
        while num_corner_points > 10:
            corner_points = []
            i = 0
            while i < len(d) - 1: 
                
                if abs(d[i+1] - d[i]) > 0.4 + ite:
                    corner_points.append(i)
                    
                    i+=1
                else:
                    i+=1
            num_corner_points = len(corner_points)
            ite += 0.1



        # filters out non-corner points
        j = 1
        while len(corner_points) > 3:
            i = 0
            while i < (len(corner_points) - 1):
                if corner_points[i+1] - corner_points[i] < j:
                    if j < 18:
                        if i+2 != len(corner_points):
                            corner_indices = [corner_points[i], corner_points[i+1]]
                            y_vals = g.T[corner_indices]
                            if i == 0:
                                if abs(y_vals[:,1][1] - y_vals[:,1][0]) > 0.001:
                                    y_max = np.argmax(y_vals, axis=0)[1]
                                    corner_points[i] = corner_indices[y_max]
                                    corner_points.pop(i+1)
                                else:
                                    corner_points.pop(i+1)

                            elif i > 0:
                                if abs(y_vals[:,1][1] - y_vals[:,1][0]) > 1.5:
                                    y_max = np.argmax(y_vals, axis=0)[1]
                                    corner_points[i] = corner_indices[y_max]
                                    corner_points.pop(i+1)
                                #elif abs(y_vals[:,1][1] - y_vals[:,1][0]) < 0.2:
                                #    corner_points.pop(i)
                                #    print('asdfsdf', img)
                                
                                else:
                                    corner_points.pop(i)
                                
                        else:
                            corner_points.pop(i)
                    else:
                        corner_points.pop(i)
                else:
                    i+=1

            j += 1
 

        #corner_points = [corner_points[0], corner_points[-2], corner_points[-1]]

        if abs(g.T[corner_points[1]][1] - g.T[corner_points[1] - 1][1]) > 0.3:
            if abs(g.T[corner_points[1] - 1][1] - g.T[corner_points[1] -2][1]) < abs(g.T[corner_points[1] + 1][1] - g.T[corner_points[1]][0]):
                corner_points[1] = corner_points[1] - 1

        #if abs(g.T[corner_indices[0]][1] - g.T[corner_indices[0] - 1][1]) > 0.3:
        #    if abs(g.T[corner_indices[0] - 1][1] - g.T[corner_indices[0] - 2][1]) < abs(g.T[corner_indices[0] + 1][1] - g.T[corner_indices[0]][1]):
        #        corner_points[i] = corner_points[i] - 1
        #assert len(corner_points) == 3, f'ERROR: the number of corner points extracted using finite differences should equal 3 {corner_points}'
            
        #hei = [d[i] for i in corner_points]
        #xs = [g[0][i] for i in corner_points]
        #plt.scatter(xs, hei, s=1, color='g')
    #    plt.show()

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
        cp = np.repeat(cross_point[:2], 640)
        cp = cp.reshape(2,-1)
        gt_to_cross_point = g - cp
        gt_to_cross_point = np.linalg.norm(gt_to_cross_point, axis=0)
        closest = np.argmin(gt_to_cross_point)

        cross_point = g[:,closest]
        
        groove_corners = np.vstack((cross_point, groove_corners))
        groove_corners = np.vstack((groove_corners, g[:,-1]))
    # groove_corners = groove_corners.reshape(2,-1)
        #print(groove_corners.shape)
        
        #plt.scatter(cross_point[0], cross_point[1], s=20, color='r')

       # if img % 1 == 0:
       #     plt.scatter(g[0], g[1], s=1)
       #     plt.scatter(groove_corners[:,0], groove_corners[:,1], s=20, color='g')
            #plt.show()
       
       # np.save()

        render = str((img // 20) + 1)
        idx = str((img % 20) + 1)
        while len(idx) < 4:
            idx = '0' + idx
        
        path = os.path.join(root, render, 'processed_images' ,'points_' + idx)
        np.save(path + '/' + idx + '_labels.npy', groove_corners)
        img += 1
    #plt.show()