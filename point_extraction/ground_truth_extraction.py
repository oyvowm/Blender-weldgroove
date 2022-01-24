from numpy import core
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

class GTPointExtractionDataset(Dataset):
    def __init__(self, root, corrected=False):
        self.root = root
        self.corrected = corrected

    def __getitem__(self, index):
        render = str((index // 20) + 1)
        idx = str((index % 20) + 1)
        while len(idx) < 4:
            idx = '0' + idx
        
        path = os.path.join(self.root, render, 'processed_images' ,'points_' + idx)
        if self.corrected:
            print("corrected")
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

    corrected = True

    root = '/home/oyvind/Blender-weldgroove/render'
    dataset = GTPointExtractionDataset(root, corrected)
    print('length dataset:', len(dataset))
    img = 268 # 1340 first images ok
    h = img
    # extract only ground-truth from the datataset
    while img < len(dataset):
        print('i = ',img)
        g, est = dataset[img]

        # converts to [mm]
        g = g[1:] * 1000
        # finite differences
        d = []
        for i in range(len(g[0])):
            if i == 0:
                d.append((g[1][i+1] - g[1][i]) / (g[0][i+1] - g[0][i]))
            elif i == len(g[0]) - 1:
                d.append((g[1][i] - g[1][i-1]) / (g[0][i] - g[0][i-1]))
            else:
                d.append((g[1][i+1] - g[1][i-1]) / ((g[0][i+1] - g[0][i]) + (g[0][i] - g[0][i-1])))

        
        
        num_corner_points = 0
        ite = 0
        desired_corner_points = 9 # == 9 for first 67 renders
        while num_corner_points < desired_corner_points:
            corner_points = []
            i = 0
            while i < len(d) - 1: 
                last = abs(d[i] - d[i-1])
                #current = abs(d[i+1] - d[i]) # test ut ala: 
                current = abs((d[i+1] - d[i] + d[i] - d[i-1]) / 2)
                print(current)
                #if current > 0.7:
                    #print(current)
                #print(f'last: {last} current: {current}')
                if (current > 6 + ite):# and abs(current - last) > 0.5:
                    corner_points.append(i)
                    i+=1
                else:
                    i+=1
            num_corner_points = len(corner_points)
            #print(num_corner_points)

            #print(num_corner_points)
            if num_corner_points > 15:
                ite += 0.35
                num_corner_points = 0
                #print()
            else:
                ite -= 0.04
            #print(num_corner_points)

      

        #print(corner_points)

        # filters out non-corner points
        j = 1
        while len(corner_points) > 3:
            i = 0
            while i < (len(corner_points) - 1):
                
                if corner_points[i+1] - corner_points[i] < j:
                    if j < 25:
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

                            else: # i > 0:
                                if abs(y_vals[:,1][1] - y_vals[:,1][0]) > 2.:
                                    y_max = np.argmax(y_vals, axis=0)[1]
                                    corner_points[i+1] = corner_indices[y_max]
                                    corner_points.pop(i)  
                                else:
                                    corner_points.pop(i) 
                        else:
                            corner_points.pop(i)       
                    else:
                        corner_points.pop(i)
                        
                else:
                    i+=1
                #print(len(corner_points))

            j += 1


        
        #corner_points = [corner_points[0], corner_points[-2], corner_points[-1]]

        # moves the first corner point towards the left i t
        #print(g.T[corner_points[0] - 1][1])
        #print(g.T[corner_points[0]][1])
        #print(corner_points)
        while (abs(g.T[corner_points[0] - 1][1] - g.T[corner_points[0]][1]) < 0.50):

            #print('adfgadsfg')
            #print(abs(d[corner_points[0] - 1]) - abs(d[corner_points[0]]))
            avg_deriv = 0
            avg_y = g.T[corner_points[0] - 4:corner_points[0] - 1]
            avg_y = np.average(avg_y[:,1])
            #print('average y:' , avg_y)
            for i in range(1, 10):
                avg_deriv = avg_deriv + (d[corner_points[0] - i] - d[corner_points[0] - (i+1)])
            #print(avg_deriv)
            if (abs(avg_deriv) < 0.7) and (avg_y <= g.T[corner_points[0]][1]):
                print(avg_deriv)
                if g.T[corner_points[0] - 1][1] - g.T[corner_points[0]][1] < 0.05:
                    #print('slope too small, breaking loop')
                    break
            if abs(d[corner_points[0] - 1]) - abs(d[corner_points[0]]) > 0.7 and g.T[corner_points[0] - 1][1] < g.T[corner_points[0]][1] and abs(avg_deriv) < 0.5:
                #print('jadda')
                break
            if (avg_y > g.T[corner_points[0]][1] - 0.5) or (abs(g.T[corner_points[0] - 1][1] - g.T[corner_points[0] - 2][1]) > 3 * abs(g.T[corner_points[0]][1] - g.T[corner_points[0] - 1][1])):
                print('moving corner point')
                corner_points[0] = corner_points[0] - 1
            else:
                break
            #if abs(g.T[corner_points[1] - 1][1] - g.T[corner_points[1] -2][1]) < abs(g.T[corner_points[1] + 1][1] - g.T[corner_points[1]][0]):
        
        #print(corner_points)

        if len(corner_points) < 3:
            #print('too few corner points, duplicating the first point')
            corner_points.insert(1, corner_points[0])
        # moves the second corner point towards the left if it is on the downward slope
        steps = 0
        
        dist_right = np.linalg.norm(g.T[corner_points[1] + 1] - g.T[corner_points[1]])
        while ((g.T[corner_points[1] - 1][1] - g.T[corner_points[1]][1]) > 0.11) and (((g.T[corner_points[1]][1] - g.T[corner_points[1] + 1][1]) / (g.T[corner_points[1] - 1][1] - g.T[corner_points[1]][1])) < 7):
            print('æ')
            if abs(g.T[corner_points[1] - 1][1] - g.T[corner_points[1] -2][1]) < abs(g.T[corner_points[1] + 1][1] - g.T[corner_points[1]][0]):
                dist_one_left = np.linalg.norm(g.T[corner_points[1] - 1] - g.T[corner_points[1]])
                dist_two_left = np.linalg.norm(g.T[corner_points[1] - 2] - g.T[corner_points[1] - 1])
                #print(dist_two_left)
                #print(dist_one_left)
                avg_y = g.T[corner_points[1] - 4:corner_points[1] - 1]
                avg_y = np.average(avg_y[:,1])
                if dist_one_left > dist_right * 0.15 and dist_two_left < dist_one_left * 0.75:
                    corner_points[1] = corner_points[1] - 1
                    steps += 1
                    print('eeeee')
                elif abs((dist_two_left / dist_one_left) / (dist_one_left / dist_right)) > 0.8 and abs((dist_two_left / dist_one_left) / (dist_one_left / dist_right)) < 1.2:
                    #print()
                    #print((dist_two_left / dist_one_left))
                    #print((dist_one_left / dist_right))
                    #print(abs((dist_two_left / dist_one_left) / (dist_one_left / dist_right)))
                    #print("slope approx. constant")
                    corner_points[1] = corner_points[1] - 1
                elif avg_y > g.T[corner_points[1]][1]:
                    #print('avg to the left larger than y')
                    corner_points[1] = corner_points[1] - 1
                else:
                    #print('breaking')
                    break
            else:
                #slope_diff -= 0.01
                #print('breaking')
                break

        #print(corner_points)
        
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

          
        if img % 1 == 0:
            plt.scatter(g[0][corner_points[0]-30:corner_points[-1] + 10], g[1][corner_points[0]-30:corner_points[-1] + 10], s=1)
            plt.scatter(groove_corners[:,0][1:-1], groove_corners[:,1][1:-1], s=20, color='g')
            plt.show()
        
        
        est = est[1:] * 1000
        
        """
        if img % 1 == 0:
            plt.scatter(g[0], g[1], s=1)
            #plt.scatter(est[0], est[1], s=1)
            plt.scatter(groove_corners[:,0], groove_corners[:,1], s=20, color='g')
            plt.show()
        """
        
       
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