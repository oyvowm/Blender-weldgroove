from audioop import avg
from tkinter import E
from turtle import update
from types import new_class
from cv2 import projectPoints
from matplotlib import lines
import numpy as np
import torch
import matplotlib.pyplot as plt
import model
import resnet
import os
import time
import copy
from torchvision import transforms
import loss_functions


from sklearn.linear_model import HuberRegressor

net = model.SimpleNetwork23b()
#net = resnet.ResidualNetwork()
state_dict = torch.load('/home/oyvind/Blender-weldgroove/ResNet_NewSet62.pth')
net.load_state_dict(state_dict['model_state_dict'])



grooves = os.listdir('/home/oyvind/Downloads/grooves')
print(grooves)
print(grooves[1:2])
for groove in grooves[:]:
    groove_start_time = time.time()

    path = '/home/oyvind/Downloads/grooves/' + groove
    #print(path)
    hei = np.load(path)
    chei = hei
    hei = np.flip(hei, axis=1)


    hei = hei/1000
    hei = torch.from_numpy(np.array(hei))
    hei = hei.type(torch.float32)
    t = transforms.Normalize((0.0017, 0.2328), (0.0310, 0.0198))
    hei = hei.unsqueeze(0)
    hei = hei.permute(1, 0, 2)
    hei = t(hei)
    hei = hei.squeeze()


    hei = hei.unsqueeze(0)

    
    net.eval()
    net_start = time.time()
    with torch.no_grad():
        ut = net(hei)
    net_time = time.time() - net_start

    hmm = ut.detach().numpy()
    hmm = hmm*1000
    hmm = hmm.squeeze()
    

    cheiflip = np.flip(chei, axis=1)
    for iterations in range(7):
        print(iterations)
        idx = np.searchsorted(cheiflip[0], hmm.T[0], side="left") 
        start = time.time()
        
        for i in range(len(idx)-1):
            segment1 = cheiflip[:,idx[i]:idx[i+1]-1]
            #segment1 = np.c_[hmm[i],segment1, hmm[i+1]]
            #if len(segment1[0]) > 175:
            #    print(len(segment1[0]))
            #    #x = segment1[0, int(len(segment1[0]) / 1.5):]
            #    #y = segment1[1, int(len(segment1[0]) / 1.5):]
            #    x = segment1[0, -100:]
            #    y = segment1[1, -100:]
            #else:
            x = segment1[0]
            y = segment1[1]

            point1 = np.append(hmm[i], 1)
            point2 = np.append(hmm[i+1], 1)
            #point1 = np.array([x[0], y[0], 1])
            #point2 = np.array([x[-1], y[-1], 1])
            line = np.cross(point1, point2)
            line[0] = -line[0] / line[1]
            line[2] = -line[2] / line[1]
            line[1] = -1.0
            slope_est, intercept_est = line[0], line[2]

            
            # noise removal
            y_est = [slope_est * i + intercept_est for i in x]


            y_diff = y - y_est


            if (i == 0 or i == 2) and iterations == 0:
                #print('qwtqwe')
                non_noise = np.where(abs(y_diff - np.median(y_diff)) < (1.))
                #print(non_noise)
                #print(x)
                x = x[non_noise]
                y = y[non_noise]
            #y_est_new = y_est + np.average(y_diff)

            else:
                #print('æææ')
                if i == 1:
                    noise_threshold = 0.7 - 0.1 * iterations
                    limit = 0.85
                    reduction = 0.10
                else:
                    noise_threshold = 2. - 0.3 * iterations
                    limit = 0.2
                    reduction = 0.03
                #noise_threshold = abs(np.average(y_diff)) * 0.5**(iterations+1)
                #print(noise_threshold,'noise threshold')
                #non_noise = np.where(abs(y - y_est_new) < noise_threshold)
                non_noise = np.where(abs(y_diff) < noise_threshold)
           

                #print(len(non_noise[0]), 'num points used for linear regression', len(x))

                
                
                while len(non_noise[0]) < len(x) * (limit - (iterations * reduction)):
                    #print(len(non_noise[0]),'len1')
                    #print(len(x),'len2')
                    noise_threshold += 0.05
                    #print(noise_threshold)
                    #non_noise = np.where(abs(y - y_est_new) < noise_threshold)
                    non_noise = np.where(abs(y_diff) < noise_threshold)

                #print(non_noise, 'before')
                if i == 0:
                    non_noise = non_noise[0][:-3]
                elif i == 2:
                    non_noise = non_noise[0][3:]
                #print(non_noise, 'after')
                #print(len(non_noise[0]),'final len1')


                #print(noise_threshold)
                x = x[non_noise]
                y = y[non_noise]
                #print(x.shape,'x')
            

            #huber = HuberRegressor().fit(x.reshape(-1,1), y.reshape(-1,1))
            #end_points = np.array([x[0], x[-1]])
            #line_values = huber.predict(end_points.reshape(-1,1))
            #slope = (line_values[-1] - line_values[0]) / (x[-1] - x[0])
            slope, intercept = np.polyfit(x, y, 1)
            

            # just to plot the estimated line segment
            
            #line = np.array([line[0], line[2]])

        

            line_values = [slope * i + intercept for i in x]
            line_values_est = [slope_est * i + intercept_est for i in [point1[0], point2[0]]]
            xs = [point1[0], point2[0]]
            
            
            mean_x = np.mean(x)
            mean_y = np.mean(y)
            center_x = 0.5 * (point1[0] + point2[0])
            center_y = 0.5 * (point1[1] + point2[1])
            diff_x = mean_x - center_x
            diff_y = mean_y - center_y

            corners = np.vstack((np.array([point1[0], point2[0]]), np.array([point1[1], point2[1]])))
            
            old_corners = copy.deepcopy(corners)
            
            corners[0] = corners[0] + diff_x
            corners[1] = corners[1] + diff_y
            
            corners[0] = corners[0] - mean_x
            corners[1] = corners[1] - mean_y

            angle = np.arctan((slope - slope_est) / (1 + slope * slope_est))
            #print(angle, 'angle')
            if i == 1:
                if iterations == 0:
                    angle = 0
                else:
                    if angle > 0.2:
                        angle = angle / 2

            rot = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])

            corners = rot @ corners
            
            corners[0] = corners[0] + mean_x
            corners[1] = corners[1] + mean_y

            corners_homogenous = np.vstack((corners, np.ones((1,2))))
            new_line = np.cross(corners_homogenous[:,0], corners_homogenous[:,1])
            new_line[0] = -new_line[0] / new_line[1]
            new_line[2] = -new_line[2] / new_line[1]
            new_line[1] = -1.0


            

            #plt.scatter(x, y, s = 10)
            #plt.plot(x, line_values, c='r')
            #plt.plot(xs, line_values_est)
            #plt.scatter(old_corners[0], old_corners[1], c='g')
            #plt.scatter(corners[0], corners[1], c='m')
            ##plt.show()
##
            #plt.scatter(chei[0], chei[1],s=1)
            #plt.scatter(hmm[:,0], hmm[:,1], s=20) # når output (B, 5, 2)
            #plt.show()
            if i == 0:
                updated_corners = corners
                #line_segments = np.array([line[0], line[2]])
                line_segments = np.array(new_line)
                #print(np.array([line[0], line[2]]))

            else:
                updated_corners = np.c_[updated_corners, corners]
                #line_segments = np.vstack((line_segments, np.array([line[0], line[2]])))
                line_segments = np.vstack((line_segments, new_line))
                #print(np.array([line[0], line[2]]))

            

        print(time.time() - start, 'time')
        #plt.show()


        #print(updated_corners)    
        #print(line_segments)

        l1 = np.vstack((line_segments[0], line_segments[:-1]))
        l2 = np.vstack((line_segments[-1], line_segments[1:]))
        new_output = np.cross(l1, l2)
        #new_output = np.cross(line_segments[:-1], line_segments[1:])
        new_corners = new_output[:,:-1] / new_output[:,-1][:,None]
        new_corners = np.vstack((new_corners, updated_corners[:,-1]))
        
        
        #print(new_corners)

        # find the average distance each point has been moved
        avg_dist = np.average(np.linalg.norm(hmm - new_corners, axis=1)[:-1])
        print(f'average distance moved: {avg_dist}')
#
        #plt.scatter(chei[0], chei[1], s=1)
        #plt.scatter(new_corners[:,0], new_corners[:,1], s=40, c='g')
        #plt.scatter(hmm[:,0], hmm[:,1], s=20, c='r')
        #plt.show()
        if avg_dist < 0.2:
            plt.clf()
            plt.scatter(chei[0], chei[1], s=1)
            plt.scatter(new_corners[:,0], new_corners[:,1], s=20, c='g')
            plt.show()
            #path = groove + '_corrected.png'
            #plt.savefig('results/corrected/' + path)
            print(f'\n groove iterative correction time = {time.time() - groove_start_time} \n')
            print(f'net time: {net_time}')
            break
        hmm = new_corners
        


    