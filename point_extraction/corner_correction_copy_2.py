from audioop import avg
from dis import dis
from tkinter import E
from turtle import update
from types import new_class
#from cv2 import projectPoints
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



class LineSegment():
    def __init__(self, path, groove, corner_points, iteration, display_line_segments=False, labels=np.zeros((1,1))):
        self.groove = groove
        self.orig = corner_points
        self.corner_points = corner_points
        self.iteration = iteration
        self.noise = np.array([])

        self.dist_moved = np.ones(5)
        self.angle = np.ones(4)
        self.line_segments = np.zeros((4, 3))

        self.display_line_segments = display_line_segments



        for i in range(7):
            #print(i)
            self.update(i)
            avg_dist = np.average(self.dist_moved)
            #print(avg_dist, 'average distance moved')
            if avg_dist < 0.08 + i * 0.01:
                plt.clf()
                
                plt.scatter(groove[0], groove[1], s=1, c='gray') # original groove
                plt.scatter(self.groove[0], self.groove[1], s=1, c='g')
                
                plt.scatter(self.orig[:,0], self.orig[:,1], s=15, c='b')
                plt.scatter(self.corner_points[:,0], self.corner_points[:,1], s=15, c='r')
                if len(labels) == 5:
                    plt.scatter(labels[:,0], labels[:,1], s=10, c='yellow')
                    #plt.scatter(self.orig[:,0], self.orig[:,1], s=20)
                #try:
                #    plt.scatter(self.noise[0], self.noise[1], s=1)
                #except:
                #    print('no noise')
                
                plt.xlabel('x [mm]')
                plt.ylabel('y [mm]')
                plt.grid()
                
                save_path = path[:-4] + '_corrected.png'
                #print(save_path,'pathhhhh')
                #plt.savefig('C:/Users/oyvin/Desktop/groove_images/' + save_path)


                #plt.show()



                plt.clf()
               
                plt.scatter(groove[0][self.idx[0] - 20:self.idx[-2] + 20], groove[1][self.idx[0] - 20:self.idx[-2] + 20], s=1, c='gray') # original groove
                plt.scatter(self.groove[0][self.idx[0] - 20:self.idx[-2] + 20], self.groove[1][self.idx[0] - 20:self.idx[-2] + 20], s=1, c='g')
                
                plt.scatter(self.orig[:-1,0], self.orig[:-1,1], s=15, c='b')
                plt.scatter(self.corner_points[:-1,0], self.corner_points[:-1,1], s=15, c='r')
                if len(labels) == 5:
                    plt.scatter(labels[1:,0], labels[1:,1], s=10, c='yellow')
                #plt.scatter(self.orig[:,0], self.orig[:,1], s=20)
                #try:
                #    plt.scatter(self.noise[0], self.noise[1], s=1)
                #except:
                #    print('no noise')
                
                plt.xlabel('x [mm]')
                plt.ylabel('y [mm]')
                plt.grid()

                save_path = path[:-4] + '_corrected_close_up.png'
                #plt.savefig('C:/Users/oyvin/Desktop/groove_images/' + save_path)


                #plt.show()
                break
            
    def divide_into_segments(self):
        self.idx = np.searchsorted(self.groove[0], self.corner_points.T[0], side="left")
        #print(self.idx, 'idx')
        if self.idx[-1] - self.idx[-2] < 50:
            #print("asdfasdasdafda")
            try:
                self.idx[-1] = len(self.groove[0]) - 1
            except:
                pass
            print(self.idx, 'new idx')
        self.segments = [self.groove[:, self.idx[i]:self.idx[i+1] - 1] for i in range(len(self.idx)-1)]

    def segment_line_fit(self, index, iterations):

        if index == 1:
            move_limit = 0.01
        else:
            move_limit = 0.2
        if ((self.dist_moved[index] + self.dist_moved[index+1]) / 2) < move_limit:
            #print('skipping segment', index)
            return
        if abs(self.angle[index]) < 0.008 and self.dist_moved[index] < 1:
            #print(f'angle for index {index} is {self.angle[index]}, skipping segment')
            return

        x = self.segments[index][0]
        y = self.segments[index][1]

        point1 = np.append(self.corner_points[index], 1)
        point2 = np.append(self.corner_points[index+1], 1)

        line = np.cross(point1, point2)
        line[0] = -line[0] / line[1]
        line[2] = -line[2] / line[1]
        line[1] = -1.0
        slope_est, intercept_est = line[0], line[2]

        if index == 1:# and iterations == 0:
            #print('***************')
            segment1 = self.line_segments[0]
            segment2 = self.line_segments[2]

            new_point1 = np.cross(line, segment1)
            new_point1 = new_point1 / new_point1[-1]

            new_point2 = np.cross(line, segment2)
            new_point2 = new_point2 / new_point2[-1]

            idx = np.searchsorted(self.groove[0], np.array([new_point1[0], new_point2[0]]), side="left")
            new_segment = self.groove[:,idx[0]:idx[1]]

            x = new_segment[0]
            y = new_segment[1]

            line = np.cross(new_point1, new_point2)
            line[0] = -line[0] / line[1]
            line[2] = -line[2] / line[1]
            line[1] = -1.0
            slope_est, intercept_est = line[0], line[2]

            point1 = new_point1
            point2 = new_point2

        if len(x) < 3: # == 0 før
            #print('æææææææ')
            # if the angles are wrong during the first iteration causing the root corners to collapse towards
            # a single point, the x- and y-values are manually defined as to give a constant line at this point.
            x = np.array((self.corner_points[index:index+1, 0] - 1, self.corner_points[index:index+1, 0] + 1))
            y = np.array((self.corner_points[index:index+1, 1], self.corner_points[index:index+1, 1]))
            #print(len(x),'lenx')
            



        # noise removal
        y_est = [slope_est * i + intercept_est for i in x]

        y_diff = y - y_est
        #print(np.std(y_diff), 'sdt')
        if np.std(y_diff) > 8.0:
            indices_to_remove = np.where(abs(y[:-1] - y[1:]) > 0.3)[0]
            #if len(self.noise) > 0:
            #    self.noise = np.concatenate((self.noise, self.segments[index][:,indices_to_remove]))
            #else:
            #    self.noise = self.segments[index][:,indices_to_remove]
            x = np.delete(self.segments[index][0], indices_to_remove)
            y = np.delete(self.segments[index][1], indices_to_remove)
            self.segments[index] = np.vstack((x, y))

            y_est = [slope_est * i + intercept_est for i in x]

            y_diff = y - y_est

                  
        #print(index, 'i')
        if (index == 0 or index == 2) and iterations == 0:
            #non_noise = np.where(abs(y_diff - np.median(y_diff)) < (1))
            non_noise = np.where(abs(y_diff) < abs(np.median(y_diff)) + 1)
            x = x[non_noise]
            y = y[non_noise]
            to_keep = np.where(x > self.corner_points[index, 0] + 1)
            to_keep = to_keep[0][np.where(x[to_keep] < self.corner_points[index+1, 0] - 1)]

            x = x[to_keep]
            y = y[to_keep]

        elif index == 3 and iterations == 0:
            non_noise = np.where(abs(y_diff) < abs(np.median(y_diff)) + 1.5)
            #print(x)
            x = x[non_noise]
            y = y[non_noise]

        else:
            if index == 1:
                if iterations == 0:
                    noise_threshold = 3
                else:
                    noise_threshold = 0.7 - 0.1 * iterations
                limit = 0.75
                reduction = 0.10

            else:
                if index == 3:
                    noise_threshold = np.amax(np.array([0.1, 3 - 0.4 * iterations]))
                    limit = 0.2
                    reduction = 0.8

                else:
                    noise_threshold = 3 - 0.25 * iterations
                    limit = 0.2
                    reduction = 0.03

            non_noise = np.where(abs(y_diff) < noise_threshold)

        

            #print(len(non_noise[0]), 'num points used for linear regression', len(x))

            
            
            while len(non_noise[0]) < len(x) * (limit - (iterations * reduction)):
                #print(len(non_noise[0]),'len1')
                #print(len(x),'len2')
                noise_threshold += 0.05
                #print(noise_threshold)
                #non_noise = np.where(abs(y - y_est_new) < noise_threshold)
                non_noise = np.where(abs(y_diff) < noise_threshold)

            non_noise = non_noise[0]
            if index == 0:
                i = 3
                while len(non_noise) > 0.8 * len(x):
                    non_noise = non_noise[i:]
            elif index == 2:
                non_noise = non_noise[3:]

            x = x[non_noise]
            y = y[non_noise]
        

        try:
            slope, intercept = np.polyfit(x, y, 1)
        except:
            print("settting slope and intercept manually...")
            slope = 0
            #print(y)
            intercept = y[0]    

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

        self.angle[index] = np.arctan((slope - slope_est) / (1 + slope * slope_est))
        #print(angle, 'angle')
        if index == 1 and (iterations == 0 or iterations == 1):
            if iterations == 0:
                if self.angle[index] > 0.25:
                    self.angle[index] = self.angle[index] / 3
                elif self.angle[index] > 0.1:
                    self.angle[index] = self.angle[index] / 1.5
            if iterations == 1:
                if self.angle[index] > 0.5:
                    self.angle[index] = self.angle[index] / 2

        rot = np.array([[np.cos(self.angle[index]), -np.sin(self.angle[index])],
                        [np.sin(self.angle[index]), np.cos(self.angle[index])]])

        corners = rot @ corners
        
        corners[0] = corners[0] + mean_x
        corners[1] = corners[1] + mean_y

        corners_homogenous = np.vstack((corners, np.ones((1,2))))
        new_line = np.cross(corners_homogenous[:,0], corners_homogenous[:,1])
        new_line[0] = -new_line[0] / new_line[1]
        new_line[2] = -new_line[2] / new_line[1]
        new_line[1] = -1.0

        #new_line = np.append(new_line, index)
        
        if self.display_line_segments:
            plt.scatter(x, y, s = 10)
            plt.plot(x, line_values, c='r')
            plt.plot(xs, line_values_est)
            plt.scatter(old_corners[0], old_corners[1], c='g')
            plt.scatter(corners[0], corners[1], c='m')
            #plt.show()
    #
            plt.scatter(self.groove[0], self.groove[1],s=1)
            plt.scatter(self.corner_points[:,0], self.corner_points[:,1], s=20) # når output (B, 5, 2)
            plt.show()

        if index == len(self.segments) - 1:
            self.end_corner = corners[:,1]
        self.line_segments[index, :] = new_line


    def update(self, iterations):
        self.divide_into_segments()

        self.segment_line_fit(0, iterations)
        self.segment_line_fit(2, iterations)
        self.segment_line_fit(1, iterations)
        self.segment_line_fit(3, iterations)

        l1 = np.vstack((self.line_segments[0], self.line_segments[:-1]))
        l2 = np.vstack((self.line_segments[-1], self.line_segments[1:]))
        new_output = np.cross(l1, l2)
        #new_output = np.cross(line_segments[:-1], line_segments[1:])
        new_corners = new_output[:,:-1] / new_output[:,-1][:,None]
        new_corners = np.vstack((new_corners, self.end_corner))

        # find the average distance each point has been moved
 
        # the last segment tends to move needlessly in the x-direction, so zeroing this movement out
        self.corner_points[-2:,0] = new_corners[-2:,0]
        self.dist_moved = np.linalg.norm(self.corner_points - new_corners, axis=1) 

        self.corner_points = new_corners


        # if the intersection of segments 1, 2 and 3 lead to corner point 1 having a larger
        # x-value than corner points 2, then these are manually adjusted.
        while self.corner_points[1,0] > self.corner_points[2, 0] - 0.5:
            #print('changing...')
            #print(self.corner_points[1, 0])
            self.corner_points[1, 0] = self.corner_points[1, 0] - 0.25
            #print(self.corner_points[1, 0])
            self.corner_points[2, 0] = self.corner_points[2, 0] + 0.25

        segments = np.hstack(self.segments)
        self.groove = np.hstack((self.groove[:, :self.idx[0]], segments, self.groove[:, self.idx[-1]:]))

        return


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    net = model.SimpleNetwork23b()
    net = net.to(device)
    #net = resnet.ResidualNetwork()
    # C:\Users\oyvin\Desktop\dfsdf
    #state_dict = torch.load('C:/Users/oyvin/Desktop/dfsdf/ResNet_NewSet123.pth')
    state_dict = torch.load('/home/oyvind/Blender-weldgroove/ResNet_NewSet123.pth')
    net.load_state_dict(state_dict['model_state_dict'])



    grooves = os.listdir('/home/oyvind/Downloads/grooves')
    #grooves = os.listdir('C:/Users/oyvin/Downloads/grooves')
    #print(grooves)
    #print(grooves[1:2])

    for groove in grooves[:]:
        groove_start_time = time.time()

        #path = 'C:/Users/oyvin/Downloads/grooves/' + groove
        path = '/home/oyvind/Downloads/grooves/' + groove
        #print(path)
        hei = np.load(path)
        chei = hei
        hei = np.flip(hei, axis=1)


        hei = hei/1000
        hei = torch.from_numpy(np.array(hei))
        hei = hei.type(torch.float32)
        #t = transforms.Normalize((0.0017, 0.2328), (0.0310, 0.0198))
        #t = transforms.Normalize((0.0018, 0.2383), (0.0311, 0.0226)) # with noise
        t = transforms.Normalize((0.0018, 0.2333), (0.0310, 0.0205))
        
        hei = hei.unsqueeze(0)
        hei = hei.permute(1, 0, 2)
        hei = t(hei)
        hei = hei.squeeze()


        hei = hei.unsqueeze(0)
        hei = hei.to(device)

        net.eval()
        x = torch.rand(1, 2, 640)
        x = x.to(device)
        with torch.no_grad():
            for _ in range(5):
                _ = net(x)
            inference_start = time.time()
            ut = net(hei)
            inference_time = time.time() - inference_start
           # print(f'inference time: {inference_time}')

        hmm = (ut.detach().cpu().numpy() * 1000).squeeze()

        #np.save('corner_result_' + groove, hmm)
        

        cheiflip = np.flip(chei, axis=1)

        # definer objekt
        line = LineSegment(groove, cheiflip, hmm, 0, display_line_segments=False)
        print(f'\n groove iterative correction time = {time.time() - inference_start} \n')

    
        


    