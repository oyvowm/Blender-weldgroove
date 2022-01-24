import cv2 
import numpy as np
import matplotlib.pyplot as plt
import time
import exr_util
import os
from tqdm import tqdm

def rot_matrix(axis, angle):
    assert axis in ['x', 'y', 'z'], "The axis parameters needs to be one of 'x', 'y' and 'z'"
    angle = np.radians(angle)
    if axis == 'x':
        return np.array([[1, 0, 0],
                        [0, np.cos(angle), -np.sin(angle)],
                        [0, np.sin(angle), np.cos(angle)]])
    elif axis == 'y':
        return np.array([[np.cos(angle), 0, np.sin(angle)],
                         [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
    elif axis == 'z':
        return np.array([[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0],
                        [0, 0, 1]])

def gray_gravity(img_path):

    #print(f'image path: {img_path}')

    im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #im_rgb = im
    #plt.imshow(cv2.cvtColor(im_rgb, cv2.COLOR_BGR2RGB))
    #plt.show()conda
    #print(im.shape)

    start = time.time()
    top_indices = np.argpartition(im, -3, axis=0)[-3:]
    top_values = np.partition(im, -3, axis=0)[-3:]

    h = top_values * top_indices

    numerator = np.sum(h, axis=0)
    denominator = np.sum(top_values, axis=0)
    center = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0, casting='unsafe')
    xs = np.arange(0,im.shape[1], 1)
    #points = np.stack((xs, center), axis=0)

    #plt.scatter(xs, center, s=1, color='g')
    #plt.xlim(0,im.shape[1])
    #plt.ylim(im.shape[0], 0)
    #plt.show()
    
    #print(f'execution time:', time.time() - start)

    return center

def extract_position(points, exr_file, mask):
    exr_dict = exr_util.load_exr_to_dict(exr_file)
    position_img = np.dstack((exr_dict["Image"]["X"], exr_dict["Image"]["Y"], exr_dict["Image"]["Z"]))
    #plt.imshow(position_img)
    #plt.show()

    # center point of mask laser line
    gt_mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    gt_center_line = np.argmax(gt_mask, axis=0)
    
    #plt.scatter(np.arange(0,len(gt_center_line), 1), gt_center_line, s = 1)
    #plt.show()
    # indices of laser line
    #print(f'max center line value: {np.amax(gt_center_line)}')
    # try to extract only those points actually on the laser line
    minimum_value = np.amax(gt_center_line) * 0.1
    laser_line_indices = np.where(gt_center_line > minimum_value)

    # use only 640 evenly spaced points
    idx = np.round(np.linspace(0, len(laser_line_indices[0]) - 1, 640)).astype(int)
    
    gt_center_line = gt_center_line[laser_line_indices]
    laser_line_indices_shortened = laser_line_indices[0][idx]
    points = points[laser_line_indices_shortened]
    #print(f'point: {points.shape}')


    #plt.imshow(position_img)
    #plt.show()
    # extracts the x,y and z position of the laser center points

    position_gt = position_img[gt_center_line, laser_line_indices[0]]
    position_est = position_img[points, laser_line_indices_shortened]
    #print(position_est.shape)

    
    #plt.scatter(np.arange(0, len(gt_center_line), 1),gt_center_line, s = 1)
    #plt.scatter(np.arange(0, len(points), 1),points, s = 1, color='g')
    #plt.scatter(laser_line_indices,points, s = 1, color='g')
    #plt.show()

    return position_gt.T, position_est.T

if __name__ == "__main__":

    settings = {
        'process_all': False
    }

    render_path = "/home/oyvind/Blender-weldgroove/render/"
    os.chdir(render_path)
    renders = os.listdir(render_path)
    renders = [render for render in renders if (render[-3:] != "npy" and render[-3:] != "exr")]
    renders = [int(i) for i in renders]
    renders.sort()
    renders.pop()
    #renders = renders[52:] # to only process a certain subset
    renders = [str(i) for i in renders]  
    print(renders)

    
    for render in tqdm(renders):
        images_path = render_path + render
        images = os.listdir(images_path + "/processed_images")
        images = [image for image in images if image[-3:] == "png"]
        matrices = os.listdir(images_path)
        matrices = [i for i in matrices if i[-3:] == "npy"]

        os.chdir(images_path)
        for image in images:
            img_num = image[:4]
            if os.path.exists(images_path + '/processed_images' + '/points_' + img_num):
                if not settings["process_all"]:
                    continue
            else:
                os.mkdir('processed_images/points_' + img_num)

            points = gray_gravity("processed_images/" + image)
            ground_truth, estimate = extract_position(points, img_num + ".exr", "mask" + img_num + ".png")

            # replace NaN values with zero
            ground_truth = np.nan_to_num(ground_truth, posinf=0.9, neginf=-0.9)
            estimate = np.nan_to_num(estimate, posinf=0.9, neginf=-0.9)

            # loads the transformation matrix from world origin to laser
            tmatrix = np.load(images_path + '/' + img_num + '.npy')

            # inverts the transformation matrix
            rotation_matrix = tmatrix[0:3,0:3]

            # since scale transform cannot be applied in Blender for light objects, it's applied here
            
            rotation_matrix[0][0] = rotation_matrix[0][0] * (1 / 0.8)
         
            translation = tmatrix[0:3,-1:]
            # For some reason the transformation matrix of the scanner is not updated when using render keyframes, it is therefore stuck with a translation of -0.18m in the world coordinate frame
            # therefore, for the first 50 renders, the x-value of the laser scanner is replaced by the average x-value of the ground truth points.
            if int(render) < 51:
                translation[0] = np.average(ground_truth[0])
                
            # for render >= 57 the tmatrix should contain the actual x-value.

            inverse_rotation = rotation_matrix.transpose()

            inverse_translation = (-inverse_rotation) @ translation
            inverse_transform = np.vstack((np.hstack((inverse_rotation, inverse_translation)), [0,0,0,1]))

            ground_truth = np.vstack((ground_truth, np.ones((1,len(ground_truth[0])))))
            estimate = np.vstack((estimate, np.ones((1, len(estimate[0])))))

            # finds position of the points in the reference frame of the laser scanner
            #print(f'inverse transform: {inverse_transform}')
            ground_truth = inverse_transform @ ground_truth
            estimate = inverse_transform @ estimate

            # rotate 180 degrees about the x-axis to make the positive z-direction towards the groove.
            rot_x = rot_matrix('x', 180)

            ground_truth = ground_truth[:3]
            ground_truth = rot_x @ ground_truth 
            estimate = estimate[:3]
            estimate = rot_x @ estimate
            
            np.save(images_path + "/processed_images/points_" + img_num + '/' + img_num + '_GT', ground_truth)
            np.save(images_path + "/processed_images/points_" + img_num + '/' + img_num + '_EST', estimate)


#array([[-0.18000001],
#       [ 0.21943058],
#       [ 0.04645395]])

#array([[ 0.80000001,  0.        ,  0.        ],
#       [ 0.        ,  0.68764287,  0.72604913],
#       [ 0.        , -0.72604913,  0.68764287]])