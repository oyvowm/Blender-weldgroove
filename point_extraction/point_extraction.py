import cv2 
import numpy as np
import matplotlib.pyplot as plt
import time
import exr_util
import os
from tqdm import tqdm

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
    laser_line_indices = laser_line_indices[0][idx]
    gt_center_line = gt_center_line[laser_line_indices]
    #print(f'points shap {points.shape}')
    points = points[laser_line_indices]
    #print(f'point: {points.shape}')



    # extracts the x,y and z position of the laser center points
    position_gt = position_img[gt_center_line, laser_line_indices]
    position_est = position_img[points, laser_line_indices]
    #print(position_est.shape)

    
    #plt.scatter(np.arange(0, len(gt_center_line), 1),gt_center_line, s = 1)
    #plt.scatter(np.arange(0, len(points), 1),points, s = 1)
    #plt.show()

    return position_gt.T, position_est.T

if __name__ == "__main__":

    settings = {
        'process_all': True
    }

    render_path = "/home/oyvind/Blender-weldgroove/render/"
    os.chdir(render_path)
    renders = os.listdir(render_path)
    renders = [render for render in renders if (render[-3:] != "npy" and render[-3:] != "exr")]
    renders = [int(i) for i in renders]
    renders.sort()
    renders.pop()
    renders = renders[:5]
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
            ground_truth = np.nan_to_num(ground_truth)
            estimate = np.nan_to_num(estimate)

            # loads the transformation matrix from world origin to laser
            tmatrix = np.load(images_path + '/' + img_num + '.npy')

            # inverts the transformation matrix
            rotation_matrix = tmatrix[0:3,0:3]
            # since scale transform cannot be applied in Blender for light objects, it's applied here
            rotation_matrix = rotation_matrix * (1 / 0.8)
         
            translation = tmatrix[0:3,-1:]
            # the laser scanners position along the x-axis is not of interest (except for later checking where along the weld groove each scan was made).
            # zeroing out these values makes the x-value of the estimate and GT give the position of the actual laser line.
            translation[0] = 0 

            inverse_rotation = rotation_matrix.transpose()
            inverse_translation = -inverse_rotation @ translation

            inverse_transform = np.vstack((np.hstack((inverse_rotation, inverse_translation)), [0,0,0,1]))
            ground_truth = np.vstack((ground_truth, np.ones((1,len(ground_truth[0])))))
            estimate = np.vstack((estimate, np.ones((1, len(estimate[0])))))

            # finds position of the points in the reference frame of the laser scanner
            #print(f'inverse transform: {inverse_transform}')
            ground_truth = inverse_transform @ ground_truth
            estimate = inverse_transform @ estimate

            np.save(images_path + "/processed_images/points_" + img_num + '/' + img_num + '_GT', ground_truth)
            np.save(images_path + "/processed_images/points_" + img_num + '/' + img_num + '_EST', estimate)


#array([[-0.18000001],
#       [ 0.21943058],
#       [ 0.04645395]])

#array([[ 0.80000001,  0.        ,  0.        ],
#       [ 0.        ,  0.68764287,  0.72604913],
#       [ 0.        , -0.72604913,  0.68764287]])