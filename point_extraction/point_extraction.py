import cv2 
import numpy as np
import matplotlib.pyplot as plt
import time
import exr_util


def gray_gravity(img_path):

    im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    #plt.imshow(im)
    #plt.show()
    #print(im.shape)

    start = time.time()
    top_indices = np.argpartition(im, -5, axis=0)[-5:]
    top_values = np.partition(im, -5, axis=0)[-5:]

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
    print(f'execution time:', time.time() - start)

    return center

def extract_position(points, exr_file, mask):
    exr_dict = exr_util.load_exr_to_dict(exr_file)
    position_img = np.dstack((exr_dict["Image"]["X"], exr_dict["Image"]["Y"], exr_dict["Image"]["Z"]))
    print((position_img.shape))
    #plt.imshow(position_img)
    
    #plt.show()

    # center point of mask laser line
    gt_mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)

    gt_center_line = np.argmax(gt_mask, axis=0)
    
    # indices of laser line
    minimum_value = np.amax(gt_center_line) - 1300
    laser_line_indices = np.where(gt_center_line > minimum_value)
    # use only 640 evenly spaced points
    idx = np.round(np.linspace(0, len(laser_line_indices[0]) - 1, 640)).astype(int)
    laser_line_indices = laser_line_indices[0][idx]
    gt_center_line = gt_center_line[laser_line_indices]
    #print(f'points shap {points.shape}')
    points = points[laser_line_indices]
    #print(points.shape)



    # extracts the x,y and z position of the laser center points
    position_gt = position_img[gt_center_line, laser_line_indices]
    position_est = position_img[points, laser_line_indices]
    #print(position_est.shape)

    plt.scatter(np.arange(0, len(gt_center_line), 1),gt_center_line, s = 1)
    plt.scatter(np.arange(0, len(points), 1),points, s = 1)
    plt.show()

    return position_gt.T, position_est.T

if __name__ == "__main__":
    points = gray_gravity("grayscale_laser.png")
    ground_truth, estimate = extract_position(points, "render/0001.exr", "render/mask0001.png")

    # loads the transformation matrix from world origin to laser
    tmatrix = np.load("/home/oyvind/ip/render/1.npy")
    #print(tmatrix)
    
    # inverts the transformation matrix
    rotation_matrix = tmatrix[0:3,0:3]
    translation = tmatrix[0:3,-1:]

    inverse_rotation = rotation_matrix.transpose()
    inverse_translation = -inverse_rotation @ translation

    inverse_transform = np.vstack((np.hstack((inverse_rotation, inverse_translation)), [0,0,0,1]))
    ground_truth = np.vstack((ground_truth, np.ones((1,len(ground_truth[0])))))
    estimate = np.vstack((estimate, np.ones((1, len(estimate[0])))))
    # finds position of the points in the reference frame of the laser scanner
    ground_truth = inverse_transform @ ground_truth
    estimate = inverse_transform @ estimate

    print(ground_truth[0])
    print("hold")
    #print(inverse_rotation, '\n')
    #print(inverse_transform)
