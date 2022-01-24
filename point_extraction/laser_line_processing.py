import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm

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

for render in tqdm(renders, position=0, leave=True):

    os.chdir(render_path + render)
    path = os.getcwd()
    if os.path.exists(path + "/" + "processed_images"):
        if not settings["process_all"]:
            continue
    else:
        os.mkdir("processed_images")
    img_list = os.listdir(os.getcwd())
    os.chdir(os.getcwd() + "/processed_images")

    denoised_images = [x for x in img_list if (len(x) > 13 and x != "processed_images")]

    #print(denoised_images)

    for img in denoised_images:
        img_num = img[14:-4]
        img = cv.imread(path + "/" + img)

        # convert to HSV
        hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        hsv_img = cv.medianBlur(hsv_img, 5)

        lower1 = np.array([0, 90, 90])
        upper1 = np.array([10, 255, 255])
        mask1 = cv.inRange(hsv_img, lower1, upper1)

        lower2 = np.array([170,90,90])
        upper2 = np.array([180,255,255])
        mask2 = cv.inRange(hsv_img, lower2, upper2)

        red_line = mask1 + mask2

        imgny = cv.bitwise_and(img,img, mask = red_line)


        kernel_closing = np.ones((5,5), np.uint8)
        kernel_opening = np.ones((1,1), np.uint8)
        #imgny = cv.erode(imgny, kernel, iterations=1)
        imgny = cv.morphologyEx(imgny, cv.MORPH_CLOSE, kernel_closing)
        imgny = cv.morphologyEx(imgny, cv.MORPH_OPEN, kernel_opening)
        #imgny = cv.GaussianBlur(imgny, (5,5), 0)


        gray = cv.cvtColor(imgny, cv.COLOR_BGR2GRAY)
        #gray = cv.medianBlur(gray,5)
        #cv.imshow('groove', gray)
        #k = cv.waitKey(0)
        
        cv.imwrite(img_num + ".png", gray)
