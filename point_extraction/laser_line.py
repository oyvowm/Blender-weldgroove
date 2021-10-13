import cv2 as cv
import numpy as np

img = cv.imread("hei.png")
img2 = cv.imread("test2.png", 0)



rows, cols = img2.shape
# convert to HSV
hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

hsv_img = cv.medianBlur(hsv_img, 5)

lowerb = np.array([0, 105, 105])
upperb = np.array([10, 255, 255])
red_line = cv.inRange(hsv_img, lowerb, upperb)

imgny = cv.bitwise_and(img,img, mask = red_line)


kernel_closing = np.ones((10,10), np.uint8)
kernel_opening = np.ones((1,1), np.uint8)
#imgny = cv.erode(imgny, kernel, iterations=1)
imgny = cv.morphologyEx(imgny, cv.MORPH_CLOSE, kernel_closing)
imgny = cv.morphologyEx(imgny, cv.MORPH_OPEN, kernel_opening)
#imgny = cv.GaussianBlur(imgny, (5,5), 0)

'''
#fourier transform test

fftimg = cv.dft(np.float32(img2), flags = cv.DFT_COMPLEX_OUTPUT)
fftimgshift = np.fft.fftshift(fftimg)

magnitude_spectrum = 20*np.log(cv.magnitude(fftimgshift[:,:,0], fftimgshift[:,:,1]))

mask = np.zeros((rows, cols, 2), np.uint8)

mask[int(rows/2) - 60:int(rows/2) + 60, int(cols/2)-60:int(cols/2) + 60] = 1

#fshift = fftimgshift * mask
f_ishift = np.fft.ifftshift(fftimgshift)
img_back = cv.idft(f_ishift)
img_back = cv.magnitude(img_back[:,:,0], img_back[:,:,1])
'''

gray = cv.cvtColor(imgny, cv.COLOR_BGR2GRAY)
#gray = cv.medianBlur(gray,5)
cv.imshow('groove', gray)
k = cv.waitKey(0)

cv.imwrite("grayscale_laser.png", gray)