import numpy as np
import cv2 as cv

img = cv.imread("ANO_detekcia_hran_resized_obrazovka.jpg")
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def nothing(x):
    pass

cv.namedWindow('image')
cv.createTrackbar('maxThreshold', 'image', 0, 255, nothing)
cv.createTrackbar('minThreshold', 'image', 0, 255, nothing)
edge = img

while 1:
    cv.imshow('image', edge)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break

    minT = cv.getTrackbarPos('minThreshold', 'image')
    maxT = cv.getTrackbarPos('maxThreshold', 'image')
    edge = cv.Canny(gray_img, minT, maxT)
    # cv.imwrite("canny.jpg", edge)

cv.destroyAllWindows()

