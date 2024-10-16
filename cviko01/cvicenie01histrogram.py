import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

img = cv.imread('input/janzizka.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print(img.shape)

hist = cv.calcHist(img,[0],
                  None,
                  [256],
                  [0, 256])
print(hist)
plt.plot(hist)
plt.show()
# viridis color scheme
plt.imshow(img, cmap='gray')
plt.show()
