import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# pip install opencv-python

img = cv.imread('input/janzizka.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
print(img.shape)  # height, width, rgb
print(img[50, 50, 0])

#cv.imshow('image', img)
#cv.waitKey(0)
#cv.destroyAllWindows()

cv.circle(img, (440, 310), radius=100,
          color=(0, 0, 255), thickness=5)

part = img[200:400, 300:400]
img[100:300, 200:300] = part

red = img[:, :, 0]
img[:,:,0] = cv.multiply(red, 2)

plt.imshow(img)
plt.show()
