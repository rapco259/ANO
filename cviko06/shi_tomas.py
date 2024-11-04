import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('ANO_detekcia_hran_resized_obrazovka.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int8(corners)
print(corners.shape)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),10,255, -1)

img_plt = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
cv2.imshow('img',img)
cv2.waitKey()
plt.imshow(img_plt),plt.show()
## it dont work