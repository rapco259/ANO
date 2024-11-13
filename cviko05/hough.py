import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("ANO_detekcia_hran_resized_obrazovka.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# rgb - gray - canny - binnary img - houghLinesP - img RGB lines

edges = cv2.Canny(gray_img,10,50, apertureSize = 3)

# plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
#
# plt.show()

minLineLength = 50
maxLineGap = 5

lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
#for line in lines:
for x1,y1,x2,y2 in lines.squeeze():
    print(x1)
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imwrite('houghlines5.jpg',img)
