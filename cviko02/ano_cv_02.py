# -*- coding: utf-8 -*-
"ANO_CV_02.ipynb"

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2 #pip install opencv-python

img = cv2.imread("/content/drive/MyDrive/Colab Notebooks/input/doggo.jpg")
print(img.shape)

img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #opencv zobrazuje zložky v poradi BGR - divne farby su preto
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

cv2.imwrite("/content/drive/MyDrive/Colab Notebooks/input/doggo_gray.jpg", img) #uloenie siveho obrazka

images = [img1, img]
titles = ['RGB', 'Gray']

for i in range(2):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()

"""###Tresholding"""

ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

titles = ['RGB','Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']

images = [img1,img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(7):
    plt.subplot(2,4,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()

ret,th1 = cv2.threshold(img,60,255,cv2.THRESH_BINARY_INV)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

titles = ['RGB','Original Image', 'Global Thresholding (v = 60)',
            'Adaptive Mean Thresho', 'Adapt Gaussian Thresholding']
images = [img1,img, th1, th2, th3]

for i in range(5):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

"""### Morfológia"""

th_o = th3.copy()
kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(th3,kernel,iterations = 1)
dilation = cv2.dilate(th3,kernel,iterations = 1)


images1=[img, th_o, erosion, dilation]
titles1 = ["original", "tresholding", "erosion", "dilation"]
for i in range(4):
    plt.subplot(2,4,i+1),plt.imshow(images1[i],'gray')
    plt.title(titles1[i])
    plt.xticks([]),plt.yticks([])
plt.show()

"""- erózia biele odstráni, čierne rozšíri
- dilatácia čierne odstráni biele rozšíri


"""

th_o = th1.copy()
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(th1,kernel,iterations = 1)
dilation = cv2.dilate(th1,kernel,iterations = 1)


images1=[img, th_o, erosion, dilation]
titles1 = ["original", "tresholding", "erosion", "dilation"]
for i in range(4):
    plt.figure(dpi=150)
    plt.subplot(2,4,i+1),plt.imshow(images1[i],'gray')
    plt.title(titles1[i])
    plt.xticks([]),plt.yticks([])
plt.show()

th_o = th1.copy()
kernel_elipsa=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
kernel = np.ones((8,8),np.uint8)
kernel_elipsa_big=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
erosion = cv2.erode(th1,kernel_elipsa,iterations = 3)
dilation = cv2.dilate(erosion,kernel_elipsa_big,iterations = 1)


images1=[img, th_o, erosion, dilation]
titles1 = ["original", "tresholding", "erosion", "erosion + dilation"]
for i in range(4):
    plt.figure(dpi=150)
    plt.subplot(2,4,i+1),plt.imshow(images1[i],'gray')
    plt.title(titles1[i])
    plt.xticks([]),plt.yticks([])
plt.show()

"""- erozia 15x15 filtrom zachrani len tie veci ktore mali aspon 15 pixelov a zuzi ich na

- ak filter bude krizik erozia necha stredy krizikov a dilatacia s tym istym filtrom potom z toho spravi kriziky

### Granulomertria
- analyza porovytich materialov
- chceme pocitat kolko guliciek roznych rozmerov tam je --> budeme aplikovat otvorenie s nejakou velkostou a porovnavat ako sa meni obrazok
- ako ked sme chceli odstranit hlavu - ak sme pouzili filter s dostatocne velkou elipsou - vedeli sme zistit velkost hlavy
-

#### Distance transform
- vzdialenost od okraja objektu --> bude neskor este

### Top Hat
- na sedtonovy aplikujeme otvorenie (vyhladime pozadie) aplikujeme treshold --> aplikujeme odcitanie veci

"""