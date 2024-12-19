import matplotlib.pyplot as plt
import cv2 as cv2  # pip install opencv-python
import numpy as np

def show(image, dpi):
    plt.figure(dpi=dpi)
    plt.imshow(image, cmap='gray')
    plt.axis('off')  # Skrýt osy pro čistší vzhled
    plt.show()

#######
####### kod mozno nefunguje bol skopirovany ale testuje sa tu clahe
#######
# vyhladzovanie histogramu pomocou clahe

img_clahe = cv2.imread('mmm_cierny.png',0)

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img_clahe)

cv2.imwrite('mmm_clahe.png',cl1)
show(cl1,200, "clahe")

# vyhladzvoanie historgramu globalne

img_ge = cv2.imread('mmm_cierny.png',0)

hist,bins = np.histogram(img_ge.flatten(),256,[0,256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

plt.plot(cdf_normalized, color = 'b')
plt.hist(img_ge.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.title("histogram globalny equalization")
plt.show()

cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')

img2 = cdf[img_ge]

img_ge = cv2.imread('mmm_cierny.png',0)
equ = cv2.equalizeHist(img_ge)
res = np.hstack((img_ge,equ)) #stacking images side-by-side
cv2.imwrite('mmm_global_equalization.png',equ)
cv2.imwrite('mmm_cierny_a_global_equalization.png',res)

show(res, 200, "povodny cierny obrazok a global eq")
show(equ,200, "global eq")

res_globaleq_vs_clahe = np.hstack((equ, img_clahe))
cv2.imwrite("mmm_ge_vs_clahe.png", res_globaleq_vs_clahe)
show(res_globaleq_vs_clahe,300, "Global equalization vs Clahe")