import cv2
import numpy as np
import matplotlib.pyplot as plt

# Načítanie obrázkov
image1 = cv2.imread('image1.png')
image2 = cv2.imread('image2.png')

# Konverzia na grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Detekcia príznakov pomocou SIFT
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Porovnávanie príznakov (Brute-Force Matcher)
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Výber najlepších zhôd (Loweho ratio test)
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Získanie bodov zo zhodných príznakov
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Výpočet homogénnej matice
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Transformácia druhého obrázka
height, width, _ = image1.shape
result = cv2.warpPerspective(image2, H, (width + image2.shape[1], height))
result[0:height, 0:width] = image1

# Zobrazenie výsledkov
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Panoráma')
plt.axis('off')
plt.show()
