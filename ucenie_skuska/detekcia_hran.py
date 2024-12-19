import cv2
import numpy as np
import matplotlib.pyplot as plt

# Načítanie obrázka v grayscale
image = cv2.imread('RD2.jpg', cv2.IMREAD_GRAYSCALE)

# Sobelov filter (gradient)
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Horizontálny gradient
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Vertikálny gradient
sobel = cv2.magnitude(sobel_x, sobel_y)  # Kombinácia gradientov

# Laplacian (druhá derivácia)
laplacian = cv2.Laplacian(image, cv2.CV_64F)

# Cannyho detektor
edges = cv2.Canny(image, 50, 150)

# Zobrazenie výsledkov
titles = ['Original', 'Sobel', 'Laplacian', 'Canny']
images = [image, sobel, laplacian, edges]

plt.figure(figsize=(12, 8))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
