import cv2
import matplotlib.pyplot as plt

# Načítanie obrázka
image = cv2.imread('RD2.jpg', cv2.IMREAD_GRAYSCALE)

# Inicializácia SIFT detektora
sift = cv2.SIFT_create()

# Detekcia významných bodov a výpočet deskriptorov
keypoints, descriptors = sift.detectAndCompute(image, None)

# Vykreslenie významných bodov
output = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Zobrazenie výsledkov
plt.imshow(output, cmap='gray')
plt.title('SIFT Detekcia Významných Bodov')
plt.axis('off')
plt.show()
