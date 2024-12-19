import cv2
import numpy as np
import matplotlib.pyplot as plt

# Načítanie obrázka
image = cv2.imread('RD2.jpg')
mask = np.zeros(image.shape[:2], np.uint8)

# Vytvorenie modelov pre GrabCut
bg_model = np.zeros((1, 65), np.float64)
fg_model = np.zeros((1, 65), np.float64)

# Počiatočný obdĺžnik okolo objektu
rect = (50, 50, image.shape[1] - 100, image.shape[0] - 100)

# Použitie GrabCut
cv2.grabCut(image, mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)

# Zmena masky na binárnu (foreground = 1, background = 0)
mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
result = image * mask[:, :, np.newaxis]

# Zobrazenie výsledkov
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Pôvodný obrázok')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Segmentovaný objekt (GrabCut)')
plt.axis('off')

plt.tight_layout()
plt.show()
