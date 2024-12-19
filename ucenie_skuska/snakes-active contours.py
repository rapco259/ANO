import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour

# Načítanie testovacieho obrázka
image = data.astronaut()  # Obrázok z knižnice skimage
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Prevod na grayscale
smoothed_image = gaussian(gray_image, sigma=3)  # Rozmazanie pre lepší výpočet gradientov

# Inicializácia kontúry (kruh)
s = np.linspace(0, 2 * np.pi, 100)
x = 220 + 100 * np.cos(s)
y = 120 + 100 * np.sin(s)
init = np.array([x, y]).T

# Aktívna kontúra
snake = active_contour(smoothed_image, init, alpha=0.1, beta=1, gamma=0.01)

# Vizualizácia
plt.figure(figsize=(8, 8))
plt.imshow(gray_image, cmap=plt.cm.gray)
plt.plot(init[:, 0], init[:, 1], '--r', label='Počiatočná kontúra')
plt.plot(snake[:, 0], snake[:, 1], '-b', label='Aktívna kontúra')
plt.legend()
plt.title('Aktívne kontúry')
plt.show()
