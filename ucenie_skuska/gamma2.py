import cv2
import numpy as np
import matplotlib.pyplot as plt

def gamma_correction(image, gamma):
    corrected = np.power(image / 255.0, gamma)
    return (corrected * 255).astype(np.uint8)

# Načítanie obrázka
image = cv2.imread('RD2.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Aplikácia gamma korekcie
gamma_low = gamma_correction(image, 2.0)   # Tmavý obraz (gamma > 1)
gamma_high = gamma_correction(image, 0.5)  # Svetlý obraz (gamma < 1)
gamma_normal = gamma_correction(image, 1.0)  # Bez zmeny

# Zobrazenie výsledkov
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(gamma_low)
axes[0].set_title("a) Gamma = 2.0 (Tmavý obraz)")

axes[1].imshow(gamma_high)
axes[1].set_title("b) Gamma = 0.5 (Svetlý obraz)")

axes[2].imshow(gamma_normal)
axes[2].set_title("c) Gamma = 1.0 (Pôvodný obraz)")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
