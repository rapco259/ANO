import cv2
import numpy as np
import matplotlib.pyplot as plt

def gamma_correction(image, gamma):
    # Normalizácia pixelov na rozsah [0, 1] a aplikácia gamma korekcie
    corrected = np.power(image / 255.0, gamma)
    return np.uint8(corrected * 255)  # Prevod späť do rozsahu [0, 255]

# Načítanie obrázka v grayscale
image = cv2.imread("RD2.jpg", cv2.IMREAD_GRAYSCALE)

# Aplikácia gamma korekcie pre rôzne hodnoty
gamma_values = [2, 1, 0.5, 0.33, 0.25]  # Gamma = 2, 1, 1/2, 1/3, 1/4
corrected_images = [gamma_correction(image, g) for g in gamma_values]

# Vizualizácia výsledkov
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

axes[0].imshow(image, cmap="gray")
axes[0].set_title("Original Image (γ=1)")

for i, (img, gamma) in enumerate(zip(corrected_images, gamma_values)):
    axes[i+1].imshow(img, cmap="gray")
    axes[i+1].set_title(f"Gamma = {gamma}")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
