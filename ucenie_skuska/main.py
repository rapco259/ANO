import cv2
import numpy as np
import matplotlib.pyplot as plt

# Načítanie obrázka
image = cv2.imread('RD2.jpg')  # Nahraďte cestou k vášmu súboru
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Prevod z BGR na RGB (pre vizualizáciu)

# Darken: Odčítanie hodnoty
darken = np.clip(image - 128, 0, 255).astype(np.uint8)

# Lighten: Pripočítanie hodnoty
lighten = np.clip(image + 128, 0, 255).astype(np.uint8)

# Invert: Inverzia farieb
invert = 255 - image

# Zobrazenie obrázkov
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].imshow(image)
axs[0, 0].set_title("Original")

axs[0, 1].imshow(darken)
axs[0, 1].set_title("Darken (f - 128)")

axs[1, 0].imshow(lighten)
axs[1, 0].set_title("Lighten (f + 128)")

axs[1, 1].imshow(invert)
axs[1, 1].set_title("Invert (255 - f)")

for ax in axs.flat:
    ax.axis("off")

plt.tight_layout()
plt.show()
