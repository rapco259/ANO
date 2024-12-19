import cv2
import matplotlib.pyplot as plt

# Načítanie obrázka
image = cv2.imread("RD2.jpg")  # Nahraďte názvom vášho súboru
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Prevod z BGR na RGB pre správne zobrazenie

# Prevod RGB obrázka do HSV
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Rozdelenie HSV na jednotlivé kanály
hue, saturation, value = cv2.split(image_hsv)

# Zobrazenie originálneho RGB obrázka a jeho HSV kanálov
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Originálny obrázok
axes[0, 0].imshow(image_rgb)
axes[0, 0].set_title("Original RGB Image")
axes[0, 0].axis("off")

# Hue (odtieň)
axes[0, 1].imshow(hue, cmap="hsv")
axes[0, 1].set_title("Hue Channel")
axes[0, 1].axis("off")

# Saturation (sýtosť)
axes[1, 0].imshow(saturation, cmap="gray")
axes[1, 0].set_title("Saturation Channel")
axes[1, 0].axis("off")

# Value (jas)
axes[1, 1].imshow(value, cmap="gray")
axes[1, 1].set_title("Value Channel")
axes[1, 1].axis("off")

plt.tight_layout()
plt.show()
