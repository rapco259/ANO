import cv2
import numpy as np
import matplotlib.pyplot as plt

# Funkcia na pridanie Gaussian (White) šumu
def add_gaussian_noise(image, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), noise)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

# Funkcia na pridanie Impulse noise (Salt & Pepper)
def add_salt_and_pepper_noise(image, amount=0.02):
    noisy_image = image.copy()
    num_salt = int(amount * image.size * 0.5)
    num_pepper = int(amount * image.size * 0.5)

    # Pridanie soli (biele body)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255

    # Pridanie korenia (čierne body)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0
    return noisy_image

# Pridanie kvantizačného šumu
def add_quantization_noise(image, levels=4):
    quantized_image = (image // (256 // levels)) * (256 // levels)
    return quantized_image

# Načítanie obrázka
image = cv2.imread('RD2.jpg', cv2.IMREAD_GRAYSCALE)

# Pridanie šumu
gaussian_noisy = add_gaussian_noise(image)
salt_and_pepper_noisy = add_salt_and_pepper_noise(image)
quantization_noisy = add_quantization_noise(image)

# Filtrácia šumu
gaussian_filtered = cv2.GaussianBlur(gaussian_noisy, (5, 5), 0)
bilateral_filtered = cv2.bilateralFilter(gaussian_noisy, 9, 75, 75)
median_filtered = cv2.medianBlur(salt_and_pepper_noisy, 5)
quantization_interpolated = cv2.GaussianBlur(quantization_noisy, (3, 3), 0)

# Zobrazenie výsledkov
fig, axes = plt.subplots(4, 3, figsize=(15, 12))
axes = axes.ravel()

# Originálny obraz
axes[0].imshow(image, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis("off")

# Gaussian Noise a filtrovanie
axes[1].imshow(gaussian_noisy, cmap='gray')
axes[1].set_title("Gaussian Noise")
axes[1].axis("off")

axes[2].imshow(gaussian_filtered, cmap='gray')
axes[2].set_title("Gaussian Filter")
axes[2].axis("off")

axes[3].imshow(bilateral_filtered, cmap='gray')
axes[3].set_title("Bilateral Filter")
axes[3].axis("off")

# Salt & Pepper Noise a filtrovanie
axes[4].imshow(salt_and_pepper_noisy, cmap='gray')
axes[4].set_title("Salt & Pepper Noise")
axes[4].axis("off")

axes[5].imshow(median_filtered, cmap='gray')
axes[5].set_title("Median Filter (Salt & Pepper)")
axes[5].axis("off")

# Kvantizačný šum a interpolácia
axes[6].imshow(quantization_noisy, cmap='gray')
axes[6].set_title("Quantization Noise")
axes[6].axis("off")

axes[7].imshow(quantization_interpolated, cmap='gray')
axes[7].set_title("Gaussian Interpolation")
axes[7].axis("off")

# Ostatné prázdne okná
for i in range(8, 12):
    axes[i].axis("off")

plt.tight_layout()
plt.show()
