import cv2
import numpy as np
import matplotlib.pyplot as plt

# Načítanie obrázka
image = cv2.imread('RD2.jpg', cv2.IMREAD_GRAYSCALE)

# **White Noise**
def add_white_noise(image, intensity=50):
    noise = np.random.uniform(-intensity, intensity, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255)  # Zabezpečí rozsah hodnôt [0, 255]
    return noisy_image.astype(np.uint8)

# **Gaussovský šum**
def add_gaussian_noise(image, mean=0, std=25):
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255)  # Zabezpečí rozsah hodnôt [0, 255]
    return noisy_image.astype(np.uint8)

def resize_to_fit_screen(image, max_width=1024, max_height=720):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

# Generovanie šumu
white_noisy_image = add_white_noise(image, intensity=50)
gaussian_noisy_image = add_gaussian_noise(image, mean=0, std=25)

cv2.imshow("Original Image", resize_to_fit_screen(image))
cv2.imshow("white_noisy_image", resize_to_fit_screen(white_noisy_image))
cv2.imshow("gaussian_noisy_image", resize_to_fit_screen(gaussian_noisy_image))
cv2.waitKey(0)
cv2.destroyAllWindows()