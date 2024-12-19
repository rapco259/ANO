import cv2
import numpy as np

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

# Funkcia na zmenšenie obrázka tak, aby sa zmestil do 1920x1080
def resize_to_fit_screen(image, max_width=1024, max_height=720):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

# Načítanie obrázka
image = cv2.imread('RD2.jpg', cv2.IMREAD_GRAYSCALE)
image_resized = resize_to_fit_screen(image)

# Pridanie šumu
gaussian_noisy = resize_to_fit_screen(add_gaussian_noise(image_resized))
salt_and_pepper_noisy = resize_to_fit_screen(add_salt_and_pepper_noise(image_resized))
quantization_noisy = resize_to_fit_screen(add_quantization_noise(image_resized))

# Filtrácia šumu
gaussian_filtered = resize_to_fit_screen(cv2.GaussianBlur(gaussian_noisy, (5, 5), 0))
bilateral_filtered = resize_to_fit_screen(cv2.bilateralFilter(gaussian_noisy, 9, 75, 75))
median_filtered = resize_to_fit_screen(cv2.medianBlur(salt_and_pepper_noisy, 5))
quantization_interpolated = resize_to_fit_screen(cv2.GaussianBlur(quantization_noisy, (3, 3), 0))

# Zobrazenie obrázkov v samostatných oknách
cv2.imshow("Original Image", image_resized)
cv2.imshow("Gaussian Noise", gaussian_noisy)
cv2.imshow("Gaussian Filtered", gaussian_filtered)
cv2.imshow("Bilateral Filtered", bilateral_filtered)
cv2.imshow("Salt & Pepper Noise", salt_and_pepper_noisy)
cv2.imshow("Median Filtered", median_filtered)
cv2.imshow("Quantization Noise", quantization_noisy)
cv2.imshow("Quantization Interpolated", quantization_interpolated)

# Čakanie na stlačenie klávesy
cv2.waitKey(0)
cv2.destroyAllWindows()
