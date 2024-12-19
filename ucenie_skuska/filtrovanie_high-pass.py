import cv2
import numpy as np

# Načítanie obrázka v grayscale
image = cv2.imread('RD2.jpg', cv2.IMREAD_GRAYSCALE)

def resize_to_fit_screen(image, max_width=1024, max_height=720):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

# Fourierova transformácia
dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)  # Posun nulových frekvencií do stredu

# Vytvorenie vysokopriepustného filtra
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2
mask = np.ones((rows, cols, 2), np.uint8)  # Začíname s jednotkami
mask[crow-30:crow+30, ccol-30:ccol+30] = 0  # Nízke frekvencie v strede potlačíme

# Aplikácia filtra vo frekvenčnej oblasti
fshift = dft_shift * mask

# Inverzná Fourierova transformácia
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# Normalizácia obrázka
img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
img_back = np.uint8(img_back)

image = resize_to_fit_screen(image)
img_back = resize_to_fit_screen(img_back)

# Zobrazenie výsledkov
cv2.imshow("Original Image", image)
cv2.imshow("High-pass Filtered Image", img_back)

# Čakanie na stlačenie ľubovoľnej klávesy
cv2.waitKey(0)
cv2.destroyAllWindows()
