import cv2
import matplotlib.pyplot as plt

# Načítanie obrázka so šumom
image = cv2.imread('RD2.jpg', cv2.IMREAD_GRAYSCALE)

# Aplikácia Gaussovho filtra
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

# Aplikácia mediánového filtra
median_blur = cv2.medianBlur(image, 5)


cv2.imshow("original",image)
cv2.imshow("Gauss", gaussian_blur)
cv2.imshow("Median", median_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Zobrazenie výsledkov
