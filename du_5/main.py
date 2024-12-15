import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("input/jigsaw1.jpg")
print(img.shape)
img_resized = cv.resize(img, (780, 540),
                        interpolation=cv.INTER_LINEAR)

img_blured = cv.GaussianBlur(img_resized, (5, 5), 0)
img_blured2 = cv.medianBlur(img_resized, 5)
img_blured3 = cv.bilateralFilter(img_resized, 9, 75, 75)

#cv.imshow("Blured Gaussian", img_blured)
#cv.imshow("Blured Median", img_blured2)
cv.imshow("Bilateral filter", img_blured3)

img_gray = cv.cvtColor(img_blured3, cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
edges = cv.Canny(img, 100, 200)

plt.show()

edges_resized = cv.resize(edges, (780, 540),
                          interpolation=cv.INTER_LINEAR)

cv.imshow("Edges", edges_resized)

cv.waitKey(0)
cv.destroyAllWindows()
