import cv2 as cv
import numpy as np

def init_windows():
    cv.namedWindow("image")
    cv.moveWindow("image", 100, 100)

    cv.namedWindow("fourier")
    cv.moveWindow("fourier", 500, 100)
    cv.setMouseCallback("fourier", on_mouse, 0)

    cv.namedWindow("result")
    cv.moveWindow("result", 900, 100)


def compute_fourier_transform(image):
    opt_rows = cv.getOptimalDFTSize(image.shape[0])
    opt_cols = cv.getOptimalDFTSize(image.shape[1])
    image_ = cv.copyMakeBorder(image, 0, opt_rows - image.shape[0], 0, opt_cols - image.shape[1], cv.BORDER_CONSTANT, 0)
    return cv.dft(np.float32(image_), flags=cv.DFT_COMPLEX_OUTPUT)


def vizualize(complex):
    parts = cv.split(complex)
    distances = cv.magnitude(parts[0], parts[1])

    distances += 1
    distances = cv.log(distances)
    cv.normalize(distances, distances, 1, 0, cv.NORM_INF)

    shift_quadrants(distances)
    cv.imshow("fourier", distances)


def shift_quadrants(image):
    cy, cx = image.shape[0] // 2, image.shape[1] // 2

    q0 = image[:cy, :cx].copy()
    q1 = image[:cy, cx:2 * cx].copy()
    q2 = image[cy:2 * cy, :cx].copy()
    q3 = image[cy:2 * cy, cx:2 * cx].copy()

    image[:cy, :cx], image[:cy, cx:2 * cx], image[cy:2 * cy, :cx], image[cy:2 * cy, cx:2 * cx] = q3, q2, q1, q0


def on_mouse(event, x, y, flags, param):
    global fourier
    if event != cv.EVENT_LBUTTONDOWN:
        return
    mask = np.ones(fourier.shape, dtype=np.float32)
    cv.circle(mask, (x, y), radius, (0, 0, 0), -1)
    cv.circle(mask, (mask.shape[1] - 1 - x, mask.shape[0] - 1 - y), radius, (0, 0, 0), -1)
    shift_quadrants(mask)

    filtered = cv.mulSpectrums(fourier, mask, cv.DFT_ROWS)
    fourier = filtered
    vizualize(filtered)

    inverse = compute_inverse_fourier_transform(filtered)
    cv.imshow("result", inverse)


def compute_inverse_fourier_transform(complex):
    result = cv.idft(complex)
    parts = cv.split(result)
    output = cv.magnitude(parts[0], parts[1])
    cv.normalize(output, output, 0, 1, cv.NORM_MINMAX)
    return output

## main


radius = 5
global fourier

init_windows()
image = cv.imread("input/fourier.png", cv.IMREAD_GRAYSCALE)
fourier = compute_fourier_transform(image)
vizualize(fourier)
cv.imshow("image", image)
cv.waitKey()
cv.destroyAllWindows()