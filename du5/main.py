import cv2 as cv
import matplotlib.pyplot as plt

def du5():
    img = cv.imread("input/jigsaw1.jpg")
    print(img.shape)
    img_resized = cv.resize(img, (780, 540),
                            interpolation=cv.INTER_LINEAR)

    img_blured1 = cv.GaussianBlur(img_resized, (3, 3), 0)
    img_blured2 = cv.medianBlur(img_resized, 3)
    img_blured3 = cv.bilateralFilter(img_resized, 9, 75, 75)

    cv.imshow("Blured Gaussian", img_blured1)
    cv.imshow("Blured Median", img_blured2)
    cv.imshow("Bilateral filter", img_blured3)

    img_gray1 = cv.cvtColor(img_blured1, cv.IMREAD_GRAYSCALE)
    img_gray2 = cv.cvtColor(img_blured2, cv.IMREAD_GRAYSCALE)
    img_gray3 = cv.cvtColor(img_blured3, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"

    #laplacian = cv.Laplacian(img_gray1, ddepth=cv.CV_16S, ksize=3)

    #cv.imshow("Laplacian", laplacian)

    cv.waitKey(0)
    cv.destroyAllWindows()

    edges = cv.Canny(img_gray1, 50, 200)
    edges_resized = cv.resize(edges, (1500, 750),
                              interpolation=cv.INTER_LINEAR)

    # ced_change(img_gray, edges_resized)

    cv.imshow("Edges", edges_resized)

    cv.waitKey(0)
    cv.destroyAllWindows()

def nothing(x):
    pass

def ced_change(img, edges_resized):
    cv.namedWindow('image')
    cv.createTrackbar('maxThreshold', 'image', 0, 255, nothing)
    cv.createTrackbar('minThreshold', 'image', 0, 255, nothing)
    edge = img

    while 1:
        cv.imshow('image', edge)
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break

        minT = cv.getTrackbarPos('minThreshold', 'image')
        maxT = cv.getTrackbarPos('maxThreshold', 'image')
        edge = cv.Canny(edges_resized, minT, maxT)
        cv.imwrite("output/canny.jpg", edge)

if __name__ == '__main__':
    du5()
