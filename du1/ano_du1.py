import matplotlib.pyplot as plt
import cv2 as cv2
import numpy as np

def show(image, dpi, title):
    plt.figure(dpi=dpi)
    plt.title(title)
    plt.imshow(image, cmap='gray')
    plt.show()
    plt.close()

def sedotonovy():
    img_cierny = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite("mmm_cierny.png", img_cierny)
    show(img_cierny, 200, "povodny obrazok na sedotonovy")
    # cierny obrazok basic
def histogram():
    img_cierny = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([img_cierny], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title("histogram")
    plt.savefig("histogram.png", bbox_inches='tight')

    # spravim histogram zo sedotonoveho obrazka, ulozim si plt

    histogram_image = cv2.imread("histogram.png")
    histogram_image = cv2.cvtColor(histogram_image, cv2.COLOR_BGR2GRAY)

    # nacitam si to ako obrazok, plt malo shape (256,1)

    x_offset = img_cierny.shape[1] - histogram_image.shape[1]
    y_offset = img_cierny.shape[0] - histogram_image.shape[0]

    # mam offsety aby som dal obrazok do praveho dolneho rohu

    print(histogram_image.shape)
    print(x_offset, y_offset)

    img_cierny[y_offset:y_offset + histogram_image.shape[0],
    x_offset:x_offset + histogram_image.shape[1]] = histogram_image

    show(img_cierny, 200, "sedotonovy obrazok a histogramom")
    plt.close()
    # obrazok s histogramom, ciernobiele

def vyhladzovanie():
    img = cv2.imread('mmm.png', 0)

    # vyhladzovanie podla stranky

    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    plt.plot(cdf_normalized, color='b')
    plt.hist(img.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.title("before eq")
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.savefig("histogram_before_ge.png", bbox_inches='tight')
    plt.show()

    # vyhladzovanie
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img2 = cdf[img] # vyhladeny obrazok
    hist2, bins2 = np.histogram(img2.flatten(), 256, [0, 256])
    cdf2 = hist2.cumsum()
    cdf2_normalized = cdf2 * hist2.max() / cdf2.max()

    plt.plot(cdf2_normalized, color='b')  # Plot CDF in blue
    plt.hist(img2.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.title("after eq")
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.savefig("histogram_after_ge.png", bbox_inches='tight')
    plt.show()

    # ukazany vyhladeny obrazok
    cv2.imwrite("mmm_vyhladeny.png", img2)
    show(img2, 200, "")


def equlization_in_OPENCV():
    # tu len som skusal equalizaciu priamo z opencv
    img = cv2.imread('mmm.png', 0)
    equ = cv2.equalizeHist(img)
    res = np.hstack((img, equ))  # stacking images side-by-side
    cv2.imwrite('mmm_eqOPENCV.png', res)
    show(res, 300, "")


def clahe_eq():
    # skuska clahe priamo na obrazok
    img = cv2.imread('mmm.png', 0)

    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)

    cv2.imwrite('mmm_clahe.jpg', cl1)
    show(cl1, 100, "")


def clahe_vyhladzovanie():
    # clahe vyhladzovanie na histogram
    img = cv2.imread('mmm.png', 0)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahef = clahe.apply(img)

    hist, bins = np.histogram(clahef.flatten(),
                              bins=256,
                              range=[0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max()

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.hist(clahef.flatten(), bins=256, range=[0, 256], color='r', )
    ax.set_xlabel('Pixel Intensity')
    ax.set_xlim(0, 255)

    ax2 = ax.twinx()
    ax2.plot(cdf_normalized, color='b')
    ax2.set_ylabel('CDF')
    ax2.set_ylim(0, 1)

    # ulozim obrazok vyhladzovania
    plt.savefig("clahe.png", bbox_inches='tight')
    plt.title("clahe vyhladzovanie")
    plt.show()


def RGB():
    img = cv2.imread('mmm.png')  # bez nuly

    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # vypocitam histogram pre kazdu cast farby
    B_histo = cv2.calcHist([img_RGB], [0], None, [256], [0, 256])
    G_histo = cv2.calcHist([img_RGB], [1], None, [256], [0, 256])
    R_histo = cv2.calcHist([img_RGB], [2], None, [256], [0, 256])

    # histogramy rgb
    plt.subplot(2, 2, 1)
    plt.plot(B_histo, 'b')
    plt.subplot(2, 2, 2)
    plt.plot(G_histo, 'g')
    plt.subplot(2, 2, 3)
    plt.plot(R_histo, 'r')
    plt.title("RGB obrazok")
    plt.show()

    # vezmem si cely obrazok, kazdu farbu
    B = img_RGB[:, :, 0]  # blue layer
    G = img_RGB[:, :, 1]  # green layer
    R = img_RGB[:, :, 2]  # red layer

    # equlizujem pomocou opencv

    b_equi = cv2.equalizeHist(B)
    g_equi = cv2.equalizeHist(G)
    r_equi = cv2.equalizeHist(R)

    # mam equalizovane obrazky v kazdej farbe, spojim ich cez cv2 merge

    equi_im = cv2.merge([b_equi, g_equi, r_equi])

    # kazda zlozka solo

    # plt.imshow(b_equi)
    # plt.title("b_equi")
    # plt.show()
    # plt.imshow(g_equi)
    # plt.title("g_equi")
    # plt.show()
    # plt.imshow(r_equi)
    # plt.title("r_equi")
    # plt.show()

    # vypocitam histogramy z equilizovaneho obrazka pre kazdu zlozku rgb
    B_histo = cv2.calcHist([b_equi], [0], None, [256], [0, 256])
    G_histo = cv2.calcHist([g_equi], [0], None, [256], [0, 256])
    R_histo = cv2.calcHist([r_equi], [0], None, [256], [0, 256])

    # equi rgb zlozky histogramy
    plt.subplot(2, 2, 1)
    plt.plot(G_histo, 'g')
    plt.subplot(2, 2, 2)
    plt.plot(R_histo, 'r')
    plt.subplot(2, 2, 3)
    plt.plot(B_histo, 'b')
    plt.title("RGB eq obrazok")
    plt.show()

    plt.figure(figsize=(10, 12), )

    # porovnanie original vs equi

    plt.subplot(3, 2, 1)
    plt.title("Green Original")
    plt.plot(G_histo, 'g')

    plt.subplot(3, 2, 2)
    plt.title("Green Equilized")
    plt.plot(G_histo, 'g')

    plt.subplot(3, 2, 3)
    plt.title("Red Original")
    plt.plot(R_histo, 'r')

    plt.subplot(3, 2, 4)
    plt.title("Red Equilized")
    plt.plot(R_histo, 'r')

    plt.subplot(3, 2, 5)
    plt.title("Blue Original")
    plt.plot(B_histo, 'b')

    plt.subplot(3, 2, 6)
    plt.title("Blue Equilized")
    plt.plot(B_histo, 'b')

    plt.show()

    res = np.hstack((img_RGB, equi_im))
    res_final = cv2.cvtColor(res, cv2.COLOR_RGB2BGR) # convert naspat na BGR, cv2 potom zobrazi normalne
    cv2.imwrite('RGB_image.png', res_final)
    # porovnanie obrazkov na konci

    plt.figure(dpi=200)
    plt.imshow(res)
    plt.title("Porovnanie rgb obrazka OG vs Eq")
    plt.show()
    plt.close()



if __name__ == '__main__':
    global img_cierny
    global img
    img = cv2.imread("mmm.png")
    #### metody na skusku
    # equlization_in_OPENCV()
    # clahe_eq()
    #### metody na skusku

    ### real metody ###

    sedotonovy()
    histogram()

    # vypise aj histogram samotny a aj obrazok s histogramom,
    # myslim si ze je to tym riadkom pretoze tam
    # vkladam histogram priamo do toho obrazka

    vyhladzovanie()

    clahe_vyhladzovanie()

    RGB()