import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
TODO Part 1: Gamma correction
"""
def gamma_correction(img, gamma):

    h, w = img.shape[:2]
    enhenced_img = np.zeros_like(img)

    for y in range(h):
        for x in range(w):
            for c in range(3):
                # normalize to [0, 1]
                nor_pixel = img[y, x][c].astype(np.float64) / 255.0
                # compute with gamma and normalize to [0, 255] 
                enhenced_img[y, x][c] = (np.pow(nor_pixel, gamma) * 255.0).astype(np.uint8)
    return enhenced_img

def gamma_correction_cv2(img, gamma):

    look_up_table = (pow(np.arange(256) / 255, gamma) *
                     255).astype(np.uint8).reshape((1, 256))                 
    res = cv2.LUT(img, look_up_table)
    return res

"""
TODO Part 2: Histogram equalization
"""

def show_histogram(table):

    plt.figure(figsize=(10, 6))
    plt.bar(range(256), table, color='blue', edgecolor='black', width=1.0)
    plt.title("Histogram Distribution")
    plt.xlabel("Pixel Value (0-255)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.show()

def show_cdf(cdf_normalized):
    plt.figure(figsize=(12, 6))

    plt.title("CDF (Normalized)")
    plt.plot(cdf_normalized, color='green')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def histogram_equalization(img):
    
    h, w = img.shape[:2]
    enhenced_img = np.zeros_like(img)
    
    histogram_table = np.array([[0 for _ in range(256)] for _ in range(3)], np.int32)

    for c in range(3):
        for y in range(h):
            for x in range(w):
                # print(img[y, x, c])
                histogram_table[c][img[y, x, c]] += 1
        # get the cdf 
        cdf = histogram_table[c].cumsum()
        # normalize cdf
        cdf_normalized = cdf * float(histogram_table.max()) / cdf.max()
        # show_cdf(cdf_normalized)
        # ignore zero value
        cdf_m = np.ma.masked_equal(cdf,0)
        # normalize to [0, 255], then get the new pixel
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        # fill zero value
        cdf = np.ma.filled(cdf_m,0).astype('uint8')
        # map to new pixel
        enhenced_img[:, :, c] = cdf[img[:, :, c]]

    return enhenced_img

def histogram_equalization_cv2(img):

    enhenced_img = np.zeros_like(img)
    for c in range(3):
        enhenced_img[:, :, c] = cv2.equalizeHist(img[:, :, c])
    return enhenced_img

"""
Bonus
"""
def enhance_sharpness(img):
    # Sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened

"""
Main function
"""
def main():
    img = cv2.imread("data/image_enhancement/input.bmp")

    # # TODO: modify the hyperparameter
    gamma_list = [0.2, 1, 10] # gamma value for gamma correction

    # # TODO Part 1: Gamma correction
    for gamma in gamma_list:
        gamma_correction_img = gamma_correction(img, gamma)
        # cv2.imshow("Gamma correction | Gamma = {}".format(gamma), np.vstack([img, gamma_correction_img]))
        cv2.imwrite(f"./gamma_correction_{gamma}.png",gamma_correction_img)
    # # TODO Part 2: Image enhancement using the better balanced image as input
    histogram_equalization_img = histogram_equalization(img)
    histogram_equalization_img = histogram_equalization_cv2(img)
    cv2.imwrite(f"./histogram.png",histogram_equalization_img)
    # cv2.imshow("Histogram equalization", np.vstack([img, histogram_equalization_img]))

    ### bounus
    # sharpened_img = enhance_sharpness(img)
    # cv2.imwrite(f"./sharpened_img.png", sharpened_img)
    #
    # cv2.waitKey(0)

if __name__ == "__main__":
    main()
