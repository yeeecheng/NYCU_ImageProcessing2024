import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gaussian', action='store_true')
    parser.add_argument('--median', action='store_true')
    parser.add_argument('--laplacian', action='store_true')
    args = parser.parse_args()
    return args

def padding(input_img, kernel_size):
    # """ zero padding """
    h, w = input_img.shape[:2]
    kernel_offset = (kernel_size[0] // 2, kernel_size[1] // 2)
    output_img = np.zeros((h + 2 * kernel_offset[0], w + 2 * kernel_offset[1]), dtype= input_img.dtype)
    output_img[kernel_offset[0]: kernel_offset[0] + h, kernel_offset[1]: kernel_offset[1] + w] = input_img

    return output_img


def convolution(input_img, kernel):
        
    kernel_height, kernel_width = kernel.shape
    img_height, img_width = input_img.shape[:2]
    kernel_size = kernel.shape[:2]
    kernel_offset = (kernel_size[0] // 2, kernel_size[1] // 2)
    if input_img.shape[2] == 3:
        BGR_img = []
        for c in range(3):

            padded_img = padding(input_img[:,:,c],  kernel_size)
            output_img = np.zeros_like(input_img[:,:,c], dtype=np.float32)
            for y in range(img_height):
                for x in range(img_width):
                    region = padded_img[y : y + 2 * kernel_offset[0] + 1, x : x + 2 * kernel_offset[1] + 1]
                    output_img[y, x] =  np.sum(region * kernel)

            BGR_img.append(np.clip(output_img, 0, 255).astype(np.uint8))
            
        # concat B, G, R channel
        output_img = np.dstack(BGR_img)
    
    return output_img


def gaussian_filter(input_img):
    
    sigma_x, sigma_y = 51, 51
    # column, row 
    kernel_size = (101, 101)
    kernel = np.zeros(kernel_size)
    # y, x offset
    kernel_offset = (kernel_size[0] // 2, kernel_size[1] // 2)
    sqr_sigma_x, sqr_sigma_y = sigma_x**2, sigma_y**2
    for y in range(kernel_size[0]):
        for x in range(kernel_size[1]):
            offset_y = (y - kernel_offset[0])
            offset_x = (x - kernel_offset[1])
            a = np.exp(-1 * ((offset_x**2 / (2 * sqr_sigma_x)) + (offset_y**2 / (2 * sqr_sigma_y))))
            b = np.sqrt(2 * np.pi * sqr_sigma_x) * np.sqrt(2 * np.pi * sqr_sigma_y)
            kernel[y][x] = a / b
    kernel /= np.sum(kernel)

    # 3D surface    
    # x = np.arange(kernel_size[1])
    # y = np.arange(kernel_size[0])
    # X, Y = np.meshgrid(x, y)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, kernel, cmap='viridis')
    # ax.set_title("3D Surface Plot of Gaussian Kernel")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Intensity")
    # plt.savefig(f"map_kernel{kernel_size[0]}.png")

    return convolution(input_img, kernel)

def median_filter(input_img):
    
    kernel_size = (21, 21)
    # y, x offset
    kernel_offset = (kernel_size[0] // 2, kernel_size[1] // 2)
    
    if input_img.shape[2] == 3:
        
        BGR_img = []

        for c in range(3):
            padding_img = padding(input_img[:,:,c], kernel_size)
            output_img = np.zeros_like(padding_img)
            for y in range(kernel_offset[0], output_img.shape[0] - kernel_offset[0]):
                for x in range(kernel_offset[1], output_img.shape[1] - kernel_offset[1]):
                    output_img[y][x] = np.median(\
                            padding_img[y - kernel_offset[0]: y + kernel_offset[0] + 1,\
                                        x - kernel_offset[1]: x + kernel_offset[1] + 1].\
                                            reshape((kernel_size[0] * kernel_size[1])))
                    
            BGR_img.append(output_img)
        
        # concat B, G, R channel
        output_img = np.dstack(BGR_img)

    return output_img

def laplacian_sharpening(input_img):

    # filter1
    kernel = np.array([ [0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])

    # filter2 
    kernel = np.array([ [-1, -1, -1],
                        [-1, 9, -1],
                        [-1, -1, -1]])
    
    # filter3
    # kernel = np.array([ [-1, -1, -1],
    #                     [-1, 11, -1],
    #                     [-1, -1, -1]])
    
    # return cv2.filter2D(input_img, -1, kernel)

    return convolution(input_img, kernel)

if __name__ == "__main__":
    args = parse_args()

    if args.gaussian:
        input_img = cv2.imread("input_part1.jpg")
        output_img = gaussian_filter(input_img)
        cv2.imwrite("gaussian_output.jpg", output_img)
    elif args.median:
        input_img = cv2.imread("input_part1.jpg")
        output_img = median_filter(input_img)
        cv2.imwrite("median_output.jpg", output_img)
    elif args.laplacian:
        input_img = cv2.imread("input_part2.jpg")
        output_img = laplacian_sharpening(input_img)
        cv2.imwrite("laplacian_output.jpg", output_img)