import cv2
import numpy as np
from scipy.signal import convolve2d

"""
TODO Part 1: Motion blur PSF generation
"""
def generate_motion_blur_psf(length, theta):
    
    filter_size = (length, length)
    psf = np.zeros(filter_size, dtype= np.float32)
    center = (filter_size[1] // 2, filter_size[0] // 2)

    angle = np.deg2rad(theta)
    x_offset = int(round((length / 2) * np.cos(angle)))
    y_offset = int(round((length / 2) * np.sin(angle)))

    start_point = (center[0] - x_offset, center[1] - y_offset)
    end_point = (center[0] + x_offset, center[1] + y_offset)

    cv2.line(psf, start_point, end_point, 1, thickness=1)

    psf_sum = np.sum(psf)
    if psf_sum != 0:
        psf /= psf_sum
    
    return psf


"""
TODO Part 2: Wiener filtering
"""
def apply_blur(image, psf):
    
    blurred_img = np.zeros_like(image)

    for c in range(3):
        blurred_img[:, :, c] = convolve2d(image[:, :, c], psf, mode='same', boundary='wrap')

    return blurred_img

def wiener_filtering(blurred_img, psf, k= 0.01):
    
    restored_channels = []

    for channel in cv2.split(blurred_img):
        psf_padded = np.pad(psf, 
                        [(0, blurred_img.shape[0] - psf.shape[0]), 
                         (0, blurred_img.shape[1] - psf.shape[1])], 
                        mode='constant')
        # fourier transform 
        blurred_img_freq = np.fft.fft2(channel)
        psf_freq = np.fft.fft2(psf_padded)
        psf_freq_conj = np.conj(psf_freq)
        # wiener filter
        wiener_filter = psf_freq_conj / ((np.abs(psf_freq) ** 2) + k)
        restored_img_freq = blurred_img_freq * wiener_filter
        restored_channel = np.abs(np.fft.ifft2(restored_img_freq))
        restored_channel = np.clip(restored_channel, 0, 255)
        restored_channels.append(restored_channel)
        
    return cv2.merge(restored_channels).astype(np.uint8)


"""
TODO Part 3: Constrained least squares filtering
"""
def constrained_least_square_filtering(blurred_img, psf, L= 10):

    restored_channels = []
    for channel in cv2.split(blurred_img):
        psf_padded = np.pad(psf, 
                        [(0, blurred_img.shape[0] - psf.shape[0]), 
                         (0, blurred_img.shape[1] - psf.shape[1])], 
                        mode='constant')
        # FFT
        blurred_img_freq = np.fft.fft2(channel)
        psf_freq =  np.fft.fft2(psf_padded)
        laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        laplacian_freq = np.fft.fft2(laplacian, s= channel.shape)
        psf_freq_conj = np.conj(psf_freq)

        restored_img_freq = psf_freq_conj / (np.abs(psf_freq) ** 2 + L * np.abs(laplacian_freq) ** 2)
        restored_img_freq *= blurred_img_freq
        restored_channel = np.abs(np.fft.ifft2(restored_img_freq))
        restored_channel = np.clip(restored_channel, 0, 255)
        restored_channels.append(restored_channel)
 
    return cv2.merge(restored_channels).astype(np.uint8)

"""
Bouns
"""

def inverse_filtering(blurred_img, psf):

    restored_channels = []
    for channel in cv2.split(blurred_img):
        psf_padded = np.pad(psf, 
                        [(0, blurred_img.shape[0] - psf.shape[0]), 
                         (0, blurred_img.shape[1] - psf.shape[1])], 
                        mode='constant')
        # convert to frequency domain
        blurred_img_freq = np.fft.fft2(channel)
        psf_freq = np.fft.fft2(psf_padded) 
        epsilon = 0.05
        F_deblurred = blurred_img_freq / (psf_freq + epsilon)
        
        restored_channel = np.abs(np.fft.ifft2(F_deblurred))
        restored_channel = np.clip(restored_channel, 0, 255)
        restored_channels.append(restored_channel)

    return cv2.merge(restored_channels).astype(np.uint8)

def compute_PSNR(image_original, image_restored):
    # PSNR = 10 * log10(max_pixel^2 / MSE)
    psnr = 10 * np.log10(255 ** 2 / np.mean((image_original.astype(np.float64) - image_restored.astype(np.float64)) ** 2))

    return psnr


"""
Main function
"""
def main():
    for i in range(2):
        img_original = cv2.imread("data/image_restoration/testcase{}/input_original.png".format(i + 1))
        img_blurred = cv2.imread("data/image_restoration/testcase{}/input_blurred.png".format(i + 1))

        # TODO Part 1: Motion blur PSF generation
        length, theta = 40, -45
        psf = generate_motion_blur_psf(length, theta)
        # cv2.imshow("psf", psf * 255)
        cv2.imwrite(f"./testcase{i}_psf_length{length}_theta{theta}.png", (psf * 255))
        psf_blurred_img = apply_blur(img_original, psf)
        # cv2.imshow("psf_blurred_img", np.hstack([img_blurred, psf_blurred_img]))
        print("PSNR with psf blurred img and blurred img= {}\n".format(compute_PSNR(img_blurred, psf_blurred_img)))
        # cv2.imshow("blurred", np.hstack([img_blurred, psf_blurred_img]))
        cv2.imwrite(f"./testcase{i}_psf_blurred_img_length{length}_theta{theta}.png", psf_blurred_img)
        # # TODO Part 2: Wiener filtering
        ### experiment
        # k_wiener_img = []
        # for k in [1, 0.1, 0.01, 0.001, 0.0001]:
        #     wiener_img = wiener_filtering(img_blurred, psf, k)
        #     k_wiener_img.append(wiener_img)
        # cv2.imwrite(f"./testcase{i}_k_wiener_img_length{length}_theta{theta}.png", np.hstack(k_wiener_img))
        ###
        wiener_img = wiener_filtering(img_blurred, psf)
        print("\n---------- Testcase {} ----------".format(i))
        print("Method: Wiener filtering")
        print("PSNR = {}\n".format(compute_PSNR(img_original, wiener_img)))
        
        # # TODO Part 3: Constrained least squares filtering
        ### experiment
        # lambda_constrained_least_square_img = []
        # for L in [10, 1, 0.1, 0.01, 0.001]:
        #     constrained_least_square_img = constrained_least_square_filtering(img_blurred, psf, L)
        #     lambda_constrained_least_square_img.append(constrained_least_square_img)
        # cv2.imwrite(f"./testcase{i}_lambda_constrained_least_square_img_length{length}_theta{theta}.png", np.hstack(lambda_constrained_least_square_img))
        ###
        constrained_least_square_img = constrained_least_square_filtering(img_blurred, psf)

        print("Method: Constrained least squares filtering")
        print("PSNR = {}\n".format(compute_PSNR(img_original, constrained_least_square_img)))
        # cv2.imshow(f"window{i}", np.hstack([img_blurred, wiener_img, constrained_least_square_img]))
        cv2.imwrite(f"./testcase{i}_wiener_length{length}_theta{theta}.png", wiener_img)
        cv2.imwrite(f"./testcase{i}_cls_length{length}_theta{theta}.png", constrained_least_square_img)
        ### bonus
        # inverse_image = inverse_filtering(img_blurred, psf)
        # cv2.imwrite(f"./testcase{i}_inverse_image_length{length}_theta{theta}.png", inverse_image)
        ###
        # cv2.waitKey(0)

if __name__ == "__main__":
    main()
