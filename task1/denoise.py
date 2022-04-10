import cv2
import numpy as np

def task1_2(src_path, clean_path, dst_path):
    """
    This is main function for task 1.
    It takes 3 arguments,
    'src_path' is path for source image.
    'clean_path' is path for clean image.
    'dst_path' is path for output image, where your result image should be saved.

    You should load image in 'src_path', and then perform task 1-2,
    and then save your result image to 'dst_path'.
    """
    noisy_img = cv2.imread(src_path)
    clean_img = cv2.imread(clean_path)

    # do noise removal
    # result_img = apply_median_filter(noisy_img, 3)

    # bilateral
    result_img = apply_bilateral_filter(noisy_img,3,1,60)

    cv2.imwrite(dst_path, result_img)
    print(calculate_rms(clean_img, result_img))
    pass


def apply_median_filter(img, kernel_size):
    """
    You should implement median filter using convolution in this function.
    It takes 2 arguments,
    'img' is source image, and you should perform convolution with median filter.
    'kernel_size' is an int value, which determines kernel size of median filter.

    You should return result image.
    """
    pad_size = kernel_size/2
    temp_shape = img.shape
    img = np.lib.pad(img, [((int(pad_size)),(int(pad_size))),((int(pad_size)),(int(pad_size))),(0,0)],"mean")
    dnoise_img = np.zeros((temp_shape[0],temp_shape[1],3))
    print(img.shape)
    for i in range(3):
        for j in range(0, dnoise_img.shape[0]):
            for k in range(0, dnoise_img.shape[1]):
                med_filt = img[j:j+kernel_size, k:k+kernel_size,i].ravel()
                med_filt = np.sort(med_filt)
                # print(j,k, i)
                dnoise_img[j,k,i] = med_filt[int(kernel_size*kernel_size/2)]
    return dnoise_img.clip(0,255)


def apply_bilateral_filter(img, kernel_size, sigma_s, sigma_r):
    """
    You should implement bilateral filter using convolution in this function.
    It takes at least 4 arguments,
    'img' is source image, and you should perform convolution with median filter.
    'kernel_size' is a int value, which determines kernel size of average filter.
    'sigma_s' is a int value, which is a sigma value for G_s(gaussian function for space)
    'sigma_r' is a int value, which is a sigma value for G_r(gaussian function for range)

    You should return result image.
    """
    H, W = img.shape[0], img.shape[1]
    C = 1 if len(img.shape) == 2 else img.shape[2]
    img = img.reshape(H, W, C)
    output_image = img.copy()

    for i in range(kernel_size, H - kernel_size):
        for j in range(kernel_size, W - kernel_size):
            for k in range(C):
                weight_sum = 0.0
                pixel_sum = 0.0
                for x in range(-kernel_size, kernel_size + 1):
                    for y in range(-kernel_size, kernel_size + 1):
                        spatial_weight = -(x ** 2 + y ** 2) / (2 * (sigma_s ** 2))
                        color_weight = -(int(img[i][j][k]) - int(img[i + x][j + y][k])) ** 2 / (2 * (sigma_r ** 2))
                        weight = np.exp(spatial_weight + color_weight)
                        weight_sum += weight
                        pixel_sum += (weight * img[i + x][j + y][k])
                value = pixel_sum / weight_sum
                output_image[i][j][k] = value
                # print(output_image[i][j][k], '->', value)
    return output_image.astype(np.uint8)


def apply_my_filter(img):
    """
    You should implement additional filter using convolution.
    You can use any filters for this function, except median, bilateral filter.
    You can add more arguments for this function if you need.

    You should return result image.
    """
    return img


def calculate_rms(img1, img2):
    """
    Calculates RMS error between two images. Two images should have same sizes.
    """
    if (img1.shape[0] != img2.shape[0]) or \
            (img1.shape[1] != img2.shape[1]) or \
            (img1.shape[2] != img2.shape[2]):
        raise Exception("img1 and img2 should have same sizes.")

    diff = np.abs(img1 - img2)
    diff = np.abs(img1.astype(dtype=np.int) - img2.astype(dtype=np.int))
    return np.sqrt(np.mean(diff ** 2))

# task1_2("./test_images/cat_noisy.jpg", "./test_images/cat_clean.jpg", "./cat_median.jpg")
task1_2("./test_images/fox_noisy.jpg", "./test_images/fox_clean.jpg", "./fox_bilateral.jpg")