import cv2
import os
import numpy as np
import pywt
import pywt.data
import matplotlib.pyplot as plt

# Constant matrices for 2d-filtering
EDGE_DETECTION_KERNEL = np.array([[-1, -1, -1],
                                  [-1, 8, -1],
                                  [-1, -1, -1]])
SHARPEN_KERNEL = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
IDENTITY_KERNEL = np.array([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]])
BOX_BLUR_KERNEL = np.dot((1 / 9), np.array([[1, 1, 1],
                                            [1, 1, 1],
                                            [1, 1, 1]]))
GAUSSIAN_BLUR_KERNEL = (1 / 256) * np.array([[1, 4, 6, 4, 1],
                                             [4, 16, 24, 16, 4],
                                             [6, 24, 36, 24, 6],
                                             [4, 16, 24, 16, 4],
                                             [1, 4, 6, 4, 1]])
BLUR_IMAGE_KERNEL = np.ones((5, 5), np.float32) / 30


# Image blurring class
class ImageBlur:
    def __init__(self, input_file):
        self.input_file = input_file
        if not os.path.exists(self.input_file):
            raise ValueError(f"Input file '{self.input_file}' does not exist.")
        self.image = cv2.imread(self.input_file)
        if self.image is None:
            raise ValueError(f"Failed to load input file '{self.input_file}'.")

    def apply_gaussian_blur(self, kernel_size=(5, 5), sigma_x=0):
        if self.image is not None:
            self.image = cv2.GaussianBlur(self.image, kernel_size, sigma_x)

    def apply_box_blur(self, kernel_size=(5, 5)):
        if self.image is not None:
            self.image = cv2.blur(self.image, kernel_size)

    def save_blurred_image(self, output_file):
        if self.image is not None:
            cv2.imwrite(output_file, self.image)

    def show_blurred_image(self):
        if self.image is not None:
            cv2.imshow('Blurred Image', self.image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# Image Enhancement class
class ImageEnhancement:
    def __init__(self, input_file):
        self.input_file = input_file
        if not os.path.exists(self.input_file):
            raise ValueError(f"Input file '{self.input_file}' does not exist.")
        self.image = cv2.imread(self.input_file)
        if self.image is None:
            raise ValueError(f"Failed to load input file '{self.input_file}'.")

    def gamma_correction(self, gamma=1.0):
        if self.image is not None:
            self.image = cv2.pow(self.image / 255.0, gamma)
            self.image = cv2.normalize(self.image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    def filter2D(self, conv_kernel, output_image_depth=-1):
        if self.image is not None:
            self.image = cv2.filter2D(self.image, output_image_depth, conv_kernel)

    # 2d-filtering via various kernels
    def edge_detection_filter(self, conv_kernel, output_image_depth=-1):
        if self.image is not None:
            self.image = cv2.filter2D(self.image, output_image_depth, EDGE_DETECTION_KERNEL)

    def sharpen_filter(self, conv_kernel, output_image_depth=-1):
        if self.image is not None:
            self.image = cv2.filter2D(self.image, output_image_depth, SHARPEN_KERNEL)

    def identity_filter(self, conv_kernel, output_image_depth=-1):
        if self.image is not None:
            self.image = cv2.filter2D(self.image, output_image_depth, IDENTITY_KERNEL)

    def box_blur_filter(self, conv_kernel, output_image_depth=-1):
        if self.image is not None:
            self.image = cv2.filter2D(self.image, output_image_depth, BOX_BLUR_KERNEL)

    def gaussian_blur_filter(self, conv_kernel, output_image_depth=-1):
        if self.image is not None:
            self.image = cv2.filter2D(self.image, output_image_depth, GAUSSIAN_BLUR_KERNEL)

    def zero_kernel_blurring_filter(self, conv_kernel=BLUR_IMAGE_KERNEL, output_image_depth=-1):
        if self.image is not None:
            self.image = cv2.filter2D(self.image, output_image_depth, conv_kernel)

    # Blurring image using median filter
    def median_filter(self, filter_size=3):
        if self.image is not None:
            self.image = cv2.medianBlur(self.image, filter_size)

    # Fourier transformation

    def fourier_transform(self, method='dft'):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            if method == 'dft':
                transformed_image = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)
                shifted_image = np.fft.fftshift(transformed_image)
                magnitude_spectrum = 20 * np.log(cv2.magnitude(shifted_image[:, :, 0], shifted_image[:, :, 1]))
                self.image = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            elif method == 'sft':
                transformed_image = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)
                fourier_shift = np.fft.fftshift(fourier)
                transformed_image = cv2.sftd(gray_image)
                self.image = cv2.normalize(transformed_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            else:
                raise ValueError(f"Invalid method '{method}' specified. Supported methods are 'dft' and 'sft'.")

    def wavelet_transform(self):
        coeffs2 = pywt.dwt2(self.image, 'bior1.3')
        LL, (LH, HL, HH) = coeffs2

    def show_image(self, window_name='Image'):
        if self.image is not None:
            cv2.imshow(window_name, self.image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def save_image(self, output_file):
        if self.image is not None:
            cv2.imwrite(output_file, self.image)


class WaveletTransform:
    def __init__(self, input_file):
        self.input_file = input_file
        if not os.path.exists(self.input_file):
            raise ValueError(f"Input file '{self.input_file}' does not exist.")
        self.image = cv2.imread(self.input_file)
        if self.image is None:
            raise ValueError(f"Failed to load input file '{self.input_file}'.")
        self.coeffs = None

    def perform_wavelet_transform(self):
        if self.image is not None:
            self.coeffs = pywt.dwt2(self.image, 'bior1.3')

    def show_wavelet_transform(self):
        if self.coeffs is not None:
            LL, (LH, HL, HH) = self.coeffs
            titles = ['Approximation', 'Horizontal detail',
                      'Vertical detail', 'Diagonal detail']
            fig = plt.figure(figsize=(12, 3))
            for i, a in enumerate([LL, LH, HL, HH]):
                ax = fig.add_subplot(1, 4, i + 1)
                ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
                ax.set_title(titles[i], fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])

            fig.tight_layout()
            plt.show()

    def save_wavelet_transform_figure(self, output_file):
        if self.coeffs is not None:
            LL, (LH, HL, HH) = self.coeffs
            titles = ['Approximation', 'Horizontal detail',
                      'Vertical detail', 'Diagonal detail']
            fig = plt.figure(figsize=(12, 3))
            for i, a in enumerate([LL, LH, HL, HH]):
                ax = fig.add_subplot(1, 4, i + 1)
                ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
                ax.set_title(titles[i], fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])

            fig.tight_layout()
            fig.savefig(output_file)


if __name__ == "__main__":
    original_file = "./images/Samoyed-dog.webp"

    wavelet_transform = WaveletTransform(original_file)
    wavelet_transform.perform_wavelet_transform()
    wavelet_transform.show_wavelet_transform()
    wavelet_transform.save_wavelet_transform_figure("./images/wavelet_transform.jpg")
"""
    enchanted = ImageEnhancement(original_file)
    enchanted.show_image(window_name="Original Image")

    enchanted.filter2D(BOX_BLUR_KERNEL)
    enchanted.save_image('./images/edge_detection_image.jpg')
    enchanted.show_image(window_name="BOX_BLUR_KERNEL")

    enchanted.fourier_transform(method="sft")
    enchanted.save_image('./images/fourier_image.jpg')
    enchanted.show_image(window_name="Fourier sft")
"""
"""
    blur = ImageBlur(path_to_file)
    blur.apply_gaussian_blur(kernel_size=(15, 15), sigma_x=0)
    blur.save_blurred_image('./images/blured_image.jpg')
    blur.show_blurred_image()

    blured_image_path = "./blured_image.jpg"

    enhance = ImageEnhancement(blured_image_path)

    enhance.gamma_correction(gamma=0.5)
    enhance.save_image('./images/gamma_correction_image.jpg')
    enhance.show_image(window_name="Gamma Correction")

    kernel_edge_detection = np.array([[-1, -1, -1],
                                      [-1, 8, -1],
                                      [-1, -1, -1]])
    kernel_sharpen = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    enhance.filter2D(kernel_edge_detection)
    enhance.save_image('./images/Filter2D_image.jpg')
    enhance.show_image(window_name="Filter 2D")
    enhance.median_filter(filter_size=5)
    enhance.save_image('./images/median_filter_image.jpg')
    enhance.show_image(window_name="Median Filter")

    enhance.fourier_transform(method="dft")
    enhance.save_image('./images/fourier_transform_image.jpg')
    enhance.show_image(window_name="Fourier Transform")

    image = cv2.imread(path_to_file)

    # Creating the kernel(2d convolution matrix)
    kernel2 = np.array([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]])

    # Applying the filter2D() function
    img = cv2.filter2D(src=image, ddepth=-1, kernel=kernel2)

    # Shoeing the original and output image
    cv2.imshow('Original', image)
    cv2.imshow('Kernel Blur', img)

    cv2.waitKey()
    cv2.destroyAllWindows()"""
