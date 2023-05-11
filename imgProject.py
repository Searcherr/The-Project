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
    def edge_detection_filter(self, output_image_depth=-1):
        if self.image is not None:
            self.image = cv2.filter2D(self.image, output_image_depth, EDGE_DETECTION_KERNEL)

    def sharpen_filter(self, output_image_depth=-1):
        if self.image is not None:
            self.image = cv2.filter2D(self.image, output_image_depth, SHARPEN_KERNEL)

    def identity_filter(self, output_image_depth=-1):
        if self.image is not None:
            self.image = cv2.filter2D(self.image, output_image_depth, IDENTITY_KERNEL)

    def box_blur_filter(self, output_image_depth=-1):
        if self.image is not None:
            self.image = cv2.filter2D(self.image, output_image_depth, BOX_BLUR_KERNEL)

    def gaussian_blur_filter(self, output_image_depth=-1):
        if self.image is not None:
            self.image = cv2.filter2D(self.image, output_image_depth, GAUSSIAN_BLUR_KERNEL)

    def zero_kernel_blurring_filter(self, conv_kernel=BLUR_IMAGE_KERNEL, output_image_depth=-1):
        if self.image is not None:
            self.image = cv2.filter2D(self.image, output_image_depth, conv_kernel)

    # Blurring image using median filter
    def median_filter(self, filter_size=5):
        if self.image is not None:
            self.image = cv2.medianBlur(self.image, filter_size)

    def show_image(self, window_name='Image'):
        if self.image is not None:
            cv2.imshow(window_name, self.image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def save_image(self, output_file):
        if self.image is not None:
            cv2.imwrite(output_file, self.image)


# Implementing Wavelet Transformation
class WaveletTransform:
    def __init__(self, input_file):
        self.input_file = input_file
        if not os.path.exists(self.input_file):
            raise ValueError(f"Input file '{self.input_file}' does not exist.")
        self.image = cv2.imread(self.input_file)
        if self.image is None:
            raise ValueError(f"Failed to load input file '{self.input_file}'.")
        self.coeffs = None

    def show_original_image(self, window_name="Original Image"):
        if self.image is not None:
            cv2.imshow(window_name, self.image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


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


# Implementing Fourier Transformation
class FourierTransform:
    def __init__(self, input_file):
        self.input_file = input_file
        if not os.path.exists(self.input_file):
            raise ValueError(f"Input file '{self.input_file}' does not exist.")
        self.image = cv2.cvtColor(cv2.imread(self.input_file), cv2.COLOR_BGR2GRAY)
        if self.image is None:
            raise ValueError(f"Failed to load input file '{self.input_file}'.")
        self.magnitude_spectrum = None
        self.fft = None

    def set_fft(self):
        if self.image is not None:
            dft = cv2.dft(np.float32(self.image), flags=cv2.DFT_COMPLEX_OUTPUT)
            self.fft = np.fft.fftshift(dft)

    def set_magnitude_spectrum(self):
        if self.fft is not None:
            self.magnitude_spectrum = 20 * np.log(cv2.magnitude(self.fft[:, :, 0], self.fft[:, :, 1]))

    def show_frequency_spectrum(self, window_name="Magnitude Spectrum"):
        if self.magnitude_spectrum is not None:
            cv2.imshow(window_name, self.magnitude_spectrum)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def show_spectrum_plt(self):
        if self.magnitude_spectrum is not None:
            plt.imshow(self.magnitude_spectrum, cmap='gray')
            plt.title('Input Image')
            plt.axis('off')
            plt.show()

    def save_magnitude_image(self, output_file):
        if self.magnitude_spectrum is not None:
            cv2.imwrite(output_file, self.magnitude_spectrum)


if __name__ == "__main__":
    """
    path_to_original_image = "./images/Samoyed-dog.webp"
    image = cv2.imread(path_to_original_image)
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    # Testing ImageBlur class
    """
    blur = ImageBlur(path_to_original_image)
    blur.apply_gaussian_blur(kernel_size=(15, 15), sigma_x=0)
    blur.save_blurred_image('./images/blured_image.jpg')
    blur.show_blurred_image()
    blured_image_path = "./images/blured_image.jpg"
"""
    # Testing ImageEnhancement class
    #enhance = ImageEnhancement(path_to_original_image)

    # Gamma Correction
    """
    enhance.gamma_correction(gamma=2.5)
    enhance.save_image('./images/gamma_correction_image.jpg')
    enhance.show_image(window_name="Gamma Correction")
"""

    # 2d-filtering

    #enhance.filter2D(EDGE_DETECTION_KERNEL)
    #enhance.save_image('./images/2dfilter_with_EDGE_kernel_image.jpg')
    #enhance.show_image(window_name="filter2d with EDGE_DETECTION kernel")


    # Edge Detection Method
    """
    enhance.edge_detection_filter()
    enhance.save_image('./images/edge_detection_image.jpg')
    enhance.show_image(window_name="EDGE_DETECTION kernel")

    """
    # Sharpen Method
    """
    enhance.sharpen_filter()
    enhance.save_image('./images/sharpen_image.jpg')
    enhance.show_image(window_name="SHARPEN kernel")
    """
    # Identity Filter
    """
    enhance.identity_filter()
    enhance.save_image('./images/identity_image.jpg')
    enhance.show_image(window_name="IDENTITY kernel")
    """
    # Box Blur Filter
    """
    enhance.box_blur_filter()
    enhance.save_image('./images/box_blur_image.jpg')
    enhance.show_image(window_name="BOX_BLUR kernel")
    """
    # Gaussian Blur
    """
    enhance.gaussian_blur_filter()
    enhance.save_image('./images/gaussian_blur_image.jpg')
    enhance.show_image(window_name="GAUSSIAN_BLUR kernel")
    """

    # Zero Kernel Blurring
    """
    enhance.zero_kernel_blurring_filter()
    enhance.save_image('./images/zero_kernel_image.jpg')
    enhance.show_image(window_name="ZERO_KERNEL kernel")
    """
    # Median Blurring
    """
    enhance.median_filter()
    enhance.save_image('./images/median_blurring_image.jpg')
    enhance.show_image(window_name="Median Blurring")
    """
    # Wavelet Transformation
    """
    the_image = "./images/Samoyed-dog.webp"

    wavelet_transform = WaveletTransform(the_image)
    wavelet_transform.perform_wavelet_transform()
    wavelet_transform.show_original_image()
    wavelet_transform.show_wavelet_transform()
    wavelet_transform.save_wavelet_transform_figure("./images/wavelet_transform.jpg")
"""
    # Fourier Transformation
    fourier = FourierTransform("./images/Samoyed-dog.webp")
    fourier.set_fft()
    fourier.set_magnitude_spectrum()
    fourier.show_frequency_spectrum()
    fourier.show_spectrum_plt()

    #print(fourier.magnitude_spectrum.shape)

    """
    fourier = FourierTransform("./images/Samoyed-bw.jpg")
    fourier = FourierTransform("./images/Samoyed-bw.jpg")
    fourier.set_fft()
    fourier.set_magnitude_spectrum()
    fourier.show_frequency_spectrum()
    fourier.show_spectrum_plt()
"""

"""
    original_file = "./images/Samoyed-dog.webp"

    wavelet_transform = WaveletTransform(original_file)
    wavelet_transform.perform_wavelet_transform()
    wavelet_transform.show_wavelet_transform()
    wavelet_transform.save_wavelet_transform_figure("./images/wavelet_transform.jpg")

"""