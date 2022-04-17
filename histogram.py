from matplotlib import pyplot as plt
from skimage.io import imshow


def create_canvas_white_noise(image1, image2, image3, image4, image5, image6):
    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(2, 3, 1)
    plt.title("Noised image 1")
    imshow(image1, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 3, 2)
    plt.title("Noised image relation 1")
    imshow(image3, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 3, 3)
    plt.title("Filtered image 1")
    imshow(image5, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 3, 4)
    plt.title("Noised image 2")
    imshow(image2, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 3, 5)
    plt.title("Noised image relation 2")
    imshow(image4, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 3, 6)
    plt.title("Filtered image 2")
    imshow(image6, cmap='gray', vmin=0, vmax=255)
    return fig


def show_canvas_impulse(imp_noise, img_imp_noise, linear_filtered_img, median_filtered_img):
    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(2, 2, 1)
    plt.title("Image of noise")
    imshow(imp_noise, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 2, 2)
    plt.title("Source image with noise")
    imshow(img_imp_noise, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 2, 3)
    plt.title("Noised image median_filter")
    imshow(median_filtered_img, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 2, 4)
    plt.title("Noised image linear_filter")
    imshow(linear_filtered_img, cmap='gray', vmin=0, vmax=255)
    return fig
