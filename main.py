import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.ndimage import median_filter
from skimage.io import imshow, show
from skimage.util import random_noise

from border_processing import border_processing
from chess_desk_creator import create_chess_field_image
from consts import IMAGE_HEIGHT, \
    IMAGE_LENGTH
from filter_masks import MEDIAN_FILTER_MASK_1, LINEAR_FILTER_MASK_A
from helper import correct_limits_function, border_processing_function, impulse_noise_function, swap_zero_and_ones, \
    fill_salt_and_pepper_help
from histogram import create_canvas_white_noise, show_canvas_impulse


def window_processing(matrix, window):
    return signal.convolve2d(matrix, window, boundary='symm', mode='same').astype(int)


def coefficient_of_decreasing_noise(src_img, img_and_noise, filtered_img):
    divided = middle_square_error_pow_2(src_img, filtered_img)
    delimiter = np.mean(np.square(img_and_noise - src_img))
    return (divided/delimiter).astype(float)


def middle_square_error_pow_2(src_img, filtered_img):
    return np.mean(np.square(filtered_img - src_img))


def fill_salt_and_pepper(noise_matrix, src_img):
    noise_matrix_copy = noise_matrix
    shape = np.shape(noise_matrix_copy)
    new_noise_list = list(map(swap_zero_and_ones, np.reshape(noise_matrix_copy, noise_matrix_copy.size)))
    single_dimension_array = np.array(new_noise_list)
    noise_matrix_copy = np.multiply(np.reshape(single_dimension_array, (shape[0], shape[1])), src_img)
    new_img_list = list(map(fill_salt_and_pepper_help, np.reshape(noise_matrix_copy, noise_matrix_copy.size)))
    single_dimension_array = np.array(new_img_list)
    new_img = np.reshape(single_dimension_array, (shape[0], shape[1]))
    return new_img


chess_img = create_chess_field_image()
print(chess_img)
chess_board_img = border_processing(chess_img, border_processing_function)


img_dispersion = np.var(chess_board_img)
white_noise_10 = np.random.normal(loc=0, scale=float(np.sqrt(float(img_dispersion / 10))), size=(IMAGE_HEIGHT, IMAGE_LENGTH))
print(white_noise_10)

dispersion_noise_10 = np.var(white_noise_10)

white_noise_matrix_10 = white_noise_10.astype(int)

# Correct limits
img_with_white_noise_10 = border_processing(chess_board_img + white_noise_matrix_10, correct_limits_function)

median_filter_img_10_1 = median_filter(img_with_white_noise_10, footprint=MEDIAN_FILTER_MASK_1)
#median_filter_img_10_2 = median_filter(img_with_white_noise_10, footprint=MEDIAN_FILTER_MASK_2)
#median_filter_img_10_3 = median_filter(img_with_white_noise_10, footprint=MEDIAN_FILTER_MASK_3)
#median_filter_img_10_4 = median_filter(img_with_white_noise_10, footprint=MEDIAN_FILTER_MASK_4)
linear_filter_img_10_A = window_processing(img_with_white_noise_10, LINEAR_FILTER_MASK_A)
#linear_filter_img_10_B = window_processing(img_with_white_noise_10, LINEAR_FILTER_MASK_B)
#linear_filter_img_10_C = window_processing(img_with_white_noise_10, LINEAR_FILTER_MASK_C)


print("MSE linear filter 10")
print(middle_square_error_pow_2(chess_board_img, linear_filter_img_10_A))

print("MSE median filter 10")
print(middle_square_error_pow_2(chess_board_img, median_filter_img_10_1))

# ------------


white_noise_1 = np.random.normal(loc=0, scale=float(np.sqrt(float(img_dispersion))), size=(IMAGE_HEIGHT, IMAGE_LENGTH))
print("min of noise 1")
print(np.min(white_noise_1))
print("max of noise 1")
print(np.max(white_noise_1))


white_noise_matrix_1 = white_noise_1.astype(int)
img_with_white_noise_1 = border_processing(chess_board_img + white_noise_matrix_1, correct_limits_function)

median_filter_img_1 = median_filter(img_with_white_noise_1, footprint=MEDIAN_FILTER_MASK_1)
linear_filter_img_1 = window_processing(img_with_white_noise_1, LINEAR_FILTER_MASK_A)


fig = plt.figure(figsize=(20, 10))
fig.add_subplot(1, 1, 1)
plt.title("Source image")
imshow(border_processing(chess_img), cmap='gray', vmin=0, vmax=255)
show()

create_canvas_white_noise(img_with_white_noise_1, img_with_white_noise_10, np.abs(white_noise_matrix_1), np.abs(white_noise_matrix_10), linear_filter_img_1, linear_filter_img_10_A)
show()

create_canvas_white_noise(img_with_white_noise_1, img_with_white_noise_10, np.abs(white_noise_matrix_1), np.abs(white_noise_matrix_10), median_filter_img_1, median_filter_img_10_1)
show()

array_to_generate = np.zeros((128, 128), dtype=int)
array_to_generate[0][0] = -1

impulse_noise_01 = random_noise(array_to_generate, mode="s&p", amount=0.1)
img_with_impulse_noise_intense_01 = fill_salt_and_pepper(impulse_noise_01, chess_board_img)


median_filter_img_imp_noise_01 = median_filter(img_with_impulse_noise_intense_01, footprint=MEDIAN_FILTER_MASK_1)
# Размножить
linear_filter_img_imp_noise_01 = window_processing(img_with_impulse_noise_intense_01, LINEAR_FILTER_MASK_A)

impulse_noise_03 = random_noise(array_to_generate, mode="s&p", amount=0.3)
img_with_impulse_noise_intense_03 = fill_salt_and_pepper(impulse_noise_03, chess_board_img)


median_filter_img_imp_noise_03 = median_filter(img_with_impulse_noise_intense_03, footprint=MEDIAN_FILTER_MASK_1)
linear_filter_img_imp_noise_03 = window_processing(img_with_impulse_noise_intense_03, LINEAR_FILTER_MASK_A)

# show_impulse_noise -> border processing
show_canvas_impulse(border_processing(impulse_noise_01, impulse_noise_function), img_with_impulse_noise_intense_01, linear_filter_img_imp_noise_01, median_filter_img_imp_noise_01)
show()
show_canvas_impulse(border_processing(impulse_noise_03, impulse_noise_function), img_with_impulse_noise_intense_03, linear_filter_img_imp_noise_03, median_filter_img_imp_noise_03)
show()


print("=============================================================================================================")
print("=============================================================================================================")
print("=============================================================================================================")

print("Dispersion of images and noises")
print("=============================================================================================================")

print("dispersion of image")
print(img_dispersion)

print("dispersion of noise 10")
print(dispersion_noise_10)

print("dispersion of image with noise 10")
print(np.var(img_with_white_noise_10))


print("dispersion of noise 1")
print(np.var(white_noise_1))

print("dispersion of image with noise 1")
print(np.var(img_with_white_noise_1))

print("dispersion of noise 01")
print(np.var(impulse_noise_01))

print("dispersion of image with noise 01")
print(np.var(img_with_impulse_noise_intense_01))

print("dispersion of noise 03")
print(np.var(impulse_noise_03))

print("dispersion of image with noise 03")
print(np.var(img_with_impulse_noise_intense_03))

print("=============================================================================================================")


print("Dispersion filtration errors")
print("=============================================================================================================")


print("dispersion of linear filtered image with noise 10")
print(middle_square_error_pow_2(chess_board_img, linear_filter_img_10_A))

print("dispersion of median filtered image with noise 10")
print(middle_square_error_pow_2(chess_board_img, median_filter_img_10_1))

print("dispersion of linear filtered image with noise 1")
print(middle_square_error_pow_2(chess_board_img, linear_filter_img_1))

print("dispersion of median filtered image with noise 1")
print(middle_square_error_pow_2(chess_board_img, median_filter_img_1))

print("dispersion of linear filtered image with noise 01")
print(middle_square_error_pow_2(chess_board_img, linear_filter_img_imp_noise_01))

print("dispersion of median filtered image with noise 01")
print(middle_square_error_pow_2(chess_board_img, median_filter_img_imp_noise_01))

print("dispersion of linear filtered image with noise 03")
print(middle_square_error_pow_2(chess_board_img, linear_filter_img_imp_noise_03))

print("dispersion of median filtered image with noise 03")
print(middle_square_error_pow_2(chess_board_img, median_filter_img_imp_noise_03))


print("=============================================================================================================")
print("Suppress noise coefficients")
print("=============================================================================================================")


print("Coefficient of decreasing noise linear filter 10")
print(coefficient_of_decreasing_noise(chess_board_img, img_with_white_noise_10, linear_filter_img_10_A))

print("Coefficient of decreasing noise median filter 10")
print(coefficient_of_decreasing_noise(chess_board_img, img_with_white_noise_10, median_filter_img_10_1))

print("Coefficient of decreasing noise linear filter 1")
print(coefficient_of_decreasing_noise(chess_board_img, img_with_white_noise_1, linear_filter_img_1))

print("Coefficient of decreasing noise median filter 1")
print(coefficient_of_decreasing_noise(chess_board_img, img_with_white_noise_1, median_filter_img_1))

print("Coefficient of decreasing noise linear filter 01")
print(coefficient_of_decreasing_noise(chess_board_img, img_with_impulse_noise_intense_01, linear_filter_img_imp_noise_01))

print("Coefficient of decreasing noise median filter 01")
print(coefficient_of_decreasing_noise(chess_board_img, img_with_impulse_noise_intense_01, median_filter_img_imp_noise_01))

print("Coefficient of decreasing noise linear filter 03")
print(coefficient_of_decreasing_noise(chess_board_img, img_with_impulse_noise_intense_03, linear_filter_img_imp_noise_03))

print("Coefficient of decreasing noise median filter 03")
print(coefficient_of_decreasing_noise(chess_board_img, img_with_impulse_noise_intense_03, median_filter_img_imp_noise_03))


print("=============================================================================================================")
print("Sobolev")

