import numpy as np

from consts import IMAGE_LENGTH, IMAGE_HEIGHT, CELL_HEIGHT


def create_chess_field_image():
    img = np.ones((IMAGE_LENGTH, IMAGE_HEIGHT)).astype(int)
    line_index = 0
    indexes_of_start_black_odd = start_of_black_sells_odd()
    indexes_of_start_black_even = start_of_black_sells_even()
    odd_row = False

    while line_index < IMAGE_HEIGHT:

        column_index = 0

        if odd_row:
            current_indexes = indexes_of_start_black_odd
        else:
            current_indexes = indexes_of_start_black_even

        while column_index < IMAGE_LENGTH/(2*CELL_HEIGHT):
            j = 0
            while j < CELL_HEIGHT:
                img[line_index][current_indexes[column_index] + j] = 0
                j = j + 1
            column_index = column_index + 1
        line_index = line_index + 1
        if line_index % CELL_HEIGHT == 0:
            odd_row = not odd_row
    return img


def start_of_black_sells_odd():
    result = []
    i = 0
    j = 0
    while 2*i < IMAGE_HEIGHT:
        result.insert(j, 2*i)
        i = i + CELL_HEIGHT
        j = j + 1
    return result


def start_of_black_sells_even():
    result = []
    i = CELL_HEIGHT
    j = 0
    while i < IMAGE_HEIGHT:
        result.insert(j, i)
        i = i + 2*CELL_HEIGHT
        j = j + 1
    return result
