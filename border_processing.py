import numpy as np


def border_processing(img_as_arrays, processing_function):
    shape = np.shape(img_as_arrays)
    new_img_list = list(map(processing_function, np.reshape(img_as_arrays, img_as_arrays.size)))
    single_dimension_array = np.array(new_img_list)
    new_img = np.reshape(single_dimension_array, (shape[0], shape[1]))
    return new_img
