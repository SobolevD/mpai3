import numpy as np

MEDIAN_FILTER_MASK_1 = np.array([[0, 1, 0],
                                 [1, 3, 1],
                                 [0, 1, 0]])

MEDIAN_FILTER_MASK_2 = np.array([[1, 1, 1],
                                 [1, 3, 1],
                                 [1, 1, 1]])

MEDIAN_FILTER_MASK_3 = np.array([[0, 1, 0],
                                 [1, 1, 1],
                                 [0, 1, 0]])


MEDIAN_FILTER_MASK_4 = np.array([[1, 1, 1],
                                 [1, 1, 1],
                                 [1, 1, 1]])

LINEAR_FILTER_MASK_A = np.array([[1, 1, 1],
                                 [1, 1, 1],
                                 [1, 1, 1]])*(1/9)

LINEAR_FILTER_MASK_B = np.array([[1, 1, 1],
                                 [1, 2, 1],
                                 [1, 1, 1]])*(1/10)

LINEAR_FILTER_MASK_C = np.array([[1, 2, 1],
                                 [2, 4, 2],
                                 [1, 2, 1]])*(1/16)