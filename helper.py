from consts import MIN_BRIGHTNESS_VALUE, MAX_BRIGHTNESS_VALUE, BORDER_PROCESSING_PARAMETER, RIGHT_BRIGHTNESS_VALUE, \
    LEFT_BRIGHTNESS_VALUE


def correct_limits_function(element_value):
    if element_value < MIN_BRIGHTNESS_VALUE:
        return MIN_BRIGHTNESS_VALUE
    if element_value > MAX_BRIGHTNESS_VALUE:
        return MAX_BRIGHTNESS_VALUE
    return element_value


def impulse_noise_function(element_value):
    if element_value == -1:
        return MIN_BRIGHTNESS_VALUE
    if element_value == 1:
        return MAX_BRIGHTNESS_VALUE
    return 128


def border_processing_function(element_value):
    if element_value >= BORDER_PROCESSING_PARAMETER:
        return RIGHT_BRIGHTNESS_VALUE
    else:
        return LEFT_BRIGHTNESS_VALUE


def swap_zero_and_ones(element_value):
    if element_value == 0:
        return 1
    if element_value == 1:
        return 0
    return element_value


def fill_salt_and_pepper_help(element_value):
    if element_value == 0:
        return MAX_BRIGHTNESS_VALUE
    if element_value < 0:
        return MIN_BRIGHTNESS_VALUE
    return element_value
