import numpy as np


def reshape_to_image(data, x, y):
    xres = np.size(np.unique(x))
    yres = np.size(np.unique(y))

    return np.reshape(data, (yres, xres))
