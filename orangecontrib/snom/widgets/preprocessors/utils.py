import numpy as np

from Orange.preprocess import Preprocess


def reshape_to_image(data, x, y):
    xres = np.size(np.unique(x))
    yres = np.size(np.unique(y))

    return np.reshape(data, (yres, xres))


class PreprocessImageOpts(Preprocess):
    pass
