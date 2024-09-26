import math
from typing import Optional

import numpy as np
from Orange.data import Table, Domain
from Orange.data.util import SharedComputeValue


def reshape_to_image(data,x,y):
    xres = np.size(np.unique(x))
    yres = np.size(np.unique(y))

    return np.reshape(data,(yres,xres))