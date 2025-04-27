import numpy as np

import Orange
import Orange.data
from Orange.preprocess.preprocess import Preprocess

from orangecontrib.spectroscopy.preprocess.utils import (
    SelectColumn,
    CommonDomain,
)


class _PhaseUnwrapCommon(CommonDomain):
    # A possible optimization in the future:
    # the "order" functionality of CommonDomainOrderUnknowns is not needed,
    # but does not break anything either
    def __init__(self, domain):
        super().__init__(domain)

    def transformed(self, data):  # noqa: N803
        return np.unwrap(data.X)


class PhaseUnwrap(Preprocess):
    """
    Unwrap phase values using numpy.unwrap defaults or bypass the data.
    """

    def __call__(self, data):
        common = _PhaseUnwrapCommon(data.domain)
        atts = [
            a.copy(compute_value=SelectColumn(i, common))
            for i, a in enumerate(data.domain.attributes)
        ]
        domain = Orange.data.Domain(atts, data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)
