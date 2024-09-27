import numpy as np

from Orange.data import Domain
from Orange.preprocess import Preprocess
from orangecontrib.spectroscopy.preprocess import CommonDomain, SelectColumn
from orangecontrib.spectroscopy.utils import (
    InvalidAxisException,
    values_to_linspace,
    index_values,
)


class PreprocessImageOpts(Preprocess):
    pass


class PreprocessImageOpts2D(PreprocessImageOpts):
    def __call__(self, data, image_opts):
        common = self.image_transformer(data, image_opts)
        at = data.domain[image_opts["attr_value"]].copy(
            compute_value=SelectColumn(0, common)
        )
        domain = domain_with_single_attribute_in_x(at, data.domain)
        return data.transform(domain)

    def image_transformer(self, data, image_opts):
        raise NotImplementedError


def axes_to_ndim_linspace(coordinates):
    # modified to avoid domains as much as possible
    ls = []
    indices = []

    for i in range(coordinates.shape[1]):
        coor = coordinates[:, i]
        lsa = values_to_linspace(coor)
        if lsa is None:
            raise InvalidAxisException(i)
        ls.append(lsa)
        indices.append(index_values(coor, lsa))

    return ls, tuple(indices)


def get_ndim_hyperspec(data, attrs):
    # mostly copied from orangecontrib.spectroscopy.utils,
    # but returns the indices too
    # also avoid Orange domains as much as possible
    coordinates = np.hstack([data.get_column(a).reshape(-1, 1) for a in attrs])

    ls, indices = axes_to_ndim_linspace(coordinates)

    # set data
    new_shape = tuple([lsa[2] for lsa in ls]) + (data.X.shape[1],)
    hyperspec = np.ones(new_shape) * np.nan

    hyperspec[indices] = data.X

    return hyperspec, ls, indices


def domain_with_single_attribute_in_x(attribute, domain):
    """Create a domain with only the attribute in domain.attributes and ensure
    that the same attribute is removed from metas and class_vars if it was present
    there."""
    class_vars = [a for a in domain.class_vars if a.name != attribute.name]
    metas = [a for a in domain.metas if a.name != attribute.name]
    return Domain([attribute], class_vars, metas)


class CommonDomainImage2D(CommonDomain):
    def __init__(self, domain: Domain, image_opts: dict):
        self.domain = domain
        self.image_opts = image_opts

    def __call__(self, data):
        data = self.transform_domain(data)
        vat = data.domain[self.image_opts["attr_value"]]
        ndom = domain_with_single_attribute_in_x(vat, data.domain)
        data = data.transform(ndom)
        try:
            hypercube, _, indices = get_ndim_hyperspec(
                data, (self.image_opts["attr_x"], self.image_opts["attr_y"])
            )
            image = hypercube[:, :, 0]
            transformed = self.transform_image(image)
            return transformed[indices].reshape(-1, 1)
        except InvalidAxisException:
            return np.full((len(data), 1), np.nan)
        return self.transformed(data)

    def transform_image(self, image):
        raise NotImplementedError
