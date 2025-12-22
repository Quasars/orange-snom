import numpy as np
from enum import Enum

from Orange.data import Domain
from Orange.preprocess import Preprocess

from orangecontrib.spectroscopy.preprocess import (
    CommonDomain,
    SelectColumn,
    WrongReferenceException,
    MissingReferenceException,
)
from orangecontrib.spectroscopy.utils import (
    InvalidAxisException,
    values_to_linspace,
    index_values,
)

from pySNOM.images import mask_from_datacondition


class PreprocessImageOpts(Preprocess):
    pass


class PreprocessImageOpts2D(PreprocessImageOpts):
    def __call__(self, data, image_opts, run_all=False):
        if run_all:
            raise Exception("run_all not supported yet")
        common = self.image_transformer(data, image_opts)
        at = data.domain[image_opts["attr_value"]].copy(
            compute_value=SelectColumn(0, common)
        )
        domain = domain_with_single_attribute_in_x(at, data.domain)
        return data.transform(domain)

    def image_transformer(self, data, image_opts):
        raise NotImplementedError


class NoComputeValue:
    def __call__(self, data):
        return np.full(len(data), np.nan)


def _prepare_domain_for_image(data, image_opts):
    at = data.domain[image_opts["attr_value"]].copy(compute_value=NoComputeValue())
    return domain_with_single_attribute_in_x(at, data.domain)


def _prepare_table_for_image(data, image_opts):
    odata = data
    domain = _prepare_domain_for_image(data, image_opts)
    data = data.transform(domain)
    if len(data):
        with data.unlocked(data.X):
            data.X[:, 0] = odata.get_column(image_opts["attr_value"], copy=True)
    return data


def _image_from_table(data, image_opts):
    hypercube, _, indices = get_ndim_hyperspec(
        data, (image_opts["attr_y"], image_opts["attr_x"])
    )
    return hypercube[:, :, 0], indices


class PreprocessImageOpts2DOnlyWhole(PreprocessImageOpts):
    def __call__(self, data, image_opts, run_all=False):
        if run_all or len(data.domain.attributes) == 0:
            attrs_to_run = [v.name for v in data.domain.attributes]
            newdata = data.copy()
        else:
            # This is only for the preview for a single feature image
            # So it only processes one image
            attrs_to_run = [image_opts["attr_value"]]
            newdata = _prepare_table_for_image(data, image_opts)

        mask = get_mask_from_image_opts(data, image_opts)
        image_opts = image_opts.copy()  # otherwise this input will be changed
        new_vals = np.full_like(newdata.X, np.nan)
        for i, attr in enumerate(attrs_to_run):
            image_opts["attr_value"] = attr
            try:
                temp = _prepare_table_for_image(newdata, image_opts)
            except KeyError:
                raise WrongReferenceException(
                    "Data and reference do not contain the same features"
                )

            try:
                image, indices = _image_from_table(temp, image_opts)
                transformed = self.transform_image(image, newdata, mask=mask)
                new_vals[:, i] = transformed[indices].reshape(-1)
            except InvalidAxisException:
                new_vals[:, i] = np.full(len(newdata), np.nan)

        with newdata.unlocked(newdata.X):
            newdata.X = new_vals

        return newdata

    def transform_image(self, image, data, mask=None):
        """
        image: a numpy 2D array where image[y,x] is the value in image row y and column x
        data: image data set (used for passing meta data)
        """
        raise NotImplementedError


class PreprocessImageOpts2DOnlyWholeReference(PreprocessImageOpts):
    def __init__(self, reference):
        self.reference = reference
        if self.reference is None:
            raise MissingReferenceException("Preprocessor needs a reference.")

    def __call__(self, data, image_opts, run_all=False):
        if run_all or len(data.domain.attributes) == 0:
            attrs_to_run = [v.name for v in data.domain.attributes]
            newdata = data.copy()
        else:
            # This is only for the preview for a single feature image
            # So it only processes one image
            attrs_to_run = [image_opts["attr_value"]]
            newdata = _prepare_table_for_image(data, image_opts)

        ref_attrs = [v.name for v in self.reference.domain.attributes]

        if set(attrs_to_run) != set(ref_attrs) or len(ref_attrs) != 1:
            WrongReferenceException(
                "Reference has to contain the same features or be single-featured"
            )

        image_opts = image_opts.copy()  # because it will be modified
        image_opts_ref = image_opts

        if len(ref_attrs) == 1:  # use the only reference regardless of the name
            image_opts_ref = image_opts.copy()  # unlink to image_opts
            image_opts_ref["attr_value"] = ref_attrs[0]

        new_vals = np.full_like(newdata.X, np.nan)
        for i, attr in enumerate(attrs_to_run):
            image_opts["attr_value"] = attr

            try:
                temp = _prepare_table_for_image(newdata, image_opts)
                reference = _prepare_table_for_image(self.reference, image_opts_ref)
            except KeyError:
                raise WrongReferenceException(
                    "Data and reference do not contain the same features"
                )

            try:
                image, indices = _image_from_table(temp, image_opts)
                ref_image, _ = _image_from_table(reference, image_opts_ref)
                transformed = self.transform_image(image, ref_image, temp)
                new_vals[:, i] = transformed[indices].reshape(-1)
            except InvalidAxisException:
                new_vals[:, i] = np.full(len(newdata), np.nan)

        with newdata.unlocked(newdata.X):
            newdata.X = new_vals

        return newdata

    def transform_image(self, image, ref_image, data):
        """
        image: a numpy 2D array where image[y,x] is the value in image row y and column x
        ref_image: a numpy 2D array where image[y,x] is the value in image row y and column x
        data: image data set (used for passing meta data)
        """
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


def table_with_no_attribute(data):
    """Create a domain with only the attribute in domain.attributes and ensure
    that the same attribute is removed from metas and class_vars if it was present
    there."""

    newdomain = Domain([], data.domain.class_vars, data.domain.metas)

    data = data.transform(newdomain)
    return data


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
                data, (self.image_opts["attr_y"], self.image_opts["attr_x"])
            )
            image = hypercube[:, :, 0]
            transformed = self.transform_image(image)
            return transformed[indices].reshape(-1, 1)
        except InvalidAxisException:
            return np.full((len(data), 1), np.nan)
        return self.transformed(data)

    def transform_image(self, image):
        """
        image: a numpy 2D array where image[y,x] is the value in image row y and column x
        """
        raise NotImplementedError


def get_mask_from_image_opts(data, image_opts):
    if image_opts.get("attr_mask", None) is None:
        return None
    image_opts = image_opts.copy()
    image_opts["attr_value"] = image_opts["attr_mask"]
    try:
        masktable = _prepare_table_for_image(data, image_opts)
    except KeyError:
        return None
    maskimage, _ = _image_from_table(masktable, image_opts)
    mask = mask_from_datacondition(maskimage == image_opts["value_mask"])
    return mask


class MaskOptions(Enum):
    ignore = 0
    exclude = 1
    include = 2


def transform_mask(mask, option):
    if option == MaskOptions.ignore:
        return None
    elif option == MaskOptions.include:
        return mask_from_datacondition(mask == 1.0)
    elif option == MaskOptions.exclude:
        return mask
    else:
        raise NotImplementedError
