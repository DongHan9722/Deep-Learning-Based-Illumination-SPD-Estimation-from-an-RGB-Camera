## resources: https://github.com/psychopy/versions/blob/master/psychopy/tools/colorspacetools.py

import numpy as np
def unpackColors(colors):  # used internally, not exported by __all__
    """Reshape an array of color values to Nx3 format.
    Many color conversion routines operate on color data in Nx3 format, where
    rows are color space coordinates. 1x3 and NxNx3 input are converted to Nx3
    format. The original shape and dimensions are also returned, allowing the
    color values to be returned to their original format using 'reshape'.
    Parameters
    ----------
    colors : ndarray, list or tuple of floats
        Nx3 or NxNx3 array of colors, last dim must be size == 3 specifying each
        color coordinate.
    Returns
    -------
    tuple
        Nx3 ndarray of converted colors, original shape, original dims.
    """
    # handle the various data types and shapes we might get as input
    colors = np.asarray(colors, dtype=float)

    orig_shape = colors.shape
    orig_dim = colors.ndim
    if orig_dim == 1 and orig_shape[0] == 3:
        colors = np.array(colors, ndmin=2)
    elif orig_dim == 2 and orig_shape[1] == 3:
        pass  # NOP, already in correct format
    elif orig_dim == 3 and orig_shape[2] == 3:
        colors = np.reshape(colors, (-1, 3))
    else:
        raise ValueError(
            "Invalid input dimensions or shape for input colors.")

    return colors, orig_shape, orig_dim

def srgbTF(rgb, reverse=False, **kwargs):
    """Apply sRGB transfer function (or gamma) to linear RGB values.
    Input values must have been transformed using a conversion matrix derived
    from sRGB primaries relative to D65.
    Parameters
    ----------
    rgb : tuple, list or ndarray of floats
        Nx3 or NxNx3 array of linear RGB values, last dim must be size == 3
        specifying RBG values.
    reverse : boolean
        If True, the reverse transfer function will convert sRGB -> linear RGB.
    Returns
    -------
    ndarray
        Array of transformed colors with same shape as input.
    """
    rgb, orig_shape, orig_dim = unpackColors(rgb)

    # apply the sRGB TF
    if not reverse:
        # applies the sRGB transfer function (linear RGB -> sRGB)
        to_return = np.where(
            rgb <= 0.0031308,
            rgb * 12.92,
            (1.0 + 0.055) * rgb ** (1.0 / 2.4) - 0.055)
    else:
        # do the inverse (sRGB -> linear RGB)
        to_return = np.where(
            rgb <= 0.04045,
            rgb / 12.92,
            ((rgb + 0.055) / 1.055) ** 2.4)

    if orig_dim == 1:
        to_return = to_return[0]
    elif orig_dim == 3:
        to_return = np.reshape(to_return, orig_shape)

    return to_return

    