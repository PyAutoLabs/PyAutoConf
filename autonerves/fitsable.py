from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    try:
        from astropy.io import fits
    except ImportError:
        pass

try:
    from astropy.io import fits
except ImportError:
    pass

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union, List


def hdu_list_for_output_from(
    values_list: List[np.ndarray],
    header_dict: Optional[dict] = None,
    ext_name_list: Optional[List[str]] = None,
) -> fits.HDUList:
    """
    Returns the HDU list which can be used to output arrays to a .fits file.

    The output .fits files may contain multiple HDUs comprising different images. Conventionally, the first array,
    the `PrimaryHDU`, contains the 2D mask applied to the data and the remaining HDUs contain the data itself.
    The mask is used to add information to the header, for example the pixel scale of the data.

    Each HDU contains its `ext_name` in the header, which is visible when the .fits file is loaded in DS9.

    Parameters
    ----------
    values
        The 1D or 2D array that is written to fits.
    header_dict
        A dictionary of values that are written to the header of the .fits file.
    ext_name_list
        The names of the extension in the fits file, which displays in the header of the fits file and is visible
        for example when the .fits is loaded in DS9.

    Returns
    -------
    The HDU list containing the data and its header which can then be written to .fits.

    Examples
    --------
    data = np.ones((5,5))
    noise_map = np.ones((5,5))

    hdu_list_for_output_from(
        values_list=[data, noise_map]
        header_dict={"Example": 0.5},
        ext_name_list=["data", "noise_map"]
    )
    """
    hdu_list = []
    
    header = fits.Header()

    if header_dict is not None:
        for key, value in header_dict.items():
            # Convert enum to its string value if needed
            key_str = key.value if isinstance(key, Enum) else key
            try:
                header.append((key_str, value, [""]))
            except ValueError:
                header.append((key_str, float(value), [""]))

    for i, values in enumerate(values_list):
    
        if ext_name_list is not None:
            header["EXTNAME"] = ext_name_list[i].upper()

        # Convert from JAX
        try:
            values = np.array(values.array)
        except AttributeError:
            values = np.array(values)

        if i == 0:
            hdu_list.append(fits.PrimaryHDU(values, header=header))
        else:
            hdu_list.append(fits.ImageHDU(values, header=header))
         
    return fits.HDUList(hdus=hdu_list)

def output_to_fits(
    values: np.ndarray,
    file_path: Union[Path, str],
    overwrite: bool = False,
    header_dict: Optional[dict] = None,
    ext_name: Optional[str] = None,
):
    """
    Write a NumPy array to a .fits file.

    Parameters
    ----------
    values
        The numpy array of values that is written to fits.
    file_path
        The full path of the file that is output, including the file name and ``.fits`` extension.
    overwrite
        If `True` and a file already exists with the input file_path the .fits file is overwritten. If `False`, an
        error is raised.
    header_dict
        A dictionary of values that are written to the header of the .fits file.
    ext_name
        The name of the extension in the fits file, which displays in the header of the fits file and is visible.

    Examples
    --------
    values = np.ones((5,5))
    numpy_array_to_fits(values=values, file_path='/path/to/file/filename.fits', overwrite=True)
    """

    hdu = hdu_list_for_output_from(
        values_list=[values],
        header_dict=header_dict,
        ext_name_list=[ext_name] if ext_name is not None else None,
    )

    write_hdu_list(hdu, file_path=file_path, overwrite=overwrite)

def write_hdu_list(hdu_list, file_path, overwrite=False):
    """Write an ``HDUList`` to a FITS file, creating directories as needed.

    Parameters
    ----------
    hdu_list
        The ``astropy.io.fits.HDUList`` to write.
    file_path : str or Path
        Full output path including ``.fits`` extension.
    overwrite : bool
        If ``True`` an existing file is replaced.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and file_path.is_file():
        file_path.unlink()
    hdu_list.writeto(file_path)


def ndarray_via_hdu_from(hdu):
    """
    Returns an ``Array2D`` by from a `PrimaryHDU` object which has been loaded via `astropy.fits`

    This assumes that the `header` of the `PrimaryHDU` contains an entry named `PIXSCALE` which gives the
    pixel-scale of the array.

    For a full description of ``Array2D`` objects, including a description of the ``slim`` and ``native`` attribute
    used by the API, see
    the :meth:`Array2D class API documentation <autoarray.structures.arrays.uniform_2d.AbstractArray2D.__new__>`.

    Parameters
    ----------
    primary_hdu
        The `PrimaryHDU` object which has already been loaded from a .fits file via `astropy.fits` and contains
        the array data and the pixel-scale in the header with an entry named `PIXSCALE`.
    origin
        The (y,x) scaled units origin of the coordinate system.

    Examples
    --------

    .. code-block:: python

        from astropy.io import fits
        import autoarray as aa

        primary_hdu = fits.open("path/to/file.fits")

        array_2d = aa.Array2D.from_primary_hdu(
            primary_hdu=primary_hdu,
        )
    """
    values = hdu.data.astype("float")
    return values


def ndarray_via_fits_from(
    file_path: Union[Path, str], hdu: int, do_not_scale_image_data: bool = False
):
    """
    Read a 2D NumPy array from a .fits file.

    Parameters
    ----------
    file_path
        The full path of the file that is loaded, including the file name and ``.fits`` extension.
    hdu
        The HDU extension of the array that is loaded from the .fits file.
    do_not_scale_image_data
        If True, the .fits file is not rescaled automatically based on the .fits header info.

    Returns
    -------
    ndarray
        The NumPy array that is loaded from the .fits file.

    Examples
    --------
    array_2d = ndarray_via_fits_from(file_path='/path/to/file/filename.fits', hdu=0)
    """
    hdu_list = fits.open(file_path, do_not_scale_image_data=do_not_scale_image_data)
    return ndarray_via_hdu_from(hdu_list[hdu])


def header_obj_from(file_path: Union[Path, str], hdu: int) -> Dict:
    """
    Read a 2D NumPy array from a .fits file.

    Parameters
    ----------
    file_path
        The full path of the file that is loaded, including the file name and ``.fits`` extension.
    hdu
        The HDU extension of the array that is loaded from the .fits file.
    do_not_scale_image_data
        If True, the .fits file is not rescaled automatically based on the .fits header info.

    Returns
    -------
    dict
        The header dictionary.

    Examples
    --------
    array_2d = ndarray_via_fits_from(file_path='/path/to/file/filename.fits', hdu=0)
    """
    hdu_list = fits.open(file_path)
    return hdu_list[hdu].header





