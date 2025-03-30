Preprocessing Module API
======================

Module: ``ulmo.preproc.utils``
----------------------------

The ``utils`` module provides utilities for preprocessing oceanographic data.

Functions
--------

.. py:function:: build_mask(field, qual, qual_thresh=2, temp_bounds=(-2, 33), field='SST')

   Generate a binary mask for a field based on quality flags and temperature bounds.
   
   :param field: Temperature or other field data
   :type field: numpy.ndarray
   :param qual: Quality flags array
   :type qual: numpy.ndarray
   :param qual_thresh: Maximum quality value considered valid
   :type qual_thresh: int, optional
   :param temp_bounds: Valid temperature range (min, max)
   :type temp_bounds: tuple, optional
   :param field: Field type ('SST' or other)
   :type field: str, optional
   :return: Binary mask where 0=valid, 1=invalid
   :rtype: numpy.ndarray

.. py:function:: preproc_field(field, mask, only_inpaint=False, inpaint=True, median=True, med_size=(3, 1), downscale=True, dscale_size=(2, 2), min_mean=None, sigmoid=False, scale=None)

   Preprocess a field for analysis.
   
   :param field: Input field data
   :type field: numpy.ndarray
   :param mask: Binary mask of invalid pixels
   :type mask: numpy.ndarray
   :param only_inpaint: Whether to only perform inpainting
   :type only_inpaint: bool, optional
   :param inpaint: Whether to perform inpainting
   :type inpaint: bool, optional
   :param median: Whether to apply median filtering
   :type median: bool, optional
   :param med_size: Median filter window size
   :type med_size: tuple, optional
   :param downscale: Whether to apply downscaling
   :type downscale: bool, optional
   :param dscale_size: Downscaling factor
   :type dscale_size: tuple, optional
   :param min_mean: Minimum mean value required
   :type min_mean: float or None, optional
   :param sigmoid: Whether to apply sigmoid preprocessing
   :type sigmoid: bool, optional
   :param scale: Scaling factor
   :type scale: float or None, optional
   :return: Preprocessed field and metadata
   :rtype: (numpy.ndarray, dict)

.. py:function:: inpaint_field(field, mask)

   Inpaint missing values in a field using Laplace equation.
   
   :param field: Input field data
   :type field: numpy.ndarray
   :param mask: Binary mask of invalid pixels
   :type mask: numpy.ndarray
   :return: Inpainted field
   :rtype: numpy.ndarray

.. py:function:: calc_stats(field, mask=None)

   Calculate statistics for a field.
   
   :param field: Input field data
   :type field: numpy.ndarray
   :param mask: Binary mask of invalid pixels
   :type mask: numpy.ndarray or None, optional
   :return: Dictionary of statistics
   :rtype: dict

Module: ``ulmo.preproc.extract``
-----------------------------

The ``extract`` module provides functions for extracting cutouts from larger satellite images.

Functions
--------

.. py:function:: clear_grid(mask, field_size, method='center', CC_max=0.05, nsgrid_draw=5, return_fracCC=False)

   Find valid regions in a masked image with sufficient clear pixels.
   
   :param mask: Binary mask array
   :type mask: numpy.ndarray
   :param field_size: Size of cutout (pixels)
   :type field_size: int
   :param method: Sampling method ('center', 'random', 'grid')
   :type method: str, optional
   :param CC_max: Maximum allowed cloud coverage (fraction)
   :type CC_max: float, optional
   :param nsgrid_draw: Number of random draws per file
   :type nsgrid_draw: int, optional
   :param return_fracCC: Whether to return the clear fraction
   :type return_fracCC: bool, optional
   :return: Tuple of (rows, cols, clear_fractions) or clear_fraction if return_fracCC=True
   :rtype: tuple or float

Module: ``ulmo.preproc.io``
------------------------

The ``io`` module provides functions for preprocessing input/output operations.

Functions
--------

.. py:function:: load_options(option_type)

   Load preprocessing options from a JSON file.
   
   :param option_type: Type of preprocessing (e.g., 'standard', 'gradient')
   :type option_type: str
   :return: Dictionary of preprocessing options
   :rtype: dict
