Utils Module API
==============

Module: ``ulmo.utils``
-------------------

The ``utils`` module provides utility functions and classes for Ulmo.

Classes
------

.. py:class:: HDF5Dataset(filepath, partition='train', return_mask=False)

   PyTorch dataset for HDF5 files.
   
   :param filepath: Path to HDF5 file
   :type filepath: str
   :param partition: Partition to use ('train' or 'valid')
   :type partition: str, optional
   :param return_mask: Whether to return masks with the data
   :type return_mask: bool, optional

Functions
--------

.. py:function:: id_collate(batch)

   Identity collation function for DataLoader.
   
   :param batch: Batch of data
   :type batch: list
   :return: Same batch without additional processing
   :rtype: list

.. py:function:: get_quantiles(values)

   Compute quantiles for a set of values.
   
   :param values: Values to compute quantiles for
   :type values: numpy.ndarray or list
   :return: Quantile values
   :rtype: numpy.ndarray

Module: ``ulmo.utils.image_utils``
-------------------------------

The ``image_utils`` module provides utilities for working with oceanographic images.

Functions
--------

.. py:function:: grab_img(cutout, path, ptype='std', preproc_file=None)

   Grab the image for a cutout.
   
   :param cutout: Row from a cutout table
   :type cutout: pandas.Series
   :param path: Path to the data directory
   :type path: str
   :param ptype: Preprocessing type ('std' or 'loggrad')
   :type ptype: str, optional
   :param preproc_file: Path to preprocessed file (overrides path+ptype)
   :type preproc_file: str or None, optional
   :return: Tuple of (field, mask)
   :rtype: tuple[numpy.ndarray, numpy.ndarray]

.. py:function:: evals_to_healpix(eval_tbl, nside, log=False, mask=True)

   Generate a HEALPix map of cutout locations.
   
   :param eval_tbl: Table of cutouts with lat, lon columns
   :type eval_tbl: pandas.DataFrame
   :param nside: HEALPix resolution parameter
   :type nside: int
   :param log: Whether to use logarithmic scale
   :type log: bool, optional
   :param mask: Whether to mask areas with no data
   :type mask: bool, optional
   :return: Tuple of (count_map, longitudes, latitudes)
   :rtype: tuple[healpy.ma, numpy.ndarray, numpy.ndarray]

Module: ``ulmo.utils.models``
--------------------------

The ``models`` module provides utilities for working with models.

Functions
--------

.. py:function:: load(model_type='standard', local=False, debug=False)

   Load a pre-trained model.
   
   :param model_type: Type of model to load ('standard' or 'loggrad')
   :type model_type: str, optional
   :param local: Whether to use local storage
   :type local: bool, optional
   :param debug: Whether to print debug information
   :type debug: bool, optional
   :return: Loaded probabilistic autoencoder model
   :rtype: ulmo.ood.ProbabilisticAutoencoder

Module: ``ulmo.utils.fft``
----------------------

The ``fft`` module provides utilities for spectral analysis.

Functions
--------

.. py:function:: fast_fft(array, dim, d, small_range=[6000, 15000], large_range=[12000, 50000], Detrend_Demean=False)

   Perform FFT and calculate power spectral density.
   
   :param array: Input array
   :type array: numpy.ndarray
   :param dim: Dimension to compute FFT along
   :type dim: int
   :param d: Sample spacing in meters
   :type d: float
   :param small_range: Wavelength range for small-scale analysis
   :type small_range: list, optional
   :param large_range: Wavelength range for large-scale analysis
   :type large_range: list, optional
   :param Detrend_Demean: Whether to detrend and demean the array
   :type Detrend_Demean: bool, optional
   :return: Dictionary with PSD, wavenumbers, slopes, and intercepts
   :rtype: dict

Module: ``ulmo.utils.rossby``
--------------------------

The ``rossby`` module provides utilities for working with Rossby radii.

Functions
--------

.. py:function:: load_rossdata()

   Load the Rossby radius data from file.
   
   :return: DataFrame with Rossby radius data
   :rtype: pandas.DataFrame

.. py:function:: calc_rossby_radius(lon, lat)

   Calculate the Rossby radius for given coordinates.
   
   :param lon: Longitude values
   :type lon: numpy.ndarray
   :param lat: Latitude values
   :type lat: numpy.ndarray
   :return: Rossby radius values
   :rtype: numpy.ndarray
