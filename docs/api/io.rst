IO Module API
============

Module: ``ulmo.io``
------------------

The ``io`` module provides input/output functions for reading and writing oceanographic data and Ulmo models.

Functions
--------

.. py:function:: load_nc(filename, field='SST', verbose=True)

   Load a MODIS or equivalent .nc file.
   
   :param filename: Path to the netCDF file
   :type filename: str
   :param field: Field to load, e.g., 'SST', 'aph_443'
   :type field: str, optional
   :param verbose: Whether to print information about the loaded data
   :type verbose: bool, optional
   :return: Tuple containing (field_data, quality_flags, latitude, longitude)
   :rtype: (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)

.. py:function:: load_main_table(tbl_file, verbose=True)

   Load a table of cutouts.
   
   :param tbl_file: Path to table file (CSV, feather, or parquet format)
   :type tbl_file: str
   :param verbose: Whether to print information about the loaded table
   :type verbose: bool, optional
   :return: Table of cutouts
   :rtype: pandas.DataFrame

.. py:function:: write_main_table(main_table, outfile, to_s3=True)

   Write the main table for Ulmo analysis.
   
   :param main_table: Table of cutouts
   :type main_table: pandas.DataFrame
   :param outfile: Output filename with extension (.csv, .feather, .parquet)
   :type outfile: str
   :param to_s3: Whether to write to S3 storage
   :type to_s3: bool, optional

.. py:function:: download_file_from_s3(local_file, s3_uri, clobber_local=True, verbose=True)

   Download a file from S3 to local storage.
   
   :param local_file: Local path for the downloaded file
   :type local_file: str
   :param s3_uri: S3 URI to download
   :type s3_uri: str
   :param clobber_local: Whether to overwrite existing local file
   :type clobber_local: bool, optional
   :param verbose: Whether to print download information
   :type verbose: bool, optional

.. py:function:: upload_file_to_s3(local_file, s3_uri)

   Upload a file to S3 storage.
   
   :param local_file: Path to local file
   :type local_file: str
   :param s3_uri: Destination S3 URI
   :type s3_uri: str

.. py:function:: load_to_bytes(s3_uri)

   Load an S3 file into memory as a BytesIO object.
   
   :param s3_uri: S3 URI to load
   :type s3_uri: str
   :return: BytesIO object containing the file content
   :rtype: io.BytesIO

.. py:function:: write_bytes_to_s3(bytes_, s3_uri)

   Write bytes to S3 storage.
   
   :param bytes_: BytesIO object containing data
   :type bytes_: io.BytesIO
   :param s3_uri: Destination S3 URI
   :type s3_uri: str

.. py:function:: jsonify(obj, debug=False)

   Recursively process an object to make it JSON serializable.
   
   :param obj: Object to convert to JSON-friendly format
   :type obj: any
   :param debug: Whether to print debug information
   :type debug: bool, optional
   :return: JSON-serializable version of the input
   :rtype: dict, list, str, float, int, or bool

.. py:function:: loadjson(filename)

   Load a JSON file, with support for gzipped files.
   
   :param filename: Path to JSON file
   :type filename: str
   :return: Loaded JSON object
   :rtype: dict

.. py:function:: savejson(filename, obj, overwrite=False, indent=None, easy_to_read=False, **kwargs)

   Save a Python object to a JSON file.
   
   :param filename: Output filename
   :type filename: str
   :param obj: Object to save
   :type obj: dict
   :param overwrite: Whether to overwrite existing file
   :type overwrite: bool, optional
   :param indent: Indentation level for JSON formatting
   :type indent: int or None, optional
   :param easy_to_read: Whether to format for human readability
   :type easy_to_read: bool, optional
   :param kwargs: Additional keyword arguments for json.dump()

.. py:function:: loadyaml(filename)

   Load a YAML file.
   
   :param filename: Path to YAML file
   :type filename: str
   :return: Loaded YAML object
   :rtype: dict

Module: ``ulmo.models.io``
-------------------------

The ``ulmo.models.io`` module provides functions for loading and saving Ulmo models.

Functions
--------

.. py:function:: load_ulmo_model(model_name, datadir=None, local=False)

   Load a pre-trained Ulmo model.
   
   :param model_name: Name of the model to load
   :type model_name: str
   :param datadir: Directory containing model data
   :type datadir: str or None, optional
   :param local: Whether to use local storage
   :type local: bool, optional
   :return: Loaded probabilistic autoencoder model
   :rtype: ulmo.ood.ProbabilisticAutoencoder

Module: ``ulmo.preproc.io``
--------------------------

The ``ulmo.preproc.io`` module provides functions for preprocessing input/output operations.

Functions
--------

.. py:function:: load_options(option_type)

   Load preprocessing options from a JSON file.
   
   :param option_type: Type of preprocessing (e.g., 'standard', 'gradient')
   :type option_type: str
   :return: Dictionary of preprocessing options
   :rtype: dict
