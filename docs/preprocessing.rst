Preprocessing
============

Ulmo includes a robust preprocessing pipeline for oceanographic data, particularly designed for preparing satellite imagery for deep learning models.

Overview
--------

The preprocessing pipeline handles:

1. Mask generation for invalid pixels (clouds, land, etc.)
2. Extraction of valid cutouts from larger satellite images
3. Inpainting of missing data
4. Normalization and standardization
5. Downscaling (if needed)

Mask Generation
--------------

Masks are created to identify valid pixels in oceanographic data:

.. code-block:: python

    from ulmo.preproc import utils as pp_utils
    
    # Generate masks from quality flags and temperature bounds
    masks = pp_utils.build_mask(
        sst,                 # Sea surface temperature data
        qual,                # Quality flags
        qual_thresh=2,       # Maximum quality value considered valid
        temp_bounds=(-2, 33) # Valid temperature range (degrees C)
    )

The resulting mask is a binary array where:
- 0 = valid pixel
- 1 = invalid pixel (cloud, land, or out of temperature range)

Cutout Extraction
----------------

Extracting fixed-size cutouts from larger satellite images:

.. code-block:: python

    from ulmo.preproc import extract
    
    # Find valid regions with sufficient clear pixels
    rows, cols, clear_fracs = extract.clear_grid(
        masks,             # Binary mask array
        field_size=128,    # Size of cutout (pixels)
        method='center',   # Sampling method ('center', 'random', 'grid')
        CC_max=0.05,       # Maximum allowed cloud coverage (fraction)
        nsgrid_draw=5      # Number of random draws per file
    )

Field Preprocessing
-----------------

Preparing individual field cutouts for model input:

.. code-block:: python

    # Preprocess a field
    pp_field, meta = pp_utils.preproc_field(
        field,              # Input field data
        mask,               # Binary mask
        inpaint=True,       # Perform inpainting on missing data
        median=True,        # Apply median filtering
        med_size=(3, 1),    # Median filter window size
        downscale=True,     # Apply downscaling
        dscale_size=(2, 2), # Downscaling factor
        min_mean=None       # Minimum mean value required (optional)
    )

The returned ``meta`` dictionary contains statistics about the field, including:
- ``mu``: Mean temperature
- ``Tmin``, ``Tmax``: Min/max temperatures
- ``T10``, ``T90``: 10th and 90th percentile temperatures

Batch Processing
--------------

For processing entire datasets, scripts like ``preproc_h5.py`` can be used:

.. code-block:: bash

    python -m ulmo.scripts.preproc_h5 input.h5 0.2 standard output.h5 --ncores 8

This processes an HDF5 file of cutouts, with:
- ``0.2`` = validation fraction
- ``standard`` = preprocessing method
- ``output.h5`` = output file
- ``--ncores 8`` = use 8 CPU cores for parallel processing

Preprocessing Options
-------------------

Standard preprocessing options are stored in JSON configuration files and can be loaded with:

.. code-block:: python

    from ulmo.preproc import io as pp_io
    
    # Load standard preprocessing options
    pdict = pp_io.load_options('standard')
    
    # Or for gradient-based preprocessing
    pdict = pp_io.load_options('gradient')

Typical preprocessing steps include:

1. **Inpainting**: Fill in missing data using Laplace equation inpainting
2. **Median filtering**: Reduce noise while preserving edges
3. **Downscaling**: Reduce image size (typically from 128×128 to 64×64)
4. **Standardization**: Scale values to have zero mean and unit variance

Advanced: Custom Preprocessing
----------------------------

You can create custom preprocessing pipelines:

.. code-block:: python

    from ulmo.preproc.utils import preproc_field
    
    # Custom preprocessing function
    def my_custom_preproc(field, mask):
        # Apply standard preprocessing
        pp_field, meta = preproc_field(
            field, mask,
            inpaint=True,
            median=True,
            downscale=True
        )
        
        # Add custom steps
        if pp_field is not None:
            # Additional processing here
            # ...
            
            # Update metadata
            meta['custom_stat'] = some_value
            
        return pp_field, meta
