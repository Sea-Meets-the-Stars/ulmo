Data Preprocessing Tutorial
========================

This tutorial explains how to preprocess oceanographic data for analysis with Ulmo, including handling cloud masks, extracting cutouts, and applying preprocessing steps.

Loading Raw Satellite Data
------------------------

First, let's load some raw satellite data:

.. code-block:: python

    from ulmo import io as ulmo_io
    
    # Load MODIS L2 data
    filename = "path/to/A2010123123400.L2_LAC_SST.nc"
    sst, qual, latitude, longitude = ulmo_io.load_nc(
        filename, 
        field='SST',  # Sea Surface Temperature
        verbose=True  # Print information about the loaded data
    )
    
    print(f"SST shape: {sst.shape}")
    print(f"Quality flags shape: {qual.shape}")
    print(f"Latitude range: {latitude.min():.2f} to {latitude.max():.2f}")
    print(f"Longitude range: {longitude.min():.2f} to {longitude.max():.2f}")
    print(f"Temperature range: {sst.min():.2f} to {sst.max():.2f} °C")

Creating Cloud Masks
------------------

Next, we need to create masks to identify valid pixels (those not obscured by clouds or land):

.. code-block:: python

    from ulmo.preproc import utils as pp_utils
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Generate masks for cloudy/invalid pixels
    # 0 = valid pixel, 1 = invalid (cloud/land)
    masks = pp_utils.build_mask(
        sst,                # Sea surface temperature data
        qual,               # Quality flags
        qual_thresh=2,      # Maximum quality value considered valid
        temp_bounds=(-2, 33)  # Valid temperature range (degrees C)
    )
    
    # Calculate cloud coverage
    cloud_coverage = np.mean(masks) * 100
    print(f"Cloud coverage: {cloud_coverage:.2f}%")
    
    # Visualize mask
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(sst, cmap='viridis')
    plt.title("Sea Surface Temperature")
    plt.colorbar(label="Temperature (°C)")
    
    plt.subplot(122)
    plt.imshow(masks, cmap='gray')
    plt.title("Cloud Mask (white = cloud/invalid)")
    plt.colorbar(label="Invalid (1) / Valid (0)")
    
    plt.tight_layout()
    plt.show()

Extracting Cutouts
----------------

Now we can extract fixed-size cutouts from the larger satellite image:

.. code-block:: python

    from ulmo.preproc import extract
    
    # Limit analysis to near-nadir pixels (central part of satellite swath)
    nadir_offset = 480
    nadir_pix = sst.shape[1] // 2
    lb = nadir_pix - nadir_offset
    ub = nadir_pix + nadir_offset
    
    sst_subset = sst[:, lb:ub]
    masks_subset = masks[:, lb:ub]
    
    # Find valid regions with sufficient clear pixels
    field_size = 128
    max_cloud_coverage = 0.05  # 5% maximum cloud coverage
    
    rows, cols, clear_fracs = extract.clear_grid(
        masks_subset,         # Binary mask array
        field_size=field_size,  # Size of cutout (pixels)
        method='center',      # Sampling method ('center', 'random', 'grid')
        CC_max=max_cloud_coverage,  # Maximum allowed cloud coverage (fraction)
        nsgrid_draw=5         # Number of random draws per file
    )
    
    if rows is None:
        print("No suitable cutouts found with the current cloud coverage threshold.")
    else:
        print(f"Found {len(rows)} suitable cutouts")
        
        # Display the first cutout
        if len(rows) > 0:
            r, c = rows[0], cols[0]
            cutout = sst_subset[r:r+field_size, c:c+field_size]
            mask = masks_subset[r:r+field_size, c:c+field_size]
            
            plt.figure(figsize=(10, 5))
            plt.subplot(121)
            plt.imshow(cutout, cmap='viridis')
            plt.title(f"Cutout at ({r},{c+lb})")
            plt.colorbar(label="Temperature (°C)")
            
            plt.subplot(122)
            plt.imshow(mask, cmap='gray')
            plt.title(f"Cloud coverage: {clear_fracs[0]*100:.2f}%")
            plt.colorbar(label="Invalid (1) / Valid (0)")
            
            plt.tight_layout()
            plt.show()

Preprocessing Cutouts
------------------

After extracting cutouts, we need to preprocess them for analysis:

.. code-block:: python

    # Let's preprocess the first cutout
    cutout = sst_subset[rows[0]:rows[0]+field_size, cols[0]:cols[0]+field_size]
    mask = masks_subset[rows[0]:rows[0]+field_size, cols[0]:cols[0]+field_size]
    
    # Apply standard preprocessing
    pp_field, meta = pp_utils.preproc_field(
        cutout,              # Input field
        mask,                # Cloud mask
        inpaint=True,        # Fill in missing data
        median=True,         # Apply median filtering
        med_size=(3, 1),     # Median filter window size
        downscale=True,      # Reduce size
        dscale_size=(2, 2)   # Downscaling factor (128x128 -> 64x64)
    )
    
    # Print metadata from preprocessing
    print("Preprocessing metadata:")
    for key, value in meta.items():
        print(f"  {key}: {value}")
    
    # Visualize before and after preprocessing
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(cutout, cmap='viridis')
    plt.title("Original Cutout")
    plt.colorbar(label="Temperature (°C)")
    
    plt.subplot(132)
    # Inpainted but not downscaled (intermediate step)
    if 'inpainted' in meta:
        plt.imshow(meta['inpainted'], cmap='viridis')
        plt.title("After Inpainting")
        plt.colorbar(label="Temperature (°C)")
    
    plt.subplot(133)
    plt.imshow(pp_field, cmap='viridis')
    plt.title("Final Preprocessed")
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

Batch Preprocessing
-----------------

For analyzing multiple cutouts, we can process them in batch:

.. code-block:: python

    # Process all the cutouts
    pp_fields = []
    pp_metas = []
    
    for i, (r, c, clear_frac) in enumerate(zip(rows, cols, clear_fracs)):
        if i >= 10:  # Limit to 10 for this example
            break
            
        cutout = sst_subset[r:r+field_size, c:c+field_size]
        mask = masks_subset[r:r+field_size, c:c+field_size]
        
        pp_field, meta = pp_utils.preproc_field(
            cutout, mask,
            inpaint=True,
            median=True,
            downscale=True
        )
        
        if pp_field is not None:
            pp_fields.append(pp_field)
            pp_metas.append(meta)
    
    print(f"Successfully preprocessed {len(pp_fields)} cutouts")
    
    # Display a grid of preprocessed cutouts
    if len(pp_fields) > 0:
        n_cols = min(4, len(pp_fields))
        n_rows = (len(pp_fields) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(n_cols * 3, n_rows * 3))
        for i, pp_field in enumerate(pp_fields):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(pp_field, cmap='viridis')
            plt.title(f"Cutout {i}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

Saving Preprocessed Data
---------------------

Finally, we can save the preprocessed cutouts for future use:

.. code-block:: python

    import h5py
    import numpy as np
    
    # Convert to numpy arrays
    pp_fields_array = np.stack(pp_fields)
    
    # Create metadata table
    metadata = []
    for i, (r, c, clear_frac, meta) in enumerate(zip(rows[:len(pp_fields)], 
                                                  cols[:len(pp_fields)], 
                                                  clear_fracs[:len(pp_fields)],
                                                  pp_metas)):
        metadata.append([
            os.path.basename(filename),  # Original filename
            str(r),                      # Row in original image
            str(c + lb),                 # Column in original image
            str(meta['mu']),            # Mean temperature
            str(clear_frac)              # Clear fraction
        ])
    
    metadata_array = np.array(metadata)
    
    # Save to HDF5 file
    with h5py.File('preprocessed_cutouts.h5', 'w') as f:
        # Store the preprocessed fields
        f.create_dataset('valid', data=pp_fields_array)
        
        # Store metadata
        dset = f.create_dataset('metadata', data=metadata_array.astype('S'))
        dset.attrs['columns'] = ['filename', 'row', 'column', 'mean_temperature', 'clear_fraction']

Using the Preprocessing Script
---------------------------

For processing large datasets, Ulmo provides command-line scripts:

.. code-block:: bash

    # Extract cutouts from MODIS L2 files
    python -m ulmo.scripts.extract_modis --field SST --year 2010 --clear_threshold 95 --field_size 128 --ncores 8
    
    # Preprocess the extracted cutouts
    python -m ulmo.scripts.preproc_h5 extracted_cutouts.h5 0.2 standard preprocessed_data.h5 --ncores 8

The parameters for the preprocessing script are:
- ``extracted_cutouts.h5``: Input HDF5 file with extracted cutouts
- ``0.2``: Validation fraction (80% training, 20% validation)
- ``standard``: Preprocessing method
- ``preprocessed_data.h5``: Output file for preprocessed data
- ``--ncores 8``: Use 8 CPU cores for parallel processing

Custom Preprocessing Options
-------------------------

You can customize preprocessing by creating your own options dictionary:

.. code-block:: python

    # Define custom preprocessing options
    custom_options = {
        'inpaint': True,        # Fill missing data
        'median': True,         # Apply median filtering
        'med_size': (3, 3),     # Custom median filter size
        'downscale': True,      # Downscale the image
        'dscale_size': (2, 2),  # Downscaling factor
        'min_mean': 5.0,        # Minimum mean temperature to keep
        'sigmoid': False,       # Whether to apply sigmoid transformation
        'scale': 0.5            # Custom scaling factor
    }
    
    # Apply custom preprocessing
    pp_field, meta = pp_utils.preproc_field(cutout, mask, **custom_options)

Conclusion
---------

In this tutorial, we've covered:

1. Loading raw satellite data
2. Creating cloud masks to identify valid pixels
3. Extracting clear cutouts from larger images
4. Preprocessing cutouts using various techniques:
   - Inpainting to fill missing data
   - Median filtering to reduce noise
   - Downscaling to reduce dimensions
5. Saving preprocessed data for future analysis
6. Using batch processing for large datasets

These preprocessed cutouts are now ready for analysis with Ulmo's anomaly detection models.