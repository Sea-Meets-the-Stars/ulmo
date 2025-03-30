Quickstart
==========

This guide provides a quick introduction to using Ulmo for anomaly detection in oceanographic data.

Loading a Pre-trained Model
--------------------------

Ulmo provides pre-trained probabilistic autoencoder models for sea surface temperature (SST) data:

.. code-block:: python

    from ulmo.models import io as model_io
    
    # Load the standard model for MODIS L2 data
    pae = model_io.load_ulmo_model('model-l2-std')
    
    # Model components are accessible
    print(f"Autoencoder latent dimension: {pae.autoencoder.latent_dim}")
    print(f"Flow dimension: {pae.flow.dim}")

Evaluating a Single Cutout
-------------------------

You can analyze a single ocean temperature field:

.. code-block:: python

    import numpy as np
    from ulmo.preproc import utils as pp_utils
    from ulmo import io as ulmo_io
    
    # Load MODIS L2 data
    filename = "path/to/MODIS_file.nc"
    sst, qual, latitude, longitude = ulmo_io.load_nc(filename, field='SST')
    
    # Generate masks for cloudy/invalid pixels
    masks = pp_utils.build_mask(sst, qual)
    
    # Extract a 128x128 field
    row, col = 500, 500  # Example coordinates
    field_size = (128, 128)
    field = sst[row:row + field_size[0], col:col + field_size[1]]
    mask = masks[row:row + field_size[0], col:col + field_size[1]]
    
    # Pre-process the field
    pp_field, meta = pp_utils.preproc_field(field, mask)
    
    # Evaluate with the model
    latents, log_likelihood = pae.eval_numpy_img(pp_field)
    
    print(f"Log-likelihood: {log_likelihood}")
    # Lower log-likelihood indicates more anomalous patterns

Batch Processing
--------------

For analyzing multiple fields:

.. code-block:: python

    # Preprocess a batch of data
    from ulmo.preproc import extract
    
    # Find clear regions in an image
    rows, cols, clear_fracs = extract.clear_grid(masks, field_size=128, method='center', CC_max=0.05)
    
    # Process all fields
    results = []
    for r, c in zip(rows, cols):
        field = sst[r:r+128, c:c+128]
        mask = masks[r:r+128, c:c+128]
        pp_field, meta = pp_utils.preproc_field(field, mask)
        
        if pp_field is not None:
            latents, ll = pae.eval_numpy_img(pp_field)
            results.append((r, c, ll))
    
    # Sort by log-likelihood to find most anomalous regions
    results.sort(key=lambda x: x[2])

Visualizing Results
-----------------

Visualize the original and reconstructed fields:

.. code-block:: python

    import matplotlib.pyplot as plt
    import seaborn as sns
    from ulmo.plotting import plotting
    
    # Get the reconstruction
    pp_field_reshaped = pp_field.reshape(1, 1, 64, 64)  # Reshape for model input
    reconstruction = pae.reconstruct(pp_field_reshaped)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Load color palette
    pal, cmap = plotting.load_palette()
    
    # Original
    sns.heatmap(pp_field, ax=axes[0], cmap=cmap, vmin=-2, vmax=2)
    axes[0].set_title("Original Field")
    axes[0].axis('off')
    
    # Reconstruction
    sns.heatmap(reconstruction[0, 0], ax=axes[1], cmap=cmap, vmin=-2, vmax=2)
    axes[1].set_title("Reconstruction")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

For more detailed examples, see the :doc:`tutorials/index` section.
