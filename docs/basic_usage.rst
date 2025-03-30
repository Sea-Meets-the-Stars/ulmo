Basic Usage Tutorial
===================

This tutorial introduces the fundamental concepts of Ulmo and demonstrates how to use the core functionality of the package.

Prerequisites
------------

Before starting, make sure you have:

1. Installed Ulmo (see :doc:`../installation`)
2. Downloaded pre-trained models
3. Some sample oceanographic data (or the sample data provided with Ulmo)

Loading a Pre-trained Model
--------------------------

Let's start by loading a pre-trained model for analyzing sea surface temperature (SST) data:

.. code-block:: python

    import os
    from ulmo.models import io as model_io
    
    # Set the model directory (if not already set in environment)
    os.environ['SST_OOD_MODELDIR'] = '/path/to/models'
    
    # Load the standard model for MODIS L2 data
    pae = model_io.load_ulmo_model('model-l2-std')
    
    print("Model loaded successfully!")
    print(f"Autoencoder latent dimension: {pae.autoencoder.latent_dim}")
    print(f"Flow dimension: {pae.flow.dim}")

The model is now ready for analyzing oceanographic data.

Exploring the Model Structure
---------------------------

Let's examine the structure of the Probabilistic Autoencoder (PAE) model:

.. code-block:: python

    # Check the autoencoder architecture
    print("Autoencoder:")
    print(f"  Input shape: ({pae.autoencoder.c}, {pae.autoencoder.w}, {pae.autoencoder.h})")
    print(f"  Latent dimension: {pae.autoencoder.latent_dim}")
    
    # Check the normalizing flow architecture
    print("\nNormalizing Flow:")
    print(f"  Dimension: {pae.flow.dim}")
    print(f"  Transform type: {pae.flow.transform_type}")
    print(f"  Number of layers: {pae.flow.n_layers}")

Loading Sample Data
-----------------

Now, let's load some sample data to analyze:

.. code-block:: python

    import numpy as np
    import h5py
    
    # Load sample data from a preprocessed HDF5 file
    with h5py.File('sample_data.h5', 'r') as f:
        # Load a single preprocessed field
        sample_field = f['valid'][0]
        
        # Print field information
        print(f"Field shape: {sample_field.shape}")
        print(f"Field min/max: {sample_field.min():.2f}/{sample_field.max():.2f}")
        
        # Reshape for model input (add batch and channel dimensions if needed)
        if len(sample_field.shape) == 2:
            sample_field = sample_field.reshape(1, 1, *sample_field.shape)

Evaluating a Field
----------------

Now we can evaluate the field with our model:

.. code-block:: python

    # Convert to PyTorch tensor
    import torch
    sample_tensor = torch.from_numpy(sample_field).float()
    
    # Compute latent representation
    latent = pae.encode(sample_tensor)
    print(f"Latent vector shape: {latent.shape}")
    
    # Compute log-likelihood
    log_prob = pae.log_prob(sample_tensor)
    print(f"Log-likelihood: {log_prob.item():.2f}")
    
    # Get reconstruction
    reconstruction = pae.reconstruct(sample_tensor)
    
    # Convert back to numpy for visualization
    reconstruction_np = reconstruction.cpu().detach().numpy()

Visualizing Results
-----------------

Let's visualize the original field and its reconstruction:

.. code-block:: python

    import matplotlib.pyplot as plt
    import seaborn as sns
    from ulmo.plotting import plotting
    
    # Load color palette
    pal, cmap = plotting.load_palette()
    
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot original field
    sns.heatmap(sample_field[0, 0], ax=axes[0], cmap=cmap, 
                vmin=-2, vmax=2)
    axes[0].set_title("Original Field")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    # Plot reconstruction
    sns.heatmap(reconstruction_np[0, 0], ax=axes[1], cmap=cmap, 
                vmin=-2, vmax=2)
    axes[1].set_title("Reconstruction")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    plt.tight_layout()
    plt.show()

Interpreting the Results
----------------------

The log-likelihood score indicates how "normal" or "anomalous" the field is according to the trained model:

- Higher log-likelihood values (closer to zero) indicate patterns that are well-represented in the training data
- Lower log-likelihood values (more negative) indicate unusual or anomalous patterns

The reconstruction quality also provides insight:
- Well-reconstructed fields indicate patterns the model has learned
- Poor reconstructions with significant differences may indicate anomalous features

Batch Processing
--------------

For analyzing multiple fields:

.. code-block:: python

    # Load a batch of fields
    with h5py.File('sample_data.h5', 'r') as f:
        # Get first 10 fields
        batch_fields = f['valid'][:10]
        
        if len(batch_fields.shape) == 3:
            # Add channel dimension if missing
            batch_fields = batch_fields.reshape(10, 1, *batch_fields.shape[1:])
    
    # Convert to tensor
    batch_tensor = torch.from_numpy(batch_fields).float()
    
    # Compute log-likelihoods
    batch_log_probs = pae.log_prob(batch_tensor)
    
    # Print results
    for i, log_prob in enumerate(batch_log_probs):
        print(f"Field {i}: Log-likelihood = {log_prob.item():.2f}")
    
    # Find the most anomalous field
    most_anomalous_idx = torch.argmin(batch_log_probs).item()
    print(f"\nMost anomalous field: {most_anomalous_idx}")
    print(f"Log-likelihood: {batch_log_probs[most_anomalous_idx].item():.2f}")

Conclusion
---------

In this tutorial, we've covered:

1. Loading a pre-trained Ulmo model
2. Examining the model architecture
3. Loading and preprocessing sample data
4. Evaluating fields using the model
5. Visualizing and interpreting the results
6. Processing multiple fields in batch mode

These basic operations form the foundation for more advanced analyses, such as anomaly detection and spatial pattern analysis, which are covered in other tutorials.

Next Steps
---------

- Learn how to preprocess raw satellite data in the :doc:`data_preprocessing` tutorial
- Explore anomaly detection techniques in the :doc:`anomaly_detection` tutorial
- Understand spatial analysis in the :doc:`spatial_analysis` tutorial
