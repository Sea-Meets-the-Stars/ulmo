Model Evaluation Tutorial
======================

This tutorial demonstrates how to evaluate oceanographic data using Ulmo's pre-trained models and interpret the results, including log-likelihood scores and reconstructions.

Understanding Model Evaluation in Ulmo
------------------------------------

Ulmo evaluates oceanographic data using two main approaches:

1. **Log-likelihood scores** - Quantify how "normal" or "unusual" a pattern is
2. **Reconstruction quality** - Reveal what features the model has learned

The evaluation process involves:
1. Encoding input data to a latent representation using the autoencoder
2. Computing the log-likelihood of the latent representation using the flow model
3. Reconstructing the input from its latent representation for visual comparison

Loading Pre-trained Models
------------------------

Let's start by loading a pre-trained model:

.. code-block:: python

    import os
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import h5py
    
    from ulmo.models import io as model_io
    
    # Set model directory (if not already set in environment)
    os.environ['SST_OOD_MODELDIR'] = '/path/to/models'
    
    # Load the standard model for MODIS L2 data
    pae = model_io.load_ulmo_model('model-l2-std')
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Print model information
    print(f"Autoencoder architecture:")
    print(f"  Input shape: {pae.autoencoder.c}x{pae.autoencoder.w}x{pae.autoencoder.h}")
    print(f"  Latent dimension: {pae.autoencoder.latent_dim}")
    print(f"Flow model:")
    print(f"  Type: {pae.flow.transform_type}")
    print(f"  Layers: {pae.flow.n_layers}")

Evaluating Individual Fields
--------------------------

Let's evaluate individual oceanographic fields:

.. code-block:: python

    import seaborn as sns
    from ulmo.plotting import plotting
    
    # Load color palette
    pal, cmap = plotting.load_palette()
    
    # Load a sample field from an H5 file
    with h5py.File('sample_data.h5', 'r') as f:
        # Get the first field in the validation set
        sample_field = f['valid'][0:1]
    
    # Ensure proper dimensions (batch, channel, height, width)
    if len(sample_field.shape) == 3:
        sample_field = sample_field.reshape(1, 1, *sample_field.shape[1:])
    
    # Convert to PyTorch tensor
    sample_tensor = torch.from_numpy(sample_field).float().to(device)
    
    # Perform evaluation
    with torch.no_grad():
        # Get latent representation
        latent = pae.encode(sample_tensor)
        print(f"Latent shape: {latent.shape}")
        
        # Calculate log-likelihood
        log_prob = pae.log_prob(sample_tensor)
        print(f"Log-likelihood: {log_prob.item():.2f}")
        
        # Get reconstruction
        reconstruction = pae.reconstruct(sample_tensor)
    
    # Convert to numpy for visualization
    sample_np = sample_field[0, 0]
    reconstruction_np = reconstruction.cpu().detach().numpy()[0, 0]
    
    # Visualize original and reconstruction
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Original
    sns.heatmap(sample_np, ax=axes[0], cmap=cmap, vmin=-2, vmax=2)
    axes[0].set_title("Original Field")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    # Reconstruction
    sns.heatmap(reconstruction_np, ax=axes[1], cmap=cmap, vmin=-2, vmax=2)
    axes[1].set_title("Reconstruction")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    plt.suptitle(f"Log-Likelihood: {log_prob.item():.2f}")
    plt.tight_layout()
    plt.show()
    
    # Calculate reconstruction error
    mse = np.mean((sample_np - reconstruction_np) ** 2)
    print(f"Reconstruction Mean Squared Error: {mse:.4f}")

Batch Evaluation
--------------

For evaluating multiple fields, we can use Ulmo's batch processing capabilities:

.. code-block:: python

    from tqdm import tqdm
    
    # Load a batch of fields from an H5 file
    with h5py.File('sample_data.h5', 'r') as f:
        # Get the first 100 fields (or all if fewer)
        num_fields = min(100, f['valid'].shape[0])
        fields = f['valid'][:num_fields]
    
    # Ensure proper dimensions
    if len(fields.shape) == 3:
        fields = fields.reshape(fields.shape[0], 1, *fields.shape[1:])
    
    # Evaluate in batches
    batch_size = 16
    num_batches = (num_fields + batch_size - 1) // batch_size
    
    all_latents = []
    all_log_probs = []
    all_mses = []
    
    for i in tqdm(range(num_batches), desc="Evaluating fields"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_fields)
        
        batch = torch.from_numpy(fields[start_idx:end_idx]).float().to(device)
        
        with torch.no_grad():
            # Get latent representations
            latents = pae.encode(batch)
            all_latents.append(latents.cpu().numpy())
            
            # Calculate log-likelihoods
            log_probs = pae.log_prob(batch)
            all_log_probs.append(log_probs.cpu().numpy())
            
            # Get reconstructions
            reconstructions = pae.reconstruct(batch)
            
            # Calculate MSEs
            for j in range(batch.shape[0]):
                original = fields[start_idx + j, 0]
                recon = reconstructions[j, 0].cpu().numpy()
                mse = np.mean((original - recon) ** 2)
                all_mses.append(mse)
    
    # Concatenate results
    latents = np.concatenate(all_latents)
    log_probs = np.concatenate(all_log_probs)
    
    # Print statistics
    print(f"Evaluation of {num_fields} fields:")
    print(f"Log-likelihood statistics:")
    print(f"  Mean: {np.mean(log_probs):.2f}")
    print(f"  Std: {np.std(log_probs):.2f}")
    print(f"  Min: {np.min(log_probs):.2f} (Most anomalous)")
    print(f"  Max: {np.max(log_probs):.2f} (Most normal)")
    print(f"Reconstruction MSE statistics:")
    print(f"  Mean: {np.mean(all_mses):.4f}")
    print(f"  Std: {np.std(all_mses):.4f}")
    print(f"  Min: {np.min(all_mses):.4f} (Best reconstruction)")
    print(f"  Max: {np.max(all_mses):.4f} (Worst reconstruction)")
    
    # Plot histogram of log-likelihoods
    plt.figure(figsize=(10, 6))
    plt.hist(log_probs, bins=20, alpha=0.7, color='skyblue')
    plt.axvline(np.mean(log_probs), color='red', linestyle='--', 
                label=f'Mean: {np.mean(log_probs):.2f}')
    plt.xlabel('Log-Likelihood')
    plt.ylabel('Count')
    plt.title('Distribution of Log-Likelihood Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

Using `eval_data_file` for Large Datasets
---------------------------------------

For large datasets, use the built-in `eval_data_file` method:

.. code-block:: python

    # Evaluate an entire preprocessed file
    output_file = 'evaluation_results.h5'
    
    log_probs = pae.eval_data_file(
        data_file='large_dataset.h5',  # Input file
        dataset='valid',               # Dataset to evaluate ('valid' or 'train')
        output_file=output_file,       # Output file
        csv=True                       # Also save as CSV
    )
    
    print(f"Evaluation results saved to {output_file}")
    print(f"Statistics:")
    print(f"  Mean: {np.mean(log_probs):.2f}")
    print(f"  Std: {np.std(log_probs):.2f}")
    print(f"  Min: {np.min(log_probs):.2f}")
    print(f"  Max: {np.max(log_probs):.2f}")

Visualizing Evaluation Results
----------------------------

Let's create visualizations to better understand the model evaluation:

.. code-block:: python

    # Visualize fields across the log-likelihood spectrum
    indices = [
        np.argmin(log_probs),           # Most anomalous
        np.percentile(log_probs, 25, interpolation='nearest'),  # 25th percentile
        np.percentile(log_probs, 50, interpolation='nearest'),  # Median
        np.percentile(log_probs, 75, interpolation='nearest'),  # 75th percentile
        np.argmax(log_probs)            # Most normal
    ]
    
    # Create a figure to display them
    fig, axes = plt.subplots(len(indices), 2, figsize=(10, 12))
    
    titles = ["Most Anomalous", "25th Percentile", 
              "Median", "75th Percentile", "Most Normal"]
    
    for i, idx in enumerate(indices):
        # Get original field
        field = fields[idx, 0]
        
        # Get reconstruction
        with torch.no_grad():
            tensor_field = torch.from_numpy(fields[idx:idx+1]).float().to(device)
            reconstruction = pae.reconstruct(tensor_field)
            reconstruction = reconstruction.cpu().numpy()[0, 0]
        
        # Original
        sns.heatmap(field, ax=axes[i, 0], cmap=cmap, vmin=-2, vmax=2)
        axes[i, 0].set_title(f"Original")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        
        # Reconstruction
        sns.heatmap(reconstruction, ax=axes[i, 1], cmap=cmap, vmin=-2, vmax=2)
        axes[i, 1].set_title(f"Reconstruction")
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
        
        # Add text label with statistics
        axes[i, 0].text(-0.1, 0.5, f"{titles[i]}\nLL: {log_probs[idx]:.2f}", 
                      transform=axes[i, 0].transAxes, 
                      verticalalignment='center', horizontalalignment='right')
    
    plt.tight_layout()
    plt.show()

Analyzing the Latent Space
------------------------

Let's analyze the latent space to understand what the model has learned:

.. code-block:: python

    from sklearn.decomposition import PCA
    
    # Perform PCA on latent vectors
    pca = PCA(n_components=10)
    latents_pca = pca.fit_transform(latents)
    
    # Print explained variance
    print("PCA explained variance ratios:")
    for i, var in enumerate(pca.explained_variance_ratio_[:10]):
        print(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%)")
    print(f"Total (10 components): {sum(pca.explained_variance_ratio_[:10])*100:.2f}%")
    
    # Plot first two principal components
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latents_pca[:, 0], latents_pca[:, 1], 
               c=log_probs, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Log-Likelihood')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Latent Space Colored by Log-Likelihood')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Plot correlation between principal components and log-likelihood
    correlations = [np.corrcoef(latents_pca[:, i], log_probs)[0, 1] for i in range(10)]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(10), correlations)
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    plt.xticks(range(10), [f'PC{i+1}' for i in range(10)])
    plt.ylabel('Correlation with Log-Likelihood')
    plt.title('Correlation between Principal Components and Log-Likelihood')
    plt.grid(True, alpha=0.3)
    plt.show()

Comparing Different Models
-----------------------

If you have multiple models, you can compare their evaluations:

.. code-block:: python

    # Load a different model (e.g., with gradient-based preprocessing)
    pae_grad = model_io.load_ulmo_model('model-l2-loggrad')
    
    # Evaluate the same fields with the new model
    grad_log_probs = []
    
    for i in tqdm(range(num_batches), desc="Evaluating with gradient model"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_fields)
        
        batch = torch.from_numpy(fields[start_idx:end_idx]).float().to(device)
        
        with torch.no_grad():
            # Calculate log-likelihoods
            log_probs_grad = pae_grad.log_prob(batch)
            grad_log_probs.append(log_probs_grad.cpu().numpy())
    
    # Concatenate results
    grad_log_probs = np.concatenate(grad_log_probs)
    
    # Compare models
    plt.figure(figsize=(10, 6))
    
    plt.scatter(log_probs, grad_log_probs, alpha=0.7)
    plt.plot([min(log_probs), max(log_probs)], [min(log_probs), max(log_probs)], 
             'k--', alpha=0.5)
    
    plt.xlabel('Log-Likelihood (Standard Model)')
    plt.ylabel('Log-Likelihood (Gradient Model)')
    plt.title('Comparison of Model Evaluations')
    plt.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr = np.corrcoef(log_probs, grad_log_probs)[0, 1]
    plt.text(0.1, 0.9, f'Correlation: {corr:.3f}', 
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.show()

Using Command-line Evaluation Tools
---------------------------------

For larger scale evaluations, use Ulmo's command-line tools:

.. code-block:: bash

    # Evaluate MODIS L2 data from 2010-2012 using standard model
    python -m ulmo.scripts.eval 2010,2012 std
    
    # Evaluate specific files
    python -m ulmo.scripts.LL_modis path/to/MODIS_file.nc 500 500 --show

Interpreting Evaluation Results
-----------------------------

When interpreting log-likelihood scores:

1. **Lower values** (more negative) indicate more anomalous patterns
2. **Higher values** (closer to zero) indicate more normal patterns

Common thresholds for anomaly detection:
- Bottom 1% (1st percentile): Extreme anomalies
- Bottom 5% (5th percentile): Clear anomalies
- Bottom 10% (10th percentile): Unusual patterns

The reconstruction quality also provides insights:
- Well-reconstructed patterns are within the model's learned distribution
- Poor reconstructions indicate patterns with features the model hasn't learned

Conclusion
---------

In this tutorial, we've covered:

1. Loading and using pre-trained Ulmo models for evaluation
2. Evaluating individual fields with log-likelihood and reconstruction analysis
3. Batch processing for evaluating multiple fields efficiently
4. Visualizing the distribution of log-likelihood scores
5. Analyzing fields across the normality spectrum
6. Examining the latent space using PCA
7. Comparing evaluations from different models
8. Using command-line tools for large-scale evaluation
9. Interpreting log-likelihood scores and reconstructions

These evaluation techniques allow you to identify and analyze interesting oceanographic patterns that deviate from the learned normal distribution.