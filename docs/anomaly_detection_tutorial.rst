Anomaly Detection Tutorial
========================

This tutorial demonstrates how to identify and analyze anomalous patterns in ocean temperature data using Ulmo's probabilistic framework.

Understanding Anomaly Detection in Ulmo
-------------------------------------

Ulmo identifies anomalies using a probabilistic approach based on log-likelihood scores:

1. The model learns the distribution of normal oceanographic patterns during training
2. New observations are evaluated against this learned distribution
3. Patterns that have low probability (low log-likelihood scores) are considered anomalous

Loading a Pre-trained Model
-------------------------

First, let's load a pre-trained model:

.. code-block:: python

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    
    from ulmo.models import io as model_io
    
    # Set model directory (if not already set in environment)
    os.environ['SST_OOD_MODELDIR'] = '/path/to/models'
    
    # Load the standard model for MODIS L2 data
    pae = model_io.load_ulmo_model('model-l2-std')
    
    print("Model loaded successfully!")

Loading and Analyzing a Dataset
----------------------------

Let's load a preprocessed dataset and analyze it:

.. code-block:: python

    import h5py
    import torch
    from tqdm import tqdm
    
    # Load a batch of preprocessed data
    data_file = 'preprocessed_data.h5'
    with h5py.File(data_file, 'r') as f:
        # Load a batch of fields
        fields = f['valid'][:]
        
        # Load metadata if available
        if 'valid_metadata' in f:
            meta = f['valid_metadata']
            metadata = pd.DataFrame(meta[:].astype(np.unicode_), 
                                   columns=meta.attrs['columns'])
        else:
            metadata = pd.DataFrame()
    
    print(f"Loaded {fields.shape[0]} preprocessed fields")
    
    # Reshape for model input if needed (add channel dimension if missing)
    if len(fields.shape) == 3:
        fields = fields.reshape(fields.shape[0], 1, *fields.shape[1:])
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Calculate log-likelihoods in batches
    batch_size = 64
    n_samples = fields.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    all_log_probs = []
    all_latents = []
    
    for i in tqdm(range(n_batches), desc="Computing log probabilities"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        
        batch = torch.from_numpy(fields[start_idx:end_idx]).float().to(device)
        
        # Encode to latent space
        with torch.no_grad():
            latents = pae.encode(batch)
            all_latents.append(latents.cpu().numpy())
            
            # Calculate log probabilities
            log_probs = pae.log_prob(batch)
            all_log_probs.append(log_probs.cpu().numpy())
    
    # Concatenate results
    log_probs = np.concatenate(all_log_probs)
    latents = np.concatenate(all_latents)
    
    print(f"Log-likelihood statistics:")
    print(f"  Mean: {np.mean(log_probs):.2f}")
    print(f"  Std: {np.std(log_probs):.2f}")
    print(f"  Min: {np.min(log_probs):.2f}")
    print(f"  Max: {np.max(log_probs):.2f}")

Identifying Anomalous Patterns
----------------------------

Now let's identify and visualize anomalous patterns:

.. code-block:: python

    # Add log probabilities to metadata
    if not metadata.empty:
        metadata['log_likelihood'] = log_probs
    
    # Plot histogram of log-likelihoods
    plt.figure(figsize=(10, 6))
    sns.histplot(log_probs, bins=50, kde=True)
    
    # Add vertical lines for various percentile thresholds
    percentiles = [0.01, 0.05, 0.1]
    for p in percentiles:
        threshold = np.percentile(log_probs, p * 100)
        plt.axvline(threshold, color='r', linestyle='--', 
                   label=f"{p*100}% percentile: {threshold:.2f}")
    
    plt.xlabel('Log-Likelihood')
    plt.ylabel('Count')
    plt.title('Distribution of Log-Likelihood Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Find the most anomalous examples (lowest log-likelihood)
    anomaly_threshold = np.percentile(log_probs, 5)  # Bottom 5%
    anomaly_indices = np.where(log_probs < anomaly_threshold)[0]
    print(f"Found {len(anomaly_indices)} anomalies (log-likelihood < {anomaly_threshold:.2f})")
    
    # Sort anomalies by log-likelihood (most anomalous first)
    sorted_idx = anomaly_indices[np.argsort(log_probs[anomaly_indices])]

Visualizing Anomalies
-------------------

Let's visualize some of the anomalous patterns and their reconstructions:

.. code-block:: python

    from ulmo.plotting import plotting
    
    # Load color palette
    pal, cmap = plotting.load_palette()
    
    # Function to visualize original and reconstruction
    def visualize_field(idx, title=None):
        # Get original field
        field = fields[idx]
        if len(field.shape) == 3:
            field = field[0]  # Remove channel dimension for plotting
        
        # Get reconstruction
        with torch.no_grad():
            tensor_field = torch.from_numpy(fields[idx:idx+1]).float().to(device)
            reconstruction = pae.reconstruct(tensor_field)
            reconstruction = reconstruction.cpu().numpy()[0, 0]
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot original
        sns.heatmap(field, ax=axes[0], cmap=cmap, vmin=-2, vmax=2)
        axes[0].set_title("Original Field")
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        
        # Plot reconstruction
        sns.heatmap(reconstruction, ax=axes[1], cmap=cmap, vmin=-2, vmax=2)
        axes[1].set_title("Reconstruction")
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        
        # Add metadata if available
        if not metadata.empty:
            ll = log_probs[idx]
            metadata_str = f"Log-Likelihood: {ll:.2f}"
            
            if 'latitude' in metadata.columns and 'longitude' in metadata.columns:
                lat = metadata.iloc[idx]['latitude']
                lon = metadata.iloc[idx]['longitude']
                metadata_str += f"\nLocation: ({lat}, {lon})"
                
            if 'datetime' in metadata.columns:
                date = metadata.iloc[idx]['datetime']
                metadata_str += f"\nDate: {date}"
            
            plt.suptitle(f"{title}\n{metadata_str}")
        else:
            plt.suptitle(f"{title}\nLog-Likelihood: {log_probs[idx]:.2f}")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        plt.show()
    
    # Visualize the top 5 most anomalous fields
    for i, idx in enumerate(sorted_idx[:5]):
        visualize_field(idx, f"Anomaly #{i+1} (rank {idx})")

Comparing Normal and Anomalous Patterns
-------------------------------------

Let's compare the most normal (highest log-likelihood) and most anomalous patterns:

.. code-block:: python

    # Find the most normal examples (highest log-likelihood)
    normal_indices = np.argsort(log_probs)[-5:][::-1]
    
    # Visualize the top 5 most normal fields
    for i, idx in enumerate(normal_indices):
        visualize_field(idx, f"Normal #{i+1} (rank {idx})")
    
    # Compare the distribution of features for normal vs anomalous
    plt.figure(figsize=(12, 6))
    
    # Define normal and anomalous groups
    normal_threshold = np.percentile(log_probs, 95)  # Top 5%
    anomalous_threshold = np.percentile(log_probs, 5)  # Bottom 5%
    
    normal_group = latents[log_probs > normal_threshold]
    anomalous_group = latents[log_probs < anomalous_threshold]
    
    # Use PCA to visualize the first 2 principal components
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    all_latents_2d = pca.fit_transform(np.vstack([normal_group, anomalous_group]))
    
    # Split back into normal and anomalous
    n_normal = normal_group.shape[0]
    normal_latents_2d = all_latents_2d[:n_normal]
    anomalous_latents_2d = all_latents_2d[n_normal:]
    
    # Plot
    plt.scatter(normal_latents_2d[:, 0], normal_latents_2d[:, 1], 
               label='Normal', alpha=0.5, s=10)
    plt.scatter(anomalous_latents_2d[:, 0], anomalous_latents_2d[:, 1], 
               label='Anomalous', alpha=0.5, s=10, color='red')
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Projection of Latent Space')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

Analyzing Anomalies by Location
-----------------------------

If location data is available, let's analyze the geographical distribution of anomalies:

.. code-block:: python

    if not metadata.empty and 'latitude' in metadata.columns and 'longitude' in metadata.columns:
        # Create a new DataFrame with anomaly flags
        geo_df = metadata.copy()
        geo_df['log_likelihood'] = log_probs
        geo_df['is_anomaly'] = log_probs < anomaly_threshold
        
        # Plot geographical distribution using Ulmo's spatial plotting tools
        from ulmo.spatial_plots import show_avg_LL
        
        # Create map of log-likelihood values
        plt.figure(figsize=(12, 8))
        ax = show_avg_LL(
            geo_df,           # DataFrame with lat, lon, log_likelihood columns
            nside=64,         # HEALPix resolution
            use_mask=True,    # Mask areas with no data
            color='viridis',  # Colormap
            show=False        # Don't display yet
        )
        plt.title('Geographical Distribution of Log-Likelihood Scores')
        plt.show()

Temporal Analysis of Anomalies
----------------------------

If time data is available, let's analyze the temporal distribution:

.. code-block:: python

    if not metadata.empty and 'datetime' in metadata.columns:
        # Convert to datetime
        geo_df['datetime'] = pd.to_datetime(geo_df['datetime'])
        
        # Group by month
        geo_df['month'] = geo_df['datetime'].dt.month
        geo_df['year'] = geo_df['datetime'].dt.year
        
        # Count anomalies by month
        monthly_counts = geo_df.groupby(['year', 'month']).agg({
            'is_anomaly': 'sum',
            'log_likelihood': ['mean', 'std', 'count']
        })
        
        # Flatten multi-index columns
        monthly_counts.columns = ['_'.join(col).strip() for col in monthly_counts.columns.values]
        monthly_counts = monthly_counts.reset_index()
        
        # Calculate anomaly percentage
        monthly_counts['anomaly_percentage'] = (
            monthly_counts['is_anomaly_sum'] / monthly_counts['log_likelihood_count'] * 100
        )
        
        # Plot temporal distribution
        plt.figure(figsize=(12, 6))
        plt.subplot(211)
        plt.bar(range(len(monthly_counts)), monthly_counts['anomaly_percentage'],
               color='skyblue')
        plt.xticks(range(len(monthly_counts)), 
                  [f"{y}-{m}" for y, m in zip(monthly_counts['year'], monthly_counts['month'])],
                  rotation=45)
        plt.ylabel('Anomaly Percentage')
        plt.title('Percentage of Anomalies by Month')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(212)
        plt.plot(range(len(monthly_counts)), monthly_counts['log_likelihood_mean'], 'o-')
        plt.fill_between(
            range(len(monthly_counts)),
            monthly_counts['log_likelihood_mean'] - monthly_counts['log_likelihood_std'],
            monthly_counts['log_likelihood_mean'] + monthly_counts['log_likelihood_std'],
            alpha=0.2
        )
        plt.xticks(range(len(monthly_counts)), 
                  [f"{y}-{m}" for y, m in zip(monthly_counts['year'], monthly_counts['month'])],
                  rotation=45)
        plt.ylabel('Mean Log-Likelihood')
        plt.title('Monthly Mean Log-Likelihood')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

Creating an Anomaly Report
------------------------

Let's create a comprehensive report of the top anomalies:

.. code-block:: python

    # Select top N anomalies
    top_n = 20
    top_anomaly_indices = sorted_idx[:top_n]
    
    # Create a DataFrame for the report
    if not metadata.empty:
        report_df = metadata.iloc[top_anomaly_indices].copy()
        report_df['log_likelihood'] = log_probs[top_anomaly_indices]
        report_df['rank'] = range(1, top_n + 1)
        
        # Save to CSV
        report_df.to_csv('anomaly_report.csv', index=False)
        print("Anomaly report saved to 'anomaly_report.csv'")
        
        # Display top 10
        print("\nTop 10 Anomalies:")
        display(report_df.head(10))
    else:
        # Just use indices if no metadata
        report_df = pd.DataFrame({
            'index': top_anomaly_indices,
            'log_likelihood': log_probs[top_anomaly_indices],
            'rank': range(1, top_n + 1)
        })
        report_df.to_csv('anomaly_report.csv', index=False)
        print("Anomaly report saved to 'anomaly_report.csv'")
        
        # Display top 10
        print("\nTop 10 Anomalies:")
        print(report_df.head(10))

Conclusion
---------

In this tutorial, we've covered:

1. Loading and applying a pre-trained probabilistic autoencoder to detect anomalies
2. Calculating and interpreting log-likelihood scores
3. Identifying anomalous patterns based on probabilistic thresholds
4. Visualizing original fields and their reconstructions
5. Comparing normal and anomalous patterns in latent space
6. Analyzing the geographical and temporal distribution of anomalies
7. Creating a comprehensive anomaly report

This workflow can be applied to detect unusual or interesting oceanographic phenomena such as eddies, fronts, upwelling events, or other patterns that deviate from typical ocean conditions.
