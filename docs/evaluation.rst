Evaluation
==========

This section covers the evaluation of ocean data using Ulmo's probabilistic models, focusing on anomaly detection through log-likelihood scores and reconstruction analysis.

Log-Likelihood Evaluation
------------------------

The primary evaluation metric in Ulmo is the log-likelihood (LL) score, which quantifies how likely a particular pattern is according to the learned probability distribution:

.. code-block:: python

    from ulmo.models import io as model_io
    from ulmo.preproc import utils as pp_utils
    
    # Load model
    pae = model_io.load_ulmo_model('model-l2-std')
    
    # Preprocess data
    pp_field, meta = pp_utils.preproc_field(field, mask)
    
    # Calculate log-likelihood
    latent, log_likelihood = pae.eval_numpy_img(pp_field)
    
    # Lower log-likelihood values indicate more anomalous patterns
    print(f"Log-likelihood: {log_likelihood}")

Batch Evaluation
--------------

For evaluating multiple fields or complete datasets:

.. code-block:: python

    # Evaluate all fields in a preprocessed file
    log_probs = pae.eval_data_file(
        data_file='preprocessed_data.h5',
        dataset='valid',                 # Dataset to evaluate ('valid' or 'train')
        output_file='log_probs.h5',      # Output file for results
        csv=True                         # Also save results as CSV
    )
    
    # Or use the evaluation script
    # python -m ulmo.scripts.eval 2010,2012 std

Reconstruction Analysis
---------------------

Analyzing reconstructions can provide insights into what the model finds anomalous:

.. code-block:: python

    # Get both latent representation and reconstruction
    latent = pae.encode(pp_field)
    reconstruction = pae.decode(latent)
    
    # Or in one step
    reconstruction = pae.reconstruct(pp_field)
    
    # Calculate reconstruction error
    import numpy as np
    mse = np.mean((pp_field - reconstruction)**2)
    print(f"Reconstruction MSE: {mse}")

Visualizing Reconstructions
-------------------------

Visualization helps understand what features the model captures or misses:

.. code-block:: python

    import matplotlib.pyplot as plt
    import seaborn as sns
    from ulmo.plotting import plotting
    
    # Load color palette
    pal, cmap = plotting.load_palette()
    
    # Create a figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Original
    sns.heatmap(pp_field, ax=axes[0], cmap=cmap, vmin=-2, vmax=2)
    axes[0].set_title("Original Field")
    axes[0].axis('off')
    
    # Reconstruction
    sns.heatmap(reconstruction, ax=axes[1], cmap=cmap, vmin=-2, vmax=2)
    axes[1].set_title("Reconstruction")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

Distribution Analysis
-------------------

Analyze the distribution of log-likelihood scores:

.. code-block:: python

    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Load log-likelihood results
    df = pd.read_csv('log_probs.csv')
    
    # Plot histogram of log-likelihoods
    plt.figure(figsize=(10, 6))
    plt.hist(df['log_likelihood'], bins=50, alpha=0.7)
    plt.axvline(df['log_likelihood'].quantile(0.05), color='r', 
                linestyle='--', label='5% threshold')
    plt.xlabel('Log-Likelihood')
    plt.ylabel('Count')
    plt.title('Distribution of Log-Likelihood Scores')
    plt.legend()
    plt.show()
    
    # Find most anomalous examples
    anomalies = df.sort_values('log_likelihood').head(10)
    print(anomalies)

Spatial Analysis
--------------

Analyze the geographical distribution of anomalies:

.. code-block:: python

    from ulmo.spatial_plots import show_avg_LL
    
    # Create map of mean log-likelihood values
    ax = show_avg_LL(
        df,                 # DataFrame with lat, lon, and LL columns
        nside=64,           # HEALPix resolution
        color='viridis',    # Colormap
        show=True           # Display the figure
    )

Threshold Selection
-----------------

Determining appropriate thresholds for anomaly detection:

.. code-block:: python

    # Calculate percentile-based thresholds
    thresholds = {
        'extreme': df['log_likelihood'].quantile(0.01),
        'anomalous': df['log_likelihood'].quantile(0.05),
        'unusual': df['log_likelihood'].quantile(0.10)
    }
    
    # Flag anomalies based on thresholds
    for name, threshold in thresholds.items():
        df[f'is_{name}'] = df['log_likelihood'] < threshold
        count = df[f'is_{name}'].sum()
        print(f"{name.capitalize()}: {count} fields ({count/len(df)*100:.2f}%)")

Time Series Analysis
-----------------

For temporal datasets, analyze trends in anomaly detection:

.. code-block:: python

    # Convert datetime column to pandas datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Group by time period
    monthly = df.groupby(pd.Grouper(key='datetime', freq='M'))
    
    # Calculate statistics by time period
    time_stats = monthly.agg({
        'log_likelihood': ['mean', 'std', 'min'],
        'is_anomalous': 'sum'
    })
    
    # Plot time series of anomaly counts
    plt.figure(figsize=(12, 6))
    time_stats['is_anomalous']['sum'].plot()
    plt.title('Monthly Anomaly Count')
    plt.xlabel('Date')
    plt.ylabel('Number of Anomalies')
    plt.tight_layout()
    plt.show()

Feature Analysis
--------------

Examine the latent space to understand learned features:

.. code-block:: python

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    # Load latent vectors
    with h5py.File('latents.h5', 'r') as f:
        latents = f['latents'][:]
    
    # Apply dimensionality reduction for visualization
    pca = PCA(n_components=10)
    pca_result = pca.fit_transform(latents)
    
    # Visualize first two PCA components
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                c=log_probs, cmap='viridis', alpha=0.5)
    plt.colorbar(label='Log-Likelihood')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Latent Space Visualization (PCA)')
    plt.show()
    
    # Print explained variance
    print(f"Explained variance ratio: {pca.explained_variance_ratio_[:10]}")
    print(f"Total explained variance: {sum(pca.explained_variance_ratio_[:10]):.2f}")
