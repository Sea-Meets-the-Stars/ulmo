Visualization
=============

Ulmo provides specialized visualization tools for oceanographic data, focusing on both individual fields and global spatial distributions.

Field Visualization
-----------------

For visualizing individual temperature fields and their reconstructions:

.. code-block:: python

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    
    from ulmo.plotting import plotting
    
    # Load color palette and colormap
    pal, cm = plotting.load_palette()
    
    # Create figure and subplots
    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 2)
    
    # Plot original field
    ax1 = plt.subplot(gs[0])
    sns.heatmap(field, ax=ax1, xticklabels=[], yticklabels=[], 
                cmap=cm, vmin=-2, vmax=2)
    ax1.set_title("Original Field")
    
    # Plot reconstructed field
    ax2 = plt.subplot(gs[1])
    sns.heatmap(reconstruction, ax=ax2, xticklabels=[], yticklabels=[], 
                cmap=cm, vmin=-2, vmax=2)
    ax2.set_title("Reconstruction")
    
    plt.tight_layout()
    plt.show()

Grid Visualization
----------------

For comparing multiple fields in a grid layout:

.. code-block:: python

    from ulmo.plotting import grid_plot
    
    # Create a grid of plots (returns list of axis pairs)
    fig, axes = grid_plot(nrows=4, ncols=4)
    
    # Loop through axes and plot
    for i, (ax, r_ax) in enumerate(axes):
        # Original field
        ax.axis('equal')
        sns.heatmap(fields[i], ax=ax, xticklabels=[], yticklabels=[], 
                    cmap=cm, vmin=-2, vmax=2)
        
        # Reconstructed field
        r_ax.axis('equal')
        sns.heatmap(reconstructions[i], ax=r_ax, xticklabels=[], 
                    yticklabels=[], cmap=cm, vmin=-2, vmax=2)
    
    plt.tight_layout()
    plt.show()

Spatial Distribution Maps
-----------------------

For visualizing the global distribution of anomalies:

.. code-block:: python

    import cartopy.crs as ccrs
    import numpy as np
    from ulmo.spatial_plots import show_avg_LL
    
    # Create global map of mean log-likelihood values
    ax = show_avg_LL(
        main_tbl,          # DataFrame with lat, lon, LL columns
        nside=64,          # HEALPix resolution parameter
        use_mask=True,     # Mask areas with no data
        tricontour=False,  # Use scatter plot instead of contour
        color='viridis',   # Colormap
        figsize=(12, 8),   # Figure size
        show=True          # Display the plot
    )

The `show_avg_LL` function creates a map using HEALPix binning to visualize the spatial distribution of log-likelihood scores across the globe.

Comparing Different Datasets
--------------------------

For comparing the spatial distribution between different datasets:

.. code-block:: python

    from ulmo.spatial_plots import show_spatial_diff
    
    # Create map showing differences between two datasets
    ax = show_spatial_diff(
        tbl1,               # First DataFrame 
        tbl2,               # Second DataFrame
        nside=64,           # HEALPix resolution
        use_log=True,       # Use logarithmic scale
        use_mask=True,      # Mask areas with no data
        color='coolwarm',   # Colormap (cool=negative, warm=positive)
        figsize=(24, 16),   # Figure size
        show=True           # Display the figure
    )

Heatmaps with Median Values
-------------------------

For analyzing the spatial distribution of median log-likelihood values:

.. code-block:: python

    from ulmo.spatial_plots import show_med_LL
    
    # Create map of median log-likelihood values
    ax = show_med_LL(
        main_tbl,          # DataFrame with lat, lon, LL columns
        nside=64,          # HEALPix resolution
        use_mask=True,     # Mask areas with no data
        color='viridis',   # Colormap
        figsize=(12, 8),   # Figure size
        show=True          # Display the plot
    )

Time Series Visualization
-----------------------

For visualizing time series of anomaly scores:

.. code-block:: python

    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Convert datetime column to pandas datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Group by time and location
    df['month'] = df['datetime'].dt.month
    
    # Calculate statistics by region and time
    region_stats = df.groupby(['region', 'month']).agg({
        'LL': ['mean', 'median', 'std'],
        'is_anomaly': 'sum'
    })
    
    # Plot time series by region
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for region in df['region'].unique():
        data = region_stats.loc[region]
        ax.plot(data.index, data['LL']['mean'], label=region)
    
    ax.set_xlabel('Month')
    ax.set_ylabel('Mean Log-Likelihood')
    ax.set_title('Monthly Mean Log-Likelihood by Region')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

Customizing Visualizations
------------------------

You can customize the visualizations with additional parameters:

.. code-block:: python

    # Custom color palette
    from ulmo.plotting import load_palette
    
    # Use a different colormap
    pal, cm = load_palette(cmap='RdBu_r')
    
    # Add coastlines and grid to maps
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    
    ax = show_spatial_diff(tbl1, tbl2)
    
    # Add gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, 
        color='black', alpha=0.5, linestyle=':', draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_left = True
    gl.ylabels_right = False
    gl.xlines = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    # Add title and labels
    plt.title('Difference in Log-Likelihood Between Models', fontsize=18)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')

Saving Visualizations
-------------------

For saving visualizations to various formats:

.. code-block:: python

    # Save as PNG
    plt.savefig('anomaly_map.png', dpi=300, bbox_inches='tight')
    
    # Save as PDF for publication
    plt.savefig('anomaly_map.pdf', format='pdf', bbox_inches='tight')
    
    # Save as SVG for web
    plt.savefig('anomaly_map.svg', format='svg', bbox_inches='tight')
