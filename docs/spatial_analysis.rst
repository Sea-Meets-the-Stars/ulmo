Spatial Analysis Tutorial
=======================

This tutorial covers techniques for analyzing and visualizing the spatial distribution of anomalies and ocean patterns using Ulmo.

Prerequisites
------------

Before starting, make sure you have:

1. Evaluated your dataset and have log-likelihood scores available
2. Metadata with latitude and longitude information
3. Installed required packages: cartopy, healpy (for HEALPix binning)

Loading Evaluation Results
------------------------

First, let's load some evaluation results:

.. code-block:: python

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import h5py
    
    # Load evaluation results from CSV file
    eval_results = pd.read_csv('evaluation_results.csv')
    
    # Alternatively, load from H5 file
    with h5py.File('evaluation_results.h5', 'r') as f:
        log_probs = f['valid'][:]
        
        # Load metadata if available
        if 'valid_metadata' in f:
            meta = f['valid_metadata']
            metadata = pd.DataFrame(meta[:].astype(np.unicode_), 
                                   columns=meta.attrs['columns'])
            
            # Add log-likelihood scores to metadata
            metadata['LL'] = log_probs
    
    print(f"Loaded {len(metadata)} evaluation results")
    print(f"Columns available: {metadata.columns.tolist()}")

Global Distribution of Log-Likelihood
----------------------------------

Let's visualize the global distribution of log-likelihood scores:

.. code-block:: python

    import cartopy.crs as ccrs
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    from ulmo.spatial_plots import show_avg_LL
    
    # Create a global map of average log-likelihood
    plt.figure(figsize=(12, 8))
    
    # Use Ulmo's spatial plotting tools
    ax = show_avg_LL(
        metadata,         # DataFrame with lat, lon, LL columns
        nside=64,         # HEALPix resolution parameter
        use_mask=True,    # Mask areas with no data
        tricontour=False, # Use scatter plot (not contour)
        color='viridis',  # Colormap
        show=False        # Don't display yet (we'll add more elements)
    )
    
    # Add title
    plt.title('Global Distribution of Log-Likelihood Scores', fontsize=16)
    
    # Add grid lines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, 
                     color='black', alpha=0.3, linestyle=':', draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_left = True
    gl.ylabels_right = False
    gl.xlines = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    plt.tight_layout()
    plt.show()

Median Log-Likelihood Maps
------------------------

Let's create a map of median log-likelihood values, which is less sensitive to outliers:

.. code-block:: python

    from ulmo.spatial_plots import show_med_LL
    
    # Create a global map of median log-likelihood
    plt.figure(figsize=(12, 8))
    
    ax = show_med_LL(
        metadata,        # DataFrame with lat, lon, LL columns
        nside=64,        # HEALPix resolution parameter
        use_mask=True,   # Mask areas with no data
        color='viridis', # Colormap
        show=False       # Don't display yet
    )
    
    plt.title('Global Distribution of Median Log-Likelihood Scores', fontsize=16)
    
    # Add grid lines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, 
                     color='black', alpha=0.3, linestyle=':', draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_left = True
    gl.ylabels_right = False
    gl.xlines = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    plt.tight_layout()
    plt.show()

Analyzing Anomaly Hotspots
------------------------

Let's identify and analyze regions with high concentrations of anomalies:

.. code-block:: python

    # Flag anomalies (e.g., bottom 5% log-likelihood)
    anomaly_threshold = np.percentile(metadata['LL'], 5)
    metadata['is_anomaly'] = metadata['LL'] < anomaly_threshold
    
    # Count anomalies per grid cell
    from ulmo.spatial_plots import evals_to_healpix
    
    # Only keep anomalous datapoints
    anomaly_df = metadata[metadata['is_anomaly']].copy()
    
    # Get HEALPix map of anomaly counts
    hp_events, hp_lons, hp_lats = evals_to_healpix(
        anomaly_df, 
        nside=64, 
        log=True,  # Use logarithmic scale
        mask=True  # Mask areas with no data
    )
    
    # Plot anomaly hotspots
    plt.figure(figsize=(12, 8))
    
    ax = plt.axes(projection=ccrs.Mollweide())
    ax.coastlines(zorder=10)
    ax.set_global()
    
    # Only plot cells with data
    good = np.invert(hp_events.mask)
    img = plt.scatter(
        x=hp_lons[good],
        y=hp_lats[good],
        c=hp_events[good],
        cmap='Reds',
        s=10,
        transform=ccrs.PlateCarree()
    )
    
    # Add colorbar
    cbar = plt.colorbar(img, orientation='horizontal', pad=0.05, fraction=0.05)
    cbar.set_label('Log Count of Anomalies', fontsize=12)
    
    plt.title('Global Distribution of Anomaly Hotspots', fontsize=16)
    
    # Add grid lines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, 
                     color='black', alpha=0.3, linestyle=':', draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_left = True
    gl.ylabels_right = False
    gl.xlines = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    plt.tight_layout()
    plt.show()

Comparing Models or Time Periods
-----------------------------

Let's compare log-likelihood distributions between different models or time periods:

.. code-block:: python

    from ulmo.spatial_plots import show_spatial_diff
    
    # Load another dataset for comparison
    comparison_df = pd.read_csv('comparison_results.csv')
    
    # Create a global map of the difference in log-likelihood
    plt.figure(figsize=(12, 8))
    
    ax = show_spatial_diff(
        metadata,         # First DataFrame
        comparison_df,    # Second DataFrame
        nside=64,         # HEALPix resolution
        use_mask=True,    # Mask areas with no data
        color='coolwarm', # Diverging colormap (red=positive, blue=negative)
        show=False        # Don't display yet
    )
    
    plt.title('Difference in Log-Likelihood between Datasets', fontsize=16)
    
    # Add grid lines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, 
                     color='black', alpha=0.3, linestyle=':', draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_left = True
    gl.ylabels_right = False
    gl.xlines = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    plt.tight_layout()
    plt.show()

Regional Analysis
---------------

Let's focus on specific regions for a more detailed analysis:

.. code-block:: python

    # Define regions of interest
    regions = {
        'North Atlantic': {'lat_min': 30, 'lat_max': 60, 'lon_min': -80, 'lon_max': 0},
        'Equatorial Pacific': {'lat_min': -10, 'lat_max': 10, 'lon_min': 150, 'lon_max': 270},
        'Southern Ocean': {'lat_min': -70, 'lat_max': -40, 'lon_min': 0, 'lon_max': 360}
    }
    
    # Function to check if a point is in a region
    def in_region(lat, lon, region):
        if region['lon_min'] <= region['lon_max']:
            return (region['lat_min'] <= lat <= region['lat_max'] and 
                    region['lon_min'] <= lon <= region['lon_max'])
        else:  # Handle regions that cross the dateline
            return (region['lat_min'] <= lat <= region['lat_max'] and 
                    (lon >= region['lon_min'] or lon <= region['lon_max']))
    
    # Add region labels to the metadata
    metadata['region'] = 'Other'
    for region_name, region_bounds in regions.items():
        mask = metadata.apply(lambda row: in_region(row['latitude'], row['longitude'], region_bounds), axis=1)
        metadata.loc[mask, 'region'] = region_name
    
    # Plot histograms of log-likelihood by region
    plt.figure(figsize=(12, 8))
    
    for i, (region_name, region_data) in enumerate(metadata.groupby('region')):
        if region_name == 'Other':
            continue
            
        plt.subplot(len(regions), 1, i+1)
        plt.hist(region_data['LL'], bins=30, alpha=0.7, label=region_name)
        plt.axvline(np.percentile(region_data['LL'], 5), color='r', linestyle='--',
                   label='5th percentile')
        plt.title(f'Log-Likelihood Distribution: {region_name}')
        plt.xlabel('Log-Likelihood')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Compare anomaly rates by region
    region_stats = metadata.groupby('region').agg({
        'is_anomaly': 'mean',
        'LL': ['mean', 'std', 'min', 'count']
    })
    
    # Convert anomaly rate to percentage
    region_stats['is_anomaly'] = region_stats['is_anomaly'] * 100
    
    # Print statistics
    print("Regional Statistics:")
    print(region_stats)
    
    # Plot bar chart of anomaly percentages by region
    plt.figure(figsize=(10, 6))
    bars = plt.bar(region_stats.index, region_stats['is_anomaly'], color='skyblue')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}%', ha='center', va='bottom')
    
    plt.title('Percentage of Anomalies by Region')
    plt.ylabel('Anomaly Percentage')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

Time-Space Analysis
----------------

If you have time information, let's analyze spatiotemporal patterns:

.. code-block:: python

    # Assuming metadata has a 'datetime' column
    if 'datetime' in metadata.columns:
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_dtype(metadata['datetime']):
            metadata['datetime'] = pd.to_datetime(metadata['datetime'])
        
        # Add month and year columns
        metadata['month'] = metadata['datetime'].dt.month
        metadata['year'] = metadata['datetime'].dt.year
        
        # Plot anomaly rate by region and month
        plt.figure(figsize=(12, 8))
        
        # Group by region and month
        region_month_stats = metadata.groupby(['region', 'month']).agg({
            'is_anomaly': 'mean',
            'LL': ['mean', 'count']
        })
        
        # Convert to percentage
        region_month_stats['is_anomaly'] = region_month_stats['is_anomaly'] * 100
        
        # Plot for each region
        for i, region_name in enumerate(regions.keys()):
            if region_name not in region_month_stats.index.get_level_values('region'):
                continue
                
            plt.subplot(len(regions), 1, i+1)
            
            data = region_month_stats.loc[region_name]
            plt.plot(data.index, data['is_anomaly'], 'o-', linewidth=2)
            
            plt.title(f'Monthly Anomaly Rate: {region_name}')
            plt.xlabel('Month')
            plt.ylabel('Anomaly Percentage')
            plt.xticks(range(1, 13))
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

Creating Custom Spatial Visualizations
-----------------------------------

Let's create some custom visualizations for specific analysis needs:

.. code-block:: python

    import numpy as np
    import healpy as hp
    
    # Function to create a custom spatial visualization
    def plot_spatial_metric(df, metric_column, title, cmap='viridis', vmin=None, vmax=None):
        """
        Create a spatial plot of a given metric.
        
        Args:
            df: DataFrame with lat, lon columns
            metric_column: Column name for the metric to plot
            title: Plot title
            cmap: Colormap name
            vmin, vmax: Optional value range for colormap
        """
        # Create HEALPix map
        nside = 64
        npix = hp.nside2npix(nside)
        
        # Initialize map
        hpx_map = np.zeros(npix) + hp.UNSEEN
        
        # Convert lat/lon to HEALPix indices
        theta = (90 - df['latitude']) * np.pi / 180
        phi = df['longitude'] * np.pi / 180
        indices = hp.ang2pix(nside, theta, phi)
        
        # For each pixel, calculate the mean of the metric
        for idx in np.unique(indices):
            values = df.loc[indices == idx, metric_column]
            if len(values) > 0:
                hpx_map[idx] = np.mean(values)
        
        # Create masked array
        hpx_map = np.ma.masked_values(hpx_map, hp.UNSEEN)
        
        # Get pixel coordinates for plotting
        nside = hp.npix2nside(len(hpx_map))
        ipix = np.arange(len(hpx_map))
        theta, phi = hp.pix2ang(nside, ipix)
        
        # Convert to lon/lat
        lon = phi * 180 / np.pi
        lat = 90 - theta * 180 / np.pi
        
        # Create figure
        plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.Mollweide())
        ax.coastlines(zorder=10)
        ax.set_global()
        
        # Plot only unmasked pixels
        mask = ~hpx_map.mask
        scatter = ax.scatter(
            lon[mask], lat[mask],
            c=hpx_map[mask],
            cmap=cmap,
            s=15,
            transform=ccrs.PlateCarree(),
            vmin=vmin,
            vmax=vmax
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, orientation='horizontal', pad=0.05, fraction=0.05)
        cbar.set_label(metric_column)
        
        # Add title
        plt.title(title, fontsize=16)
        
        # Add grid lines
        gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, 
                         color='black', alpha=0.3, linestyle=':', draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_left = True
        gl.ylabels_right = False
        gl.xlines = True
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        
        plt.tight_layout()
        plt.show()
    
    # Calculate some custom metrics
    metadata['anomaly_score'] = -metadata['LL']  # Invert LL for easier interpretation
    
    # If we have reconstruction data
    if 'mse' in metadata.columns:
        # Plot reconstruction error
        plot_spatial_metric(
            metadata, 
            'mse', 
            'Global Distribution of Reconstruction Error',
            cmap='Reds'
        )
    
    # Plot anomaly score
    plot_spatial_metric(
        metadata, 
        'anomaly_score', 
        'Global Distribution of Anomaly Scores',
        cmap='viridis'
    )

Rossby Radius Analysis
-------------------

Let's