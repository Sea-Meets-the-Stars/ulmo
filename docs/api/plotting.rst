Plotting Module API
=================

Module: ``ulmo.plotting.plotting``
--------------------------------

The ``plotting`` module provides functions for creating visualizations of oceanographic data.

Functions
--------

.. py:function:: load_palette(cmap='cmo.thermal')

   Load a color palette and colormap for oceanographic data visualization.
   
   :param cmap: Name of the colormap to use
   :type cmap: str, optional
   :return: Tuple of (palette, colormap)
   :rtype: tuple

.. py:function:: grid_plot(nrows=4, ncols=4, figsize=None)

   Create a grid of subplots for comparing original and reconstructed fields.
   
   :param nrows: Number of rows in the grid
   :type nrows: int, optional
   :param ncols: Number of columns in the grid
   :type ncols: int, optional
   :param figsize: Figure size (width, height) in inches
   :type figsize: tuple or None, optional
   :return: Tuple of (figure, axes)
   :rtype: tuple

Module: ``ulmo.spatial_plots``
----------------------------

The ``spatial_plots`` module provides functions for creating global geographical visualizations.

Functions
--------

.. py:function:: evals_to_healpix(eval_tbl, nside, mask=True)

   Generate a HEALPix map of cutout locations and values.
   
   :param eval_tbl: Table of cutouts with lat, lon columns
   :type eval_tbl: pandas.DataFrame
   :param nside: HEALPix resolution parameter
   :type nside: int
   :param mask: Whether to mask areas with no data
   :type mask: bool, optional
   :return: Tuple of (count_map, longitudes, latitudes, value_map)
   :rtype: tuple

.. py:function:: show_avg_LL(main_tbl, nside=64, use_mask=True, tricontour=False, lbl=None, figsize=(12, 8), color='viridis', show=True)

   Generate a global map of mean log-likelihood values.
   
   :param main_tbl: Table of cutouts with lat, lon, LL columns
   :type main_tbl: pandas.DataFrame
   :param nside: HEALPix resolution parameter
   :type nside: int, optional
   :param use_mask: Whether to mask areas with no data
   :type use_mask: bool, optional
   :param tricontour: Whether to use contour plot instead of scatter
   :type tricontour: bool, optional
   :param lbl: Label for the colorbar
   :type lbl: str or None, optional
   :param figsize: Figure size (width, height) in inches
   :type figsize: tuple, optional
   :param color: Colormap name
   :type color: str, optional
   :param show: Whether to display the figure
   :type show: bool, optional
   :return: Matplotlib axis
   :rtype: matplotlib.axes.Axes

.. py:function:: evals_to_healpix_meds(eval_tbl, nside, mask=True)

   Generate a HEALPix map with median values for each pixel.
   
   :param eval_tbl: Table of cutouts with lat, lon columns
   :type eval_tbl: pandas.DataFrame
   :param nside: HEALPix resolution parameter
   :type nside: int
   :param mask: Whether to mask areas with no data
   :type mask: bool, optional
   :return: Tuple of (count_map, longitudes, latitudes, median_map)
   :rtype: tuple

.. py:function:: show_med_LL(main_tbl, nside=64, use_mask=True, tricontour=False, lbl=None, figsize=(12, 8), color='viridis', show=True)

   Generate a global map of median log-likelihood values.
   
   :param main_tbl: Table of cutouts with lat, lon, LL columns
   :type main_tbl: pandas.DataFrame
   :param nside: HEALPix resolution parameter
   :type nside: int, optional
   :param use_mask: Whether to mask areas with no data
   :type use_mask: bool, optional
   :param tricontour: Whether to use contour plot instead of scatter
   :type tricontour: bool, optional
   :param lbl: Label for the colorbar
   :type lbl: str or None, optional
   :param figsize: Figure size (width, height) in inches
   :type figsize: tuple, optional
   :param color: Colormap name
   :type color: str, optional
   :param show: Whether to display the figure
   :type show: bool, optional
   :return: Matplotlib axis
   :rtype: matplotlib.axes.Axes

.. py:function:: show_spatial_two_avg(tbl1, tbl2, nside=64, use_log=True, use_mask=True, tricontour=False, lbl=None, figsize=(12, 8), color='coolwarm', show=True)

   Generate a global map of the difference in mean log-likelihood between two datasets.
   
   :param tbl1: First table of cutouts
   :type tbl1: pandas.DataFrame
   :param tbl2: Second table of cutouts
   :type tbl2: pandas.DataFrame
   :param nside: HEALPix resolution parameter
   :type nside: int, optional
   :param use_log: Whether to use logarithmic scale
   :type use_log: bool, optional
   :param use_mask: Whether to mask areas with no data
   :type use_mask: bool, optional
   :param tricontour: Whether to use contour plot instead of scatter
   :type tricontour: bool, optional
   :param lbl: Label for the colorbar
   :type lbl: str or None, optional
   :param figsize: Figure size (width, height) in inches
   :type figsize: tuple, optional
   :param color: Colormap name
   :type color: str, optional
   :param show: Whether to display the figure
   :type show: bool, optional
   :return: Matplotlib axis
   :rtype: matplotlib.axes.Axes

.. py:function:: show_spatial_two_med(tbl1, tbl2, nside=64, use_mask=True, tricontour=False, lbl=None, figsize=(12, 8), color='coolwarm', show=True)

   Generate a global map of the difference in median log-likelihood between two datasets.
   
   :param tbl1: First table of cutouts
   :type tbl1: pandas.DataFrame
   :param tbl2: Second table of cutouts
   :type tbl2: pandas.DataFrame
   :param nside: HEALPix resolution parameter
   :type nside: int, optional
   :param use_mask: Whether to mask areas with no data
   :type use_mask: bool, optional
   :param tricontour: Whether to use contour plot instead of scatter
   :type tricontour: bool, optional
   :param lbl: Label for the colorbar
   :type lbl: str or None, optional
   :param figsize: Figure size (width, height) in inches
   :type figsize: tuple, optional
   :param color: Colormap name
   :type color: str, optional
   :param show: Whether to display the figure
   :type show: bool, optional
   :return: Matplotlib axis
   :rtype: matplotlib.axes.Axes

.. py:function:: scatter_diff_avg(tbl1, tbl2, nside=32, use_log=False, use_mask=True, tricontour=False, lbl=None, figsize=(12, 8), color='plasma', show=True)

   Create a scatter plot of the differences in mean log-likelihood vs. number of cutouts.
   
   :param tbl1: First table of cutouts
   :type tbl1: pandas.DataFrame
   :param tbl2: Second table of cutouts
   :type tbl2: pandas.DataFrame
   :param nside: HEALPix resolution parameter
   :type nside: int, optional
   :param use_log: Whether to use logarithmic scale
   :type use_log: bool, optional
   :param use_mask: Whether to mask areas with no data
   :type use_mask: bool, optional
   :param tricontour: Whether to use contour plot
   :type tricontour: bool, optional
   :param lbl: Label for the colorbar
   :type lbl: str or None, optional
   :param figsize: Figure size (width, height) in inches
   :type figsize: tuple, optional
   :param color: Colormap name
   :type color: str, optional
   :param show: Whether to display the figure
   :type show: bool, optional
   :return: Matplotlib axis
   :rtype: matplotlib.axes.Axes

Module: ``ulmo.figures``
---------------------

The ``figures`` module provides functions for creating standard visualizations.

Functions
--------

.. py:function:: show_spatial(main_tbl, nside=64, use_log=True, use_mask=True, tricontour=False, lbl=None, figsize=(12, 8), color='Reds', show=True)

   Generate a global map of cutout locations.
   
   :param main_tbl: Table of cutouts with lat, lon columns
   :type main_tbl: pandas.DataFrame
   :param nside: HEALPix resolution parameter
   :type nside: int, optional
   :param use_log: Whether to use logarithmic scale
   :type use_log: bool, optional
   :param use_mask: Whether to mask areas with no data
   :type use_mask: bool, optional
   :param tricontour: Whether to use contour plot instead of scatter
   :type tricontour: bool, optional
   :param lbl: Label for the colorbar
   :type lbl: str or None, optional
   :param figsize: Figure size (width, height) in inches
   :type figsize: tuple, optional
   :param color: Colormap name
   :type color: str, optional
   :param show: Whether to display the figure
   :type show: bool, optional
   :return: Matplotlib axis
   :rtype: matplotlib.axes.Axes
