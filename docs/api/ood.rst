OOD Module API
============

Module: ``ulmo.ood``
-------------------

The ``ood`` module contains implementations for out-of-distribution (OOD) detection in oceanographic data, primarily using probabilistic autoencoders.

Classes
------

.. py:class:: ProbabilisticAutoencoder

   A probabilistic autoencoder for anomaly detection, combining an autoencoder with a normalizing flow.

   .. py:classmethod:: from_dict(model_dict, **kwargs)
   
      Instantiate the class from a dictionary.
      
      :param model_dict: Dictionary describing the model
      :type model_dict: dict
      :param kwargs: Additional keyword arguments for the constructor
      :return: PAE object
      :rtype: ProbabilisticAutoencoder

   .. py:classmethod:: from_json(json_file, **kwargs)
   
      Instantiate the class from a JSON file.
      
      :param json_file: Path to JSON file containing model description
      :type json_file: str
      :param kwargs: Additional keyword arguments for the constructor
      :return: PAE object
      :rtype: ProbabilisticAutoencoder

   .. py:method:: __init__(autoencoder, flow, filepath, datadir=None, logdir=None, device=None, skip_mkdir=False, write_model=True)
   
      Constructor.
      
      :param autoencoder: Autoencoder model for dimensionality reduction
      :type autoencoder: ulmo.models.Autoencoder
      :param flow: Flow model for likelihood estimation
      :type flow: ulmo.models.ConditionalFlow
      :param filepath: Path to HDF5 file with training/validation data
      :type filepath: str
      :param datadir: Directory for intermediate data (latents, log probs)
      :type datadir: str or None, optional
      :param logdir: Directory for logs and model files
      :type logdir: str or None, optional
      :param device: Device for computation (CPU/GPU)
      :type device: torch.device or None, optional
      :param skip_mkdir: Whether to skip creating directories
      :type skip_mkdir: bool, optional
      :param write_model: Whether to write model to JSON
      :type write_model: bool, optional

   .. py:method:: save_autoencoder()
   
      Save the autoencoder to a PyTorch file.

   .. py:method:: load_autoencoder()
   
      Load the autoencoder from a PyTorch file.

   .. py:method:: save_flow()
   
      Save the flow model to a PyTorch file.

   .. py:method:: load_flow()
   
      Load the flow model from a PyTorch file.

   .. py:method:: write_model()
   
      Write the model architecture to a JSON file.

   .. py:method:: train_autoencoder(n_epochs, batch_size, lr, summary_interval=50, eval_interval=500, show_plots=True, force_save=False)
   
      Train the autoencoder component.
      
      :param n_epochs: Number of training epochs
      :type n_epochs: int
      :param batch_size: Batch size for training
      :type batch_size: int
      :param lr: Learning rate
      :type lr: float
      :param summary_interval: Interval for logging training progress
      :type summary_interval: int, optional
      :param eval_interval: Interval for evaluating on validation data
      :type eval_interval: int, optional
      :param show_plots: Whether to show training plots
      :type show_plots: bool, optional
      :param force_save: Whether to force saving the model
      :type force_save: bool, optional

   .. py:method:: train_flow(n_epochs, batch_size, lr, summary_interval=50, eval_interval=500, show_plots=True, force_save=False)
   
      Train the normalizing flow component.
      
      :param n_epochs: Number of training epochs
      :type n_epochs: int
      :param batch_size: Batch size for training
      :type batch_size: int
      :param lr: Learning rate
      :type lr: float
      :param summary_interval: Interval for logging training progress
      :type summary_interval: int, optional
      :param eval_interval: Interval for evaluating on validation data
      :type eval_interval: int, optional
      :param show_plots: Whether to show training plots
      :type show_plots: bool, optional
      :param force_save: Whether to force saving the model
      :type force_save: bool, optional

   .. py:method:: _compute_latents()
   
      Compute latent vectors from the autoencoder for all data.

   .. py:method:: _compute_log_probs()
   
      Compute log-probability values in the flow space for all data.

   .. py:method:: _compute_flow_latents()
   
      Compute flow latent representations for all data.

   .. py:method:: load_meta(key, filename=None)
   
      Load metadata from the input file.
      
      :param key: Group name for metadata
      :type key: str
      :param filename: File for metadata (default: use self.filepath['data'])
      :type filename: str or None, optional
      :return: Metadata
      :rtype: pandas.DataFrame

   .. py:method:: encode(x)
   
      Encode data using the autoencoder.
      
      :param x: Input data
      :type x: numpy.ndarray or torch.Tensor
      :return: Encoded latent representation
      :rtype: Same type as input

   .. py:method:: decode(z)
   
      Decode latent representation to data space.
      
      :param z: Latent representation
      :type z: numpy.ndarray or torch.Tensor
      :return: Decoded data
      :rtype: Same type as input

   .. py:method:: reconstruct(x)
   
      Reconstruct data by encoding and then decoding.
      
      :param x: Input data
      :type x: numpy.ndarray or torch.Tensor
      :return: Reconstructed data
      :rtype: Same type as input

   .. py:method:: log_prob(x)
   
      Calculate log probability of data.
      
      :param x: Input data
      :type x: numpy.ndarray or torch.Tensor
      :return: Log probability
      :rtype: Same type as input

   .. py:method:: eval_numpy_img(img, **kwargs)
   
      Run Ulmo on an input numpy image.
      
      :param img: Input image
      :type img: numpy.ndarray
      :param kwargs: Additional keyword arguments
      :return: Tuple of (latent_vector, log_likelihood)
      :rtype: (numpy.ndarray, float)

   .. py:method:: eval_data_file(data_file, dataset, output_file, csv=False, **kwargs)
   
      Evaluate all images in the input data file.
      
      :param data_file: Path to preprocessed data file (.h5)
      :type data_file: str
      :param dataset: Dataset to analyze (e.g., 'valid')
      :type dataset: str
      :param output_file: Output file for results (.h5)
      :type output_file: str
      :param csv: Whether to write results to CSV
      :type csv: bool, optional
      :param kwargs: Additional keyword arguments
      :return: Log-likelihood values
      :rtype: numpy.ndarray

   .. py:method:: compute_log_probs(dset, num_workers=16, batch_size=1024, collate_fn=id_collate)
   
      Compute log probabilities on an input dataset.
      
      :param dset: Dataset containing images
      :type dset: torch.utils.data.Dataset
      :param num_workers: Number of worker processes
      :type num_workers: int, optional
      :param batch_size: Batch size for processing
      :type batch_size: int, optional
      :param collate_fn: Function for collating batches
      :type collate_fn: callable, optional
      :return: Tuple of (latents, log_probabilities)
      :rtype: (numpy.ndarray, numpy.ndarray)

   .. py:method:: save_log_probs()
   
      Write the log probabilities to a CSV file.

   .. py:method:: plot_reconstructions(save_figure=False, skipmeta=False)
   
      Generate a grid of plots of reconstructed images.
      
      :param save_figure: Whether to save the figure
      :type save_figure: bool, optional
      :param skipmeta: Whether to skip loading metadata
      :type skipmeta: bool, optional

   .. py:method:: plot_log_probs(sample_size=10000, save_figure=False, logdir=None)
   
      Plot distribution of log probabilities.
      
      :param sample_size: Number of samples to include
      :type sample_size: int, optional
      :param save_figure: Whether to save the figure
      :type save_figure: bool, optional
      :param logdir: Directory for saving figure
      :type logdir: str or None, optional

   .. py:method:: plot_grid(kind, save_metadata=False, save_figure=False, vmin=None, vmax=None, grad_vmin=None, grad_vmax=None)
   
      Plot a grid of fields based on their log-likelihood values.
      
      :param kind: Type of fields to plot ('outliers', 'inliers', 'midliers', 'most likely', 'least likely')
      :type kind: str
      :param save_metadata: Whether to save metadata for plotted fields
      :type save_metadata: bool, optional
      :param save_figure: Whether to save the figure
      :type save_figure: bool, optional
      :param vmin: Minimum value for field colormap
      :type vmin: float or None, optional
      :param vmax: Maximum value for field colormap
      :type vmax: float or None, optional
      :param grad_vmin: Minimum value for gradient colormap
      :type grad_vmin: float or None, optional
      :param grad_vmax: Maximum value for gradient colormap
      :type grad_vmax: float or None, optional

   .. py:method:: plot_geographical(save_figure=False)
   
      Create a geographical plot of log-likelihood values.
      
      :param save_figure: Whether to save the figure
      :type save_figure: bool, optional
