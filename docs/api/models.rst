Models Module API
===============

Module: ``ulmo.models``
---------------------

The ``models`` module provides model implementations for Ulmo's deep learning components, including autoencoders and normalizing flows.

Classes
------

.. py:class:: ulmo.models.Autoencoder

   Abstract base class for implementing autoencoders.

   .. py:method:: encode(x)
   
      Abstract method to encode data.
      
      :param x: Input data (*, D) or (*, C, H, W)
      :type x: torch.Tensor
      :return: Latent representation (*, latent_dim)
      :rtype: torch.Tensor

   .. py:method:: decode(z)
   
      Abstract method to decode latent representation.
      
      :param z: Latent representation (*, latent_dim)
      :type z: torch.Tensor
      :return: Reconstructed data (*, D) or (*, C, H, W)
      :rtype: torch.Tensor

   .. py:method:: reconstruct(x)
   
      Abstract method to reconstruct input data.
      
      :param x: Input data (*, D) or (*, C, H, W)
      :type x: torch.Tensor
      :return: Reconstructed data (*, D) or (*, C, H, W)
      :rtype: torch.Tensor

.. py:class:: ulmo.models.DCAE

   A deep convolutional autoencoder.

   .. py:method:: __init__(image_shape, latent_dim)
   
      Constructor.
      
      :param image_shape: Tuple of (channels, width, height)
      :type image_shape: tuple
      :param latent_dim: Dimension of the latent space
      :type latent_dim: int

   .. py:method:: encode(x)
   
      Encode data using convolutional layers.
      
      :param x: Input data
      :type x: torch.Tensor
      :return: Latent representation
      :rtype: torch.Tensor

   .. py:method:: decode(z)
   
      Decode latent representation using transpose convolutions.
      
      :param z: Latent representation
      :type z: torch.Tensor
      :return: Reconstructed data
      :rtype: torch.Tensor

   .. py:method:: reconstruct(x)
   
      Reconstruct input data by encoding and decoding.
      
      :param x: Input data
      :type x: torch.Tensor
      :return: Reconstructed data
      :rtype: torch.Tensor

   .. py:method:: forward(x)
   
      Forward pass for training, computes reconstruction loss.
      
      :param x: Input data
      :type x: torch.Tensor
      :return: Mean squared error reconstruction loss
      :rtype: torch.Tensor

   .. py:classmethod:: from_file(f, device=None, **kwargs)
   
      Load model from file.
      
      :param f: File path to model weights
      :type f: str
      :param device: Computation device (CPU/GPU)
      :type device: torch.device or None, optional
      :param kwargs: Additional arguments for model initialization
      :return: Loaded model
      :rtype: DCAE

.. py:class:: ulmo.models.ConditionalFlow

   A conditional rational quadratic neural spline flow.

   .. py:method:: __init__(dim, context_dim, transform_type, n_layers, hidden_units, n_blocks, dropout, use_batch_norm, tails, tail_bound, n_bins, min_bin_height, min_bin_width, min_derivative, unconditional_transform, encoder=None)
   
      Constructor.
      
      :param dim: Dimension of the input data
      :type dim: int
      :param context_dim: Dimension of the conditioning variable (None for unconditional)
      :type context_dim: int or None
      :param transform_type: Type of transform ('coupling' or 'autoregressive')
      :type transform_type: str
      :param n_layers: Number of flow layers
      :type n_layers: int
      :param hidden_units: Number of hidden units in transform networks
      :type hidden_units: int
      :param n_blocks: Number of residual blocks in transform networks
      :type n_blocks: int
      :param dropout: Dropout probability
      :type dropout: float
      :param use_batch_norm: Whether to use batch normalization
      :type use_batch_norm: bool
      :param tails: Type of tails ('linear' or None)
      :type tails: str or None
      :param tail_bound: Bound for the tails
      :type tail_bound: float
      :param n_bins: Number of bins for the splines
      :type n_bins: int
      :param min_bin_height: Minimum bin height
      :type min_bin_height: float
      :param min_bin_width: Minimum bin width
      :type min_bin_width: float
      :param min_derivative: Minimum derivative
      :type min_derivative: float
      :param unconditional_transform: Whether to use unconditional transforms
      :type unconditional_transform: bool
      :param encoder: Optional encoder for context
      :type encoder: torch.nn.Module or None, optional

   .. py:method:: create_transform(type)
   
      Create an invertible rational quadratic transformation.
      
      :param type: Type of transform ('coupling' or 'autoregressive')
      :type type: str
      :return: Composite transform
      :rtype: transforms.CompositeTransform

   .. py:method:: log_prob(inputs, context=None)
   
      Forward pass in density estimation direction.
      
      :param inputs: Input data
      :type inputs: torch.Tensor
      :param context: Optional context for conditioning
      :type context: torch.Tensor or None, optional
      :return: Log probabilities
      :rtype: torch.Tensor

   .. py:method:: latent_and_prob(inputs, context=None)
   
      Forward pass that returns both latents and log probabilities.
      
      :param inputs: Input data
      :type inputs: torch.Tensor
      :param context: Optional context for conditioning
      :type context: torch.Tensor or None, optional
      :return: Tuple of (noise, logabsdet, log_prob)
      :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

   .. py:method:: forward(inputs, context=None)
   
      Forward pass that returns negative log likelihood.
      
      :param inputs: Input data
      :type inputs: torch.Tensor
      :param context: Optional context for conditioning
      :type context: torch.Tensor or None, optional
      :return: Negative log likelihood loss
      :rtype: torch.Tensor

   .. py:method:: sample(n_samples, context=None)
   
      Draw samples from the flow.
      
      :param n_samples: Number of samples to draw
      :type n_samples: int
      :param context: Optional context for conditioning
      :type context: torch.Tensor or None, optional
      :return: Tuple of (samples, log_prob)
      :rtype: tuple[torch.Tensor, torch.Tensor]

   .. py:method:: latent_representation(inputs, context=None)
   
      Get representations of data in latent space.
      
      :param inputs: Input data
      :type inputs: torch.Tensor
      :param context: Optional context for conditioning
      :type context: torch.Tensor or None, optional
      :return: Latent representations
      :rtype: torch.Tensor
