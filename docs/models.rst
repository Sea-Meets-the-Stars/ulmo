Models
======

Ulmo implements several deep learning models for analyzing oceanographic data, with the core architecture being a Probabilistic Autoencoder (PAE) that combines autoencoders and normalizing flows.

Probabilistic Autoencoder
------------------------

The Probabilistic Autoencoder is the main model architecture in Ulmo, created by combining:

1. A deep convolutional autoencoder for feature extraction and dimensionality reduction
2. A normalizing flow for density estimation in the latent space

This architecture enables Ulmo to:
- Learn compact representations of oceanographic data
- Model the probability distribution of these representations
- Identify out-of-distribution (anomalous) samples

.. code-block:: python

    from ulmo.ood import ProbabilisticAutoencoder
    from ulmo.models import DCAE, ConditionalFlow
    
    # Initialize components
    autoencoder = DCAE(
        image_shape=(1, 64, 64),
        latent_dim=512
    )
    
    flow = ConditionalFlow(
        dim=512,                    # Must match autoencoder's latent_dim
        context_dim=None,           # No conditioning
        transform_type='autoregressive',
        n_layers=10,
        hidden_units=256,
        n_blocks=2,
        dropout=0.2,
        use_batch_norm=False,
        tails='linear',
        tail_bound=10,
        n_bins=5,
        min_bin_height=1e-3,
        min_bin_width=1e-3,
        min_derivative=1e-3,
        unconditional_transform=False
    )
    
    # Combine into a PAE
    pae = ProbabilisticAutoencoder(
        autoencoder=autoencoder,
        flow=flow,
        filepath='data/training_data.h5',
        datadir='data/processed',
        logdir='logs'
    )

Deep Convolutional Autoencoder (DCAE)
-----------------------------------

The autoencoder component is a deep convolutional model that reduces image data to a latent representation:

.. code-block:: python

    from ulmo.models import DCAE
    
    # Create an autoencoder for 64x64 single-channel images
    # with 512-dimensional latent space
    autoencoder = DCAE(
        image_shape=(1, 64, 64),
        latent_dim=512
    )
    
    # Encode an image to latent space
    latent = autoencoder.encode(image)
    
    # Decode from latent space
    reconstruction = autoencoder.decode(latent)
    
    # Or do both at once
    reconstruction = autoencoder.reconstruct(image)

Architecture details:
- Encoder: Series of convolutional layers with batch normalization and LeakyReLU
- Linear bottleneck layer that encodes to latent space
- Decoder: Series of transposed convolutions to reconstruct the image

Normalizing Flow
--------------

The normalizing flow component models the probability density in latent space:

.. code-block:: python

    from ulmo.models import ConditionalFlow
    
    # Create a normalizing flow for density estimation
    flow = ConditionalFlow(
        dim=512,                  # Dimensionality of input (latent space)
        transform_type='autoregressive',
        n_layers=10,
        hidden_units=256,
        tails='linear',
        tail_bound=10
    )
    
    # Compute log probability of a latent vector
    log_prob = flow.log_prob(latent_vector)

The flow transforms the complex distribution in latent space to a simple base distribution (usually standard normal) through a series of invertible transformations.

Pre-trained Models
----------------

Ulmo provides pre-trained models for different oceanographic datasets:

.. code-block:: python

    from ulmo.models import io as model_io
    
    # Load pre-trained model for MODIS L2 data
    pae = model_io.load_ulmo_model('model-l2-std')
    
    # Available models:
    # - 'model-l2-std': Standard preprocessing for MODIS L2
    # - 'model-l2-loggrad': Gradient-based preprocessing for MODIS L2
    # - 'viirs-98': Model trained on VIIRS data with 98% clear cutouts

Model Training
------------

Training the PAE involves two phases:

1. Training the autoencoder:

.. code-block:: python

    # Train the autoencoder
    pae.train_autoencoder(
        n_epochs=100,
        batch_size=64,
        lr=1e-4
    )

2. Training the flow model on the encoded latent vectors:

.. code-block:: python

    # First compute latent vectors for all training data
    pae._compute_latents()
    
    # Then train the flow
    pae.train_flow(
        n_epochs=100,
        batch_size=256,
        lr=1e-4
    )

Once trained, the model components are saved to disk and can be loaded for inference.

Model Evaluation
--------------

Evaluating model performance on new data:

.. code-block:: python

    # Compute log probabilities for a dataset
    latents, log_probs = pae.compute_log_probs(dataset)
    
    # Evaluate a single image
    latent, log_prob = pae.eval_numpy_img(image)
    
    # Process a file of images
    log_probs = pae.eval_data_file(
        data_file='preprocessed_data.h5',
        dataset='valid',
        output_file='log_probs.h5'
    )

Low log probability values indicate anomalous patterns that are outside the distribution of the training data.

Advanced: Customizing Models
--------------------------

For specialized applications, you can create custom model architectures:

.. code-block:: python

    # Custom encoder/decoder architectures can be implemented
    # by subclassing the Autoencoder abstract base class
    
    from ulmo.models import Autoencoder
    import torch.nn as nn
    
    class CustomAutoencoder(Autoencoder, nn.Module):
        def __init__(self, custom_param1, custom_param2):
            super().__init__()
            # Define custom architecture
            
        def encode(self, x):
            # Custom encoding logic
            return z
            
        def decode(self, z):
            # Custom decoding logic
            return x
            
        def reconstruct(self, x):
            # Combine encode + decode
            z = self.encode(x)
            return self.decode(z)
