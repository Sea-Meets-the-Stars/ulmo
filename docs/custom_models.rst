Custom Models Tutorial
====================

This tutorial shows how to create, train, and evaluate custom models for specific oceanographic applications using Ulmo.

Understanding Ulmo's Model Architecture
-------------------------------------

Ulmo's probabilistic autoencoder (PAE) architecture consists of two main components:

1. **Autoencoder**: Compresses data to a latent representation and reconstructs it
2. **Normalizing Flow**: Models the probability distribution in latent space

You can customize either or both components for your specific application.

Creating a Custom Autoencoder
---------------------------

Let's start by creating a custom autoencoder:

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from ulmo.models import Autoencoder
    
    class CustomAutoencoder(Autoencoder, nn.Module):
        """A custom autoencoder with a different architecture."""
        
        def __init__(self, image_shape=(1, 64, 64), latent_dim=256):
            super().__init__()
            
            self.c, self.w, self.h = image_shape
            self.latent_dim = latent_dim
            
            # Encoder layers
            self.e_conv1 = nn.Conv2d(self.c, 32, kernel_size=4, stride=2, padding=1)
            self.e_bn1 = nn.BatchNorm2d(32)
            self.e_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
            self.e_bn2 = nn.BatchNorm2d(64)
            self.e_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
            self.e_bn3 = nn.BatchNorm2d(128)
            self.e_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
            self.e_bn4 = nn.BatchNorm2d(256)
            
            # Compute the size after convolutions
            conv_out_size = self.w // 16  # After 4 layers with stride 2
            self.conv_out_dim = 256 * conv_out_size * conv_out_size
            
            # Latent representation
            self.e_fc = nn.Linear(self.conv_out_dim, latent_dim)
            
            # Decoder input
            self.d_fc = nn.Linear(latent_dim, self.conv_out_dim)
            
            # Decoder layers
            self.d_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
            self.d_bn1 = nn.BatchNorm2d(128)
            self.d_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
            self.d_bn2 = nn.BatchNorm2d(64)
            self.d_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
            self.d_bn3 = nn.BatchNorm2d(32)
            self.d_conv4 = nn.ConvTranspose2d(32, self.c, kernel_size=4, stride=2, padding=1)
        
        def encode(self, x):
            """Encode input to latent representation."""
            # Apply encoder layers
            x = F.leaky_relu(self.e_bn1(self.e_conv1(x)), 0.2)
            x = F.leaky_relu(self.e_bn2(self.e_conv2(x)), 0.2)
            x = F.leaky_relu(self.e_bn3(self.e_conv3(x)), 0.2)
            x = F.leaky_relu(self.e_bn4(self.e_conv4(x)), 0.2)
            
            # Flatten and project to latent space
            x = x.view(x.size(0), -1)
            z = self.e_fc(x)
            
            return z
        
        def decode(self, z):
            """Decode latent representation to reconstructed input."""
            # Project and reshape
            x = self.d_fc(z)
            x = x.view(x.size(0), 256, self.w // 16, self.h // 16)
            
            # Apply decoder layers
            x = F.leaky_relu(self.d_bn1(self.d_conv1(x)), 0.2)
            x = F.leaky_relu(self.d_bn2(self.d_conv2(x)), 0.2)
            x = F.leaky_relu(self.d_bn3(self.d_conv3(x)), 0.2)
            x = torch.tanh(self.d_conv4(x))
            
            return x
        
        def reconstruct(self, x):
            """Reconstruct input by encoding and decoding."""
            z = self.encode(x)
            return self.decode(z)
        
        def forward(self, x):
            """Forward pass for training (reconstruction loss)."""
            reconstructed = self.reconstruct(x)
            return F.mse_loss(reconstructed, x)

Creating a Custom Flow Model
--------------------------

Now let's create a custom flow model:

.. code-block:: python

    from ulmo.models import ConditionalFlow
    
    # Create a custom flow with different parameters
    custom_flow = ConditionalFlow(
        dim=256,                     # Must match autoencoder's latent_dim
        context_dim=None,            # No conditioning
        transform_type='coupling',   # Use coupling transforms (alternative: 'autoregressive')
        n_layers=12,                 # More layers for flexibility
        hidden_units=512,            # Larger capacity
        n_blocks=3,                  # More residual blocks
        dropout=0.1,                 # Dropout for regularization
        use_batch_norm=True,         # Use batch normalization
        tails='linear',              # Linear tails for unbounded data
        tail_bound=8.0,              # Larger tail bound
        n_bins=16,                   # More bins for better resolution
        min_bin_height=1e-4,         # Adjusted bin parameters
        min_bin_width=1e-4,
        min_derivative=1e-4,
        unconditional_transform=True
    )

Combining into a Probabilistic Autoencoder
---------------------------------------

Now we can combine the custom components into a PAE:

.. code-block:: python

    from ulmo.ood import ProbabilisticAutoencoder
    
    # Create the autoencoder
    autoencoder = CustomAutoencoder(
        image_shape=(1, 64, 64),
        latent_dim=256
    )
    
    # Create the flow model
    flow = custom_flow  # From above
    
    # Combine into a PAE
    custom_pae = ProbabilisticAutoencoder(
        autoencoder=autoencoder,
        flow=flow,
        filepath='data/training_data.h5',
        datadir='data/processed',
        logdir='logs/custom_model',
        skip_mkdir=False
    )
    
    # Print model summary
    print("Custom PAE created:")
    print(f"Autoencoder latent dimension: {custom_pae.autoencoder.latent_dim}")
    print(f"Flow model type: {custom_pae.flow.transform_type}")
    print(f"Flow model layers: {custom_pae.flow.n_layers}")

Training the Custom Model
-----------------------

Let's train our custom model:

.. code-block:: python

    # Train the autoencoder
    custom_pae.train_autoencoder(
        n_epochs=50,
        batch_size=64,
        lr=1e-4,
        summary_interval=10,
        eval_interval=100,
        show_plots=True
    )
    
    # Compute latent representations
    custom_pae._compute_latents()
    
    # Train the flow model
    custom_pae.train_flow(
        n_epochs=100,
        batch_size=128,
        lr=5e-5,
        summary_interval=10,
        eval_interval=100,
        show_plots=True
    )
    
    # Save the model components
    custom_pae.save_autoencoder()
    custom_pae.save_flow()
    custom_pae.write_model()
    
    print("Custom model trained and saved.")

Specialized Features for Oceanographic Data
----------------------------------------

Let's add some specialized features for oceanographic data:

.. code-block:: python

    import numpy as np
    import torch.nn as nn
    from ulmo.models import Autoencoder
    
    class SSFrontAutoencoder(Autoencoder, nn.Module):
        """
        An autoencoder specialized for sea surface temperature fronts.
        Uses gradient-aware convolutional layers.
        """
        
        def __init__(self, image_shape=(1, 64, 64), latent_dim=512):
            super().__init__()
            self.c, self.w, self.h = image_shape
            self.latent_dim = latent_dim
            
            # Specialized gradient extraction layers
            self.sobel_x = nn.Conv2d(self.c, self.c, kernel_size=3, padding=1, bias=False)
            self.sobel_y = nn.Conv2d(self.c, self.c, kernel_size=3, padding=1, bias=False)
            
            # Initialize Sobel filters (gradient extraction)
            sobel_x = torch.tensor([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ], dtype=torch.float32).reshape(1, 1, 3, 3)
            sobel_y = torch.tensor([
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ], dtype=torch.float32).reshape(1, 1, 3, 3)
            
            # Repeat for multi-channel if needed
            sobel_x = sobel_x.repeat(self.c, 1, 1, 1)
            sobel_y = sobel_y.repeat(self.c, 1, 1, 1)
            
            # Set as non-trainable weights
            self.sobel_x.weight = nn.Parameter(sobel_x, requires_grad=False)
            self.sobel_y.weight = nn.Parameter(sobel_y, requires_grad=False)
            
            # Encoder layers for gradient magnitude branch
            self.grad_conv1 = nn.Conv2d(self.c, 32, kernel_size=3, stride=2, padding=1)
            self.grad_bn1 = nn.BatchNorm2d(32)
            self.grad_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            self.grad_bn2 = nn.BatchNorm2d(64)
            
            # Encoder layers for normal branch
            self.e_conv1 = nn.Conv2d(self.c, 32, kernel_size=3, stride=2, padding=1)
            self.e_bn1 = nn.BatchNorm2d(32)
            self.e_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            self.e_bn2 = nn.BatchNorm2d(64)
            
            # Merged encoder layers
            self.e_conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
            self.e_bn3 = nn.BatchNorm2d(128)
            self.e_conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
            self.e_bn4 = nn.BatchNorm2d(256)
            
            # Calculate feature map size after convolutions
            conv_size = self.w // 16
            flattened_size = 256 * conv_size * conv_size
            
            # Latent projection
            self.e_fc = nn.Linear(flattened_size, latent_dim)
            
            # Decoder layers
            self.d_fc = nn.Linear(latent_dim, flattened_size)
            self.d_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
            self.d_bn1 = nn.BatchNorm2d(128)
            self.d_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
            self.d_bn2 = nn.BatchNorm2d(64)
            self.d_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
            self.d_bn3 = nn.BatchNorm2d(32)
            self.d_conv4 = nn.ConvTranspose2d(32, self.c, kernel_size=4, stride=2, padding=1)
        
        def encode(self, x):
            # Extract gradients
            grad_x = self.sobel_x(x)
            grad_y = self.sobel_y(x)
            grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
            
            # Process gradient branch
            g = F.relu(self.grad_bn1(self.grad_conv1(grad_mag)))
            g = F.relu(self.grad_bn2(self.grad_conv2(g)))
            
            # Process normal branch
            h = F.relu(self.e_bn1(self.e_conv1(x)))
            h = F.relu(self.e_bn2(self.e_conv2(h)))
            
            # Merge branches
            combined = torch.cat([g, h], dim=1)
            
            # Continue encoding
            x = F.relu(self.e_bn3(self.e_conv3(combined)))
            x = F.relu(self.e_bn4(self.e_conv4(x)))
            
            # Flatten and project to latent space
            x = x.view(x.size(0), -1)
            z = self.e_fc(x)
            
            return z
        
        def decode(self, z):
            # Project and reshape
            x = self.d_fc(z)
            x = x.view(x.size(0), 256, self.w // 16, self.h // 16)
            
            # Decoder layers
            x = F.relu(self.d_bn1(self.d_conv1(x)))
            x = F.relu(self.d_bn2(self.d_conv2(x)))
            x = F.relu(self.d_bn3(self.d_conv3(x)))
            x = torch.tanh(self.d_conv4(x))
            
            return x
        
        def reconstruct(self, x):
            z = self.encode(x)
            return self.decode(z)
        
        def forward(self, x):
            reconstructed = self.reconstruct(x)
            return F.mse_loss(reconstructed, x)

Evaluating the Custom Model
-------------------------

Let's compare our custom model with the standard model:

.. code-block:: python

    import matplotlib.pyplot as plt
    import seaborn as sns
    from ulmo.plotting import plotting
    from ulmo.models import io as model_io
    import h5py
    
    # Load the standard model for comparison
    standard_pae = model_io.load_ulmo_model('model-l2-std')
    
    # Load test data
    with h5py.File('test_data.h5', 'r') as f:
        test_fields = f['valid'][:50]  # Get 50 test fields
        
        # Ensure proper dimensions
        if len(test_fields.shape) == 3:
            test_fields = test_fields.reshape(test_fields.shape[0], 1, 
                                             test_fields.shape[1], test_fields.shape[2])
    
    # Convert to PyTorch tensor
    test_tensor = torch.from_numpy(test_fields).float().to(device)
    
    # Evaluate with both models
    with torch.no_grad():
        # Standard model
        std_latents = standard_pae.encode(test_tensor)
        std_log_probs = standard_pae.log_prob(test_tensor)
        std_recon = standard_pae.reconstruct(test_tensor)
        
        # Custom model
        custom_latents = custom_pae.encode(test_tensor)
        custom_log_probs = custom_pae.log_prob(test_tensor)
        custom_recon = custom_pae.reconstruct(test_tensor)
    
    # Convert to numpy
    std_log_probs = std_log_probs.cpu().numpy()
    custom_log_probs = custom_log_probs.cpu().numpy()
    std_recon = std_recon.cpu().detach().numpy()
    custom_recon = custom_recon.cpu().detach().numpy()
    
    # Calculate MSE for both models
    std_mse = np.mean([(std_recon[i, 0] - test_fields[i, 0])**2 
                      for i in range(len(test_fields))])
    custom_mse = np.mean([(custom_recon[i, 0] - test_fields[i, 0])**2 
                         for i in range(len(test_fields))])
    
    print(f"Standard model MSE: {std_mse:.6f}")
    print(f"Custom model MSE: {custom_mse:.6f}")
    print(f"MSE improvement: {(1 - custom_mse/std_mse)*100:.2f}%")
    
    # Compare log-likelihood distributions
    plt.figure(figsize=(10, 6))
    sns.kdeplot(std_log_probs, label='Standard Model')
    sns.kdeplot(custom_log_probs, label='Custom Model')
    plt.xlabel('Log-Likelihood')
    plt.ylabel('Density')
    plt.title('Comparison of Log-Likelihood Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Compare reconstructions for one example
    example_idx = 0  # Choose first example
    
    # Get color palette
    pal, cmap = plotting.load_palette()
    
    plt.figure(figsize=(15, 5))
    
    # Original
    plt.subplot(131)
    sns.heatmap(test_fields[example_idx, 0], cmap=cmap, vmin=-2, vmax=2)
    plt.title('Original Field')
    plt.axis('off')
    
    # Standard model reconstruction
    plt.subplot(132)
    sns.heatmap(std_recon[example_idx, 0], cmap=cmap, vmin=-2, vmax=2)
    plt.title(f'Standard Model\nLL: {std_log_probs[example_idx]:.2f}')
    plt.axis('off')
    
    # Custom model reconstruction
    plt.subplot(133)
    sns.heatmap(custom_recon[example_idx, 0], cmap=cmap, vmin=-2, vmax=2)
    plt.title(f'Custom Model\nLL: {custom_log_probs[example_idx]:.2f}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

Saving and Loading Custom Models
-----------------------------

Let's save and load our custom model:

.. code-block:: python

    import os
    import json
    
    # Create model directory
    os.makedirs('custom_model', exist_ok=True)
    
    # Save model components
    torch.save(autoencoder.state_dict(), 'custom_model/autoencoder.pt')
    torch.save(flow.state_dict(), 'custom_model/flow.pt')
    
    # Save model architecture as JSON
    model_dict = {
        'AE': {
            'image_shape': (autoencoder.c, autoencoder.w, autoencoder.h),
            'latent_dim': autoencoder.latent_dim,
            'type': 'SSFrontAutoencoder'  # Custom type
        },
        'flow': {
            'dim': flow.dim,
            'context_dim': flow.context_dim,
            'transform_type': flow.transform_type,
            'n_layers': flow.n_layers,
            'hidden_units': flow.hidden_units,
            'n_blocks': flow.n_blocks,
            'dropout': flow.dropout,
            'use_batch_norm': flow.use_batch_norm,
            'tails': flow.tails,
            'tail_bound': flow.tail_bound,
            'n_bins': flow.n_bins,
            'min_bin_height': flow.min_bin_height,
            'min_bin_width': flow.min_bin_width,
            'min_derivative': flow.min_derivative,
            'unconditional_transform': flow.unconditional_transform,
            'encoder': flow.encoder
        }
    }
    
    with open('custom_model/model.json', 'w') as f:
        json.dump(model_dict, f, indent=4)
    
    print("Custom model saved to 'custom_model/' directory")
    
    # Custom function to load the model
    def load_custom_model(model_dir):
        # Load model architecture
        with open(os.path.join(model_dir, 'model.json'), 'r') as f:
            model_dict = json.load(f)
        
        # Create autoencoder based on type
        ae_type = model_dict['AE'].get('type', 'DCAE')
        if ae_type == 'SSFrontAutoencoder':
            autoencoder = SSFrontAutoencoder(
                image_shape=tuple(model_dict['AE']['image_shape']),
                latent_dim=model_dict['AE']['latent_dim']
            )
        else:
            # Default to standard DCAE
            from ulmo.models import DCAE
            autoencoder = DCAE(
                image_shape=tuple(model_dict['AE']['image_shape']),
                latent_dim=model_dict['AE']['latent_dim']
            )
        
        # Create flow model
        flow = ConditionalFlow(**model_dict['flow'])
        
        # Load weights
        autoencoder.load_state_dict(torch.load(os.path.join(model_dir, 'autoencoder.pt'), 
                                              map_location=device))
        flow.load_state_dict(torch.load(os.path.join(model_dir, 'flow.pt'), 
                                        map_location=device))
        
        # Create PAE
        custom_pae = ProbabilisticAutoencoder(
            autoencoder=autoencoder,
            flow=flow,
            filepath='dummy.h5',  # Not used after loading
            skip_mkdir=True,
            write_model=False
        )
        
        return custom_pae
    
    # Load the saved model
    loaded_model = load_custom_model('custom_model')
    print("Custom model loaded successfully")
    
    # Verify it works the same
    with torch.no_grad():
        test_sample = test_tensor[0:1]  # Just one sample
        original_output = custom_pae.log_prob(test_sample)
        loaded_output = loaded_model.log_prob(test_sample)
        
        print(f"Original model output: {original_output.item():.6f}")
        print(f"Loaded model output: {loaded_output.item():.6f}")
        print(f"Difference: {abs(original_output.item() - loaded_output.item()):.6f}")

Extracting Features from Custom Models
-----------------------------------

You can use custom models to extract features from oceanographic data:

.. code-block:: python

    def extract_features(model, fields):
        """
        Extract features from a dataset using a trained model.
        
        Args:
            model: PAE model
            fields: Dataset fields
            
        Returns:
            Dictionary of features
        """
        device = next(model.parameters()).device
        tensor_fields = torch.from_numpy(fields).float().to(device)
        
        with torch.no_grad():
            # Get latent representations
            latents = model.encode(tensor_fields)
            
            # Get log probabilities
            log_probs = model.log_prob(tensor_fields)
            
            # Get reconstructions
            reconstructions = model.reconstruct(tensor_fields)
            
            # Calculate reconstruction errors
            mse = torch.mean((tensor_fields - reconstructions)**2, dim=(1, 2, 3))
            
            # Calculate feature statistics
            feature_means = torch.mean(latents, dim=1)
            feature_stds = torch.std(latents, dim=1)
            
            # Flow latent representation (optional)
            flow_latents = model.flow.latent_representation(latents)
        
        # Convert to numpy
        return {
            'latents': latents.cpu().numpy(),
            'log_probs': log_probs.cpu().numpy(),
            'reconstruction_mse': mse.cpu().numpy(),
            'feature_means': feature_means.cpu().numpy(),
            'feature_stds': feature_stds.cpu().numpy(),
            'flow_latents': flow_latents.cpu().numpy()
        }
    
    # Extract features from test data
    features = extract_features(custom_pae, test_fields)
    
    # Use extracted features for additional analysis
    from sklearn.cluster import KMeans
    
    # Perform clustering on latent space
    kmeans = KMeans(n_clusters=5, random_state=0)
    clusters = kmeans.fit_predict(features['latents'])
    
    # Analyze clusters
    for i in range(5):
        cluster_idx = (clusters == i)
        print(f"Cluster {i}:")
        print(f"  Size: {sum(cluster_idx)}")
        print(f"  Mean log-likelihood: {np.mean(features['log_probs'][cluster_idx]):.3f}")
        print(f"  Mean reconstruction MSE: {np.mean(features['reconstruction_mse'][cluster_idx]):.6f}")
    
    # Visualize clusters in 2D
    from sklearn.decomposition import PCA
    
    # Reduce to 2D
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(features['latents'])
    
    # Plot
    plt.figure(figsize=(10, 8))
    for i in range(5):
        cluster_mask = (clusters == i)
        plt.scatter(
            latents_2d[cluster_mask, 0],
            latents_2d[cluster_mask, 1],
            label=f'Cluster {i}',
            alpha=0.7
        )
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Clusters in Latent Space')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

Transfer Learning with Custom Models
---------------------------------

You can use transfer learning to adapt a pre-trained model to a new dataset:

.. code-block:: python

    def finetune_model(base_model, new_data, n_epochs=10, lr=1e-5):
        """
        Fine-tune a pre-trained model on new data.
        
        Args:
            base_model: Pre-trained PAE model
            new_data: New dataset to adapt to
            n_epochs: Number of training epochs
            lr: Learning rate
            
        Returns:
            Fine-tuned model
        """
        # Create a copy of the base model
        device = next(base_model.parameters()).device
        
        # Create a new PAE with the same architecture
        finetuned_model = ProbabilisticAutoencoder(
            autoencoder=base_model.autoencoder,
            flow=base_model.flow,
            filepath='dummy.h5',
            skip_mkdir=True,
            write_model=False
        )
        
        # Create dataset
        tensor_data = torch.from_numpy(new_data).float()
        dataset = torch.utils.data.TensorDataset(tensor_data)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=16, shuffle=True
        )
        
        # Optimizer for both components
        params = list(finetuned_model.autoencoder.parameters()) + \
                 list(finetuned_model.flow.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        
        # Fine-tuning loop
        finetuned_model.autoencoder.train()
        finetuned_model.flow.train()
        
        for epoch in range(n_epochs):
            total_ae_loss = 0
            total_flow_loss = 0
            
            for batch_idx, (data,) in enumerate(dataloader):
                data = data.to(device)
                
                # Update autoencoder
                optimizer.zero_grad()
                ae_loss = finetuned_model.autoencoder(data)
                ae_loss.backward()
                optimizer.step()
                
                total_ae_loss += ae_loss.item()
                
                # Update flow
                optimizer.zero_grad()
                with torch.no_grad():
                    latents = finetuned_model.autoencoder.encode(data)
                
                flow_loss = finetuned_model.flow(latents)
                flow_loss.backward()
                optimizer.step()
                
                total_flow_loss += flow_loss.item()
            
            print(f"Epoch {epoch+1}/{n_epochs}, "
                  f"AE Loss: {total_ae_loss/len(dataloader):.6f}, "
                  f"Flow Loss: {total_flow_loss/len(dataloader):.6f}")
        
        # Switch back to eval mode
        finetuned_model.autoencoder.eval()
        finetuned_model.flow.eval()
        
        return finetuned_model
    
    # Example usage: fine-tune on new data
    new_data = np.random.randn(100, 1, 64, 64)  # Dummy data
    finetuned_model = finetune_model(custom_pae, new_data, n_epochs=5)

Creating a Model Ensemble
----------------------

For more robust anomaly detection, you can create an ensemble of models:

.. code-block:: python

    class ModelEnsemble:
        """Ensemble of multiple PAE models for more robust anomaly detection."""
        
        def __init__(self, models):
            """
            Initialize with a list of PAE models.
            
            Args:
                models: List of PAE models
            """
            self.models = models
            
        def log_prob(self, inputs):
            """
            Calculate ensemble log probability.
            
            Args:
                inputs: Input data
                
            Returns:
                Average log probability across models
            """
            device = next(self.models[0].parameters()).device
            tensor_inputs = torch.as_tensor(inputs, device=device).float()
            
            all_log_probs = []
            with torch.no_grad():
                for model in self.models:
                    log_probs = model.log_prob(tensor_inputs)
                    all_log_probs.append(log_probs)
            
            # Stack and average
            stacked = torch.stack(all_log_probs, dim=0)
            avg_log_probs = torch.mean(stacked, dim=0)
            
            return avg_log_probs
        
        def reconstruct(self, inputs):
            """
            Calculate ensemble reconstruction.
            
            Args:
                inputs: Input data
                
            Returns:
                Average reconstruction across models
            """
            device = next(self.models[0].parameters()).device
            tensor_inputs = torch.as_tensor(inputs, device=device).float()
            
            all_recons = []
            with torch.no_grad():
                for model in self.models:
                    recon = model.reconstruct(tensor_inputs)
                    all_recons.append(recon)
            
            # Stack and average
            stacked = torch.stack(all_recons, dim=0)
            avg_recon = torch.mean(stacked, dim=0)
            
            return avg_recon
    
    # Create an ensemble with multiple models
    models = [custom_pae, loaded_model, standard_pae]
    ensemble = ModelEnsemble(models)
    
    # Test the ensemble
    test_sample = test_tensor[0:1]
    ensemble_ll = ensemble.log_prob(test_sample)
    ensemble_recon = ensemble.reconstruct(test_sample)
    
    print(f"Ensemble log-likelihood: {ensemble_ll.item():.6f}")
    
    # Compare individual models with ensemble
    individual_lls = [model.log_prob(test_sample).item() for model in models]
    print(f"Individual log-likelihoods: {individual_lls}")
    
    # Visualize ensemble reconstruction
    plt.figure(figsize=(15, 5))
    
    # Original
    plt.subplot(151)
    sns.heatmap(test_fields[0, 0], cmap=cmap, vmin=-2, vmax=2)
    plt.title('Original')
    plt.axis('off')
    
    # Individual reconstructions
    for i, model in enumerate(models):
        plt.subplot(152 + i)
        recon = model.reconstruct(test_sample).cpu().detach().numpy()[0, 0]
        sns.heatmap(recon, cmap=cmap, vmin=-2, vmax=2)
        plt.title(f'Model {i+1}')
        plt.axis('off')
    
    # Ensemble reconstruction
    plt.subplot(155)
    sns.heatmap(ensemble_recon.cpu().detach().numpy()[0, 0], cmap=cmap, vmin=-2, vmax=2)
    plt.title('Ensemble')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

Conclusion
---------

In this tutorial, we've covered:

1. Creating custom autoencoder architectures for specific oceanographic applications
2. Customizing the normalizing flow component with different parameters
3. Building models with specialized features for detecting ocean patterns (e.g., fronts)
4. Training and evaluating custom models
5. Saving and loading custom models
6. Extracting features for additional analysis
7. Transfer learning to adapt pre-trained models to new datasets
8. Creating model ensembles for more robust anomaly detection

These customization techniques allow you to tailor Ulmo's core architecture to specific oceanographic phenomena and datasets, potentially improving detection performance for your application.
            