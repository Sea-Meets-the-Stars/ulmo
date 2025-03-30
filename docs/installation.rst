Installation
============

Prerequisites
------------

Ulmo requires:

* Python >= 3.8
* PyTorch >= 1.7.0
* CUDA compatible environment (recommended for model training and inference)

Basic Installation
-----------------

You can install Ulmo from source:

.. code-block:: bash

    git clone https://github.com/your-repo/ulmo.git
    cd ulmo
    pip install -e .

Dependencies
-----------

Ulmo requires several libraries for processing oceanographic data, deep learning, and visualization:

Core dependencies:
^^^^^^^^^^^^^^^^^

* numpy
* pandas
* h5py
* torch
* tqdm
* scikit-learn

Data processing:
^^^^^^^^^^^^^^^

* xarray
* smart_open
* boto3
* astropy

Visualization:
^^^^^^^^^^^^^

* matplotlib
* seaborn
* cartopy (for geographical plotting)

Development Installation
-----------------------

For development, it's recommended to create a conda environment:

.. code-block:: bash

    conda create -n ulmo python=3.8
    conda activate ulmo
    pip install -e ".[dev]"

This will install additional development dependencies like pytest and flake8.

S3 Configuration
---------------

If you're working with S3 storage for data, make sure to configure your environment:

.. code-block:: bash

    export ENDPOINT_URL='your_s3_endpoint'
    export AWS_ACCESS_KEY_ID='your_access_key'
    export AWS_SECRET_ACCESS_KEY='your_secret_key'

Model Data
---------

Pre-trained models can be downloaded from:

.. code-block:: bash

    # Set the model directory environment variable
    export SST_OOD_MODELDIR=/path/to/models

    # Download models
    python -m ulmo.models.download
