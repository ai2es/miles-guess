.. install

Install MILES GUESS
===================

MILES GUESS supports Python 3.8 to 3.11. Support
for newer versions of Python depends on the choice
of deep learning model backend.

The primary ML library dependency is keras 3. Install one
of the keras 3 backends (tensorflow, pytorch, or jax).

First, set up a base Python environment on
your system. We highly recommend using miniconda or
mambaforge to easily install all the dependencies.

To install the stable version of the package:

.. code-block:: bash

    pip install miles-guess

To use the latest developed version of the package,
first download :

.. code-block:: bash

    git clone git@github.com:ai2es/miles-guess.git
    cd miles-guess

Next, build the environment for the package.

For CPU-based systems:

.. code-block:: bash

    mamba env create -f environment.yml

For GPU-based systems:

.. code-block:: bash

    mamba env create -f environment_casper.yml

If you want to install miles-guess directly
after building your environment run:

.. code-block:: bash

    pip install .

Keras 3 Installation
--------------------
MILES GUESS depends on Keras 3 as its primary
ML backend and through Keras 3 can support
tensorflow, pytorch, or jax. The current version
of tensorflow (2.15) will downgrade keras 3 to
keras 2.15 upon installation. If you run into
this issue, reinstall keras 3 by running the
following command:

.. code-block:: bash

    pip install --upgrade keras









