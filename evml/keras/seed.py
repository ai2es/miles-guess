import os
import random
import numpy as np
import tensorflow as tf

def seed_everything(seed=42):
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set TensorFlow random seed
    tf.random.set_seed(seed)
    
    # Set environment variable for TensorFlow
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # For Keras, if you're using it with TensorFlow 2.x
    tf.compat.v1.set_random_seed(seed)