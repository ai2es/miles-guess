import os
from keras.utils import set_random_seed

def seed_everything(seed=42):
    set_random_seed(seed)
    
    # Set environment variable for TensorFlow
    os.environ['PYTHONHASHSEED'] = str(seed)

