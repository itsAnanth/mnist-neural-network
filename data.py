import numpy as np
import pathlib
from sklearn.model_selection import train_test_split


def get_mnist():
    with np.load(f"{pathlib.Path(__file__).parent.absolute()}/mnist.npz") as f:
        images, labels = f["x_train"], f["y_train"]
    images = images.astype("float32") / 255
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    labels = np.eye(10)[labels]
    
    
    return train_test_split(images, labels, test_size=0.3, random_state=123)