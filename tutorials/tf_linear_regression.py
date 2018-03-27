import tensorflow as tf
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

rnd = np.random

# Load dataset
boston = datasets.load_boston()
x, y = boston.data, boston.target
