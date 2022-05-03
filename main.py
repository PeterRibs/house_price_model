# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from loading_data import LoadingData

dataset_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"

fileName = "housing.data"

columns_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATION', 'B', 'LSTAT', 'MEDV']

data = LoadingData(fileName, dataset_path)

data.datasetLoading(columns_names)

data.splitDataset(0.8)