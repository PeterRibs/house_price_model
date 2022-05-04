import pandas as pd
from tensorflow import keras
import ssl

class LoadingData():

    def __init__(self, fileName, datasetPath):
        print("LOADING_DATA")
        ssl._create_default_https_context = ssl._create_unverified_context
        self.datasetPath = keras.utils.get_file(fileName, datasetPath) # Download data
        self.dataset = None
        self.trainingData = None
        self.testingData = None

    def datasetLoading(self, columns, naValues = "?", dataComment = '\t', separation = " ", skipInitialSpace = True): # Loading dataset
        self.columns = columns # Column names
        self.dataset = pd.read_csv(self.datasetPath, names = columns, na_values = naValues, comment = dataComment, sep = separation, skipinitialspace = skipInitialSpace)
        print(self.dataset.head())
    
    def splitDataset(self, fraction): # Split dataset
        self.trainingData = self.dataset.sample(frac = fraction, random_state = 0)
        self.testingData = self.dataset.drop(self.trainingData.index)
        print("Traing data shape: ", self.trainingData.shape)
        print("Testing data shape: ", self.testingData.shape)
