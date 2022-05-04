import tensorflow as tf
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from loading_data import LoadingData
from modeling import Modeling

class TrainingModel(LoadingData, Modeling):
    def __init__(self, fileName, datasetPath, columns, fraction, dataX, dataY):
        super().__init__(fileName, datasetPath)
        self.datasetLoading(columns)
        self.splitDataset(fraction)
        self.linearModelAdam()
        self.x_trainingData = self.trainingData[dataX]
        self.y_trainingData = self.trainingData[dataY]
        self.x_testingData = self.testingData[dataX]
        self.y_testingData = self.testingData[dataY]
        self.predictions_list = []
        self.checkpoint_path = "checkpointData/"
        self.earlyStopping = None
        self.checkpointCallback = None
        self.dataframe = None
        self.predictTest = None
        self.model.save_weights(self.checkpoint_path.format(epoch = 0)) # Save the first model version

        # Hiperparameters
        self.n_epochs = 4000
        self.batch_size = 256
        self.n_idle_epochs = 100
        self.n_epochs_log = 200
        self.n_samples_save = self.n_epochs_log * self.x_trainingData.shape[0]
        print('Checkpoint salvo a cada {} amostras'.format(self.n_samples_save))

    def callBack(self):
        self.earlyStopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', 
                                                 patience = self.n_idle_epochs, 
                                                 min_delta = 0.001) # Callback

        self.checkpointCallback = tf.keras.callbacks.ModelCheckpoint(filepath = self.checkpoint_path, 
                                                        verbose = 1, 
                                                        save_weights_only = True,
                                                        save_freq = self.n_samples_save) # Create a callback that saves the model's weights every n_samples_save

    
    def training(self):  # Model training
        history = self.model.fit(self.x_trainingData, 
                            self.y_trainingData, 
                            batch_size = self.batch_size,
                            epochs = self.n_epochs, 
                            validation_split = 0.1, 
                            verbose = 1, 
                            callbacks = [self.earlyStopping, self.checkpointCallback])

        print('keys:', history.history.keys()) # Metrics of training history

        
        mse = np.asarray(history.history['mse']) # Plot values
        val_mse = np.asarray(history.history['val_mse']) # Plot values

        # Dataframe values
        num_values = (len(mse))
        values = np.zeros((num_values, 2), dtype = float)
        values[:,0] = mse
        values[:,1] = val_mse

        
        steps = pd.RangeIndex(start = 0, stop = num_values)
        self.dataframe = pd.DataFrame(values, steps, columns = ["MSE - Training", "MSE - Validation"]) # Create the dataframe

        self.dataframe.head()

    def predictTesting(self):
        self.predictTest = self.model.predict(self.x_testingData).flatten() # Prediction with training model
        self.predictTest # Print the prediction

    def comparativePlot(self):
        sns.set(style = "whitegrid")
        sns.lineplot(data = self.dataframe, palette = "tab10", linewidth  = 2.5)
        plt.ylabel("Mean Squared Error")
        plt.show()

