import matplotlib.pyplot as plt

from loading_data import LoadingData

class DataVisualization(LoadingData):
    def __init__(self, fileName, datasetPath):
        super().__init__(fileName, datasetPath)
    
    def visualRepresentation(self, dataX, dataY): # Visual representation of training data
        fig, ax = plt.subplots()
        x = self.trainingData[dataX]
        y = self.trainingData[dataY]
        ax.scatter(x, y, edgecolors = (0, 0, 0))
        ax.set_xlabel(dataX)
        ax.set_ylabel(dataY)
        plt.show()
