import matplotlib.pyplot as plt

from loading_data import LoadingData

class DataVisualization(LoadingData):
    def __init__(self, fileName, datasetPath, columns, fraction):
        super().__init__(fileName, datasetPath)
        self.datasetLoading(columns)
        self.splitDataset(fraction)
    
    def visualRepresentation(self, dataX, dataY): # Visual representation of training data
        fig, ax = plt.subplots()
        x = self.trainingData[dataX]
        y = self.trainingData[dataY]
        ax.scatter(x, y, edgecolors = (0, 0, 0))
        ax.set_xlabel("RM - Average number of rooms/dwelling")
        ax.set_ylabel("MEDV - Median value of owner-occupied homes")
        plt.show()
