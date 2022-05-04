
dataset_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"

fileName = "housing.data"

columns_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATION', 'B', 'LSTAT', 'MEDV']

# from data_visualization import DataVisualization
# dataTraining = DataVisualization(fileName, dataset_path, columns_names, 0.8)
# dataTraining.visualRepresentation('RM', 'MEDV')

from training_model import TrainingModel
testModel = TrainingModel(fileName, dataset_path, columns_names, 0.8, 'RM', 'MEDV')
testModel.callBack()
testModel.training()
testModel.predictTesting()
testModel.comparativePlot()
