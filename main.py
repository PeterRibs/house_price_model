from training_model import TrainingModel

dataset_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"

fileName = "housing.data"

columns_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATION', 'B', 'LSTAT', 'MEDV']

testModel = TrainingModel(fileName, dataset_path, columns_names, 0.8, 'RM', 'MEDV')
testModel.callBack()
testModel.training()
testModel.predictTesting()
testModel.comparativePlot()
