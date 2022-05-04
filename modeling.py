import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Modeling:
    def __init__(self):
        self.model = None

    def linearModelAdam(self):  # Model function
        self.model = keras.Sequential([layers.Dense(1, use_bias = True, input_shape = (1,), name = 'layer')]) # Cria o modelo
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01, 
                                            beta_1 = 0.9, 
                                            beta_2 = 0.99, 
                                            epsilon = 1e-05, 
                                            amsgrad = False, 
                                            name = 'Adam') # Otimizer      
        self.model.compile(loss = 'mse', 
                    optimizer = optimizer, 
                    metrics = ['mae','mse']) # Model copile
        return self.model

    def plotModel (self): # Model Plot
        tf.keras.utils.plot_model(self.model, 
                          to_file = 'model.png', 
                          show_shapes = True, 
                          show_layer_names = True,
                          rankdir = 'TB', 
                          expand_nested = False, 
                          dpi = 100)
