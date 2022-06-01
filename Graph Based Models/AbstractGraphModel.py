import tensorflow as tf
import pickle as pkl
from abc import ABC, abstractmethod
from typing import Dict, AnyStr, Any, List



class AbstractGraphModel(ABC, tf.keras.Model):
    """ An Abstract Base Superclass for the two graph models.
    This is where the core features of both Graph-Based Neural Networks is defined"""

    # ==========================================================================================================
    # BASIC MODEL FEATURES

    def __init__(self, layers: list(), neuronCounts: list(), parameters: Dict[str, Any]):
        super(AbstractGraphModel, self).__init__()
        self.layers = layers
        self.neuronCounts = neuronCounts
        self.layerCount = len(layers)
        self.hiddenLayerCount = len(layers)-2
        self.featureCount = layers[0]
        self.classCount = layers[-1]
        self.parameters = parameters
        self.parameterCount = 0

        self.w, self.wGradients, self.b, self.bGradients = {}, {}, {}, {}
        
        for i in range(1, self.layerCount):
            self.w[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], self.layers[i-1])))
            self.b[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], 1)))

    @abstractmethod
    def call(self, inputs):
        raise NotImplementedError()

    @classmethod
    def setDefaultParameters(cls):
        defaultParameters = {'defaultLearningRate': 0.005, 'defaultHiddenLayerCount': 5, 'defaultEpochs': 50, 'defaultBatchSize': 256, 'maxBatchSize': 500,
            'internalDenseLayers': 1, 'defaultActivationFunction': 'tanh', 'maxEpochValue': 100, 'defaultOptimizer': 'Adam'}
        
        return defaultParameters

    @staticmethod
    @abstractmethod
    def setModelName(modelParameters: Dict[str, Any]) -> str:
        raise NotImplementedError()

    @abstractmethod
    def activationFunction(self):
        raise NotImplementedError()

    def saveModel(self, fileName: str):
        #TODO: Implement fully after building model
        pass
    
    def loadSavedModel(self, fileName: str):
        #TODO: Implement fully after building model
        pass

    # ==========================================================================================================

    # ==========================================================================================================
    # MODEL CONSTRUCTION
    def buildModel(self):
        #TODO: Take time to properly implement and test
        pass

    def modelSummary(self):
        #TODO: Take time to properly implement and test
        pass

    @abstractmethod
    def applyIndividualLayer(self):
        raise NotImplementedError()


    def applyDenseLayer(self):
        #TODO: Take time to properly implement and test
        pass

    def applyBackPropagation(self):
        #TODO: Take time to properly implement and test
        pass

    # ==========================================================================================================
        

    # ==========================================================================================================
    # MODEL TRAINING & TESTING

    def runIndividualEpoch(self, epochCount: int):
        #TODO: Take time to properly implement and test
        pass

    def trainModel(self, xValues, yValues):
        #TODO: Take time to properly implement and test
        pass

    def trainModelByBatches(self, xValues, yValues):
        #TODO: Take time to properly implement and test
        pass

    def getPerformaceMetrics(self):
        #TODO: Take time to properly implement and test
        pass

    # ==========================================================================================================


