from random import random
import preTraining as PT

class RNN2():
    def __init__(self, inputCount, hiddenCount, outputCount):
        self.inputCount = inputCount
        self.hiddenCount = hiddenCount
        self.outputCount = outputCount
        self.network = list()
        self.hiddenLayers = [{'hidden weights':[random() for i in range(inputCount + 1)]} for i in range(hiddenCount)]
        self.network.append(self.hiddenLayers)
        self.outputLayer = [{'output weights':[random() for i in range(hiddenCount + 1)]} for i in range(outputCount)]
        self.network.append(self.outputLayer)


    def info(self):
    	print(self.network)




rnn = RNN2(60, 2, 5)
rnn.info()