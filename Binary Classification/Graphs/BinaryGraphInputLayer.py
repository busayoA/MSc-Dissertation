import networkx as nx
import tensorflow as tf
import numpy as np
from GraphInputLayer import GraphInputLayer
from os.path import dirname, join

class BinaryGraphInputLayer(GraphInputLayer):
    def splitTrainTest(self, file1, file2, file3=None):
        graph1, labels1 = self.assignLabels(file1)
        graph2, labels2 = self.assignLabels(file2)
        split1 = int(0.6 * len(graph1))
        split2 = int(0.6 * len(graph2))

        x_train = graph1[:split1] + graph2[:split2]
        y_train = labels1[:split1] + labels2[:split2]
        
        x_test = graph1[split1:] + graph2[split2:]
        y_test = labels1[split1:] + labels2[split2:]
        
        return x_train, y_train, x_test, y_test

    def readFiles(self):
        current_dir = dirname(__file__)

        merge = "./Data/Merge Sort"
        quick = "./Data/Quick Sort"

        merge = join(current_dir, merge)
        quick = join(current_dir, quick)

        xTrain, yTrain, xTest, yTest = self.splitTrainTest(merge, quick)
        x_train_nodes, x_train_matrix, y_train, x_test_nodes, x_test_matrix, y_test = self.getDatasets(xTrain, yTrain, xTest, yTest)

        return x_train_nodes, x_train_matrix, y_train, x_test_nodes, x_test_matrix, y_test

            
    