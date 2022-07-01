import networkx as nx
import tensorflow as tf
import numpy as np
from GraphInputLayer import GraphInputLayer
from os.path import dirname, join

class MulticlassGraphInputLayer(GraphInputLayer):
    def splitTrainTest(self, file1, file2, file3):
        graph1, labels1 = self.assignLabels(file1)
        graph2, labels2 = self.assignLabels(file2)
        graph3, labels3 = self.assignLabels(file3)
        split1 = int(0.6 * len(graph1))
        split2 = int(0.6 * len(graph2))
        split3 = int(0.6 * len(graph3))

        x_train = graph1[:split1] + graph2[:split2] + graph3[:split3]
        y_train = labels1[:split1] + labels2[:split2] + labels3[:split3]
        
        x_test = graph1[split1:] + graph2[split2:] + graph3[split3:]
        y_test = labels1[split1:] + labels2[split2:] + labels3[split3:]
        
        return x_train, y_train, x_test, y_test

    def readFiles(self):
        current_dir = dirname(__file__)

        merge = "./Data/Merge Sort"
        quick = "./Data/Quick Sort"
        other = "./Data/Other"

        merge = join(current_dir, merge)
        quick = join(current_dir, quick)
        other = join(current_dir, other)

        xTrain, yTrain, xTest, yTest = self.splitTrainTest(merge, quick, other)
        x_train_nodes, x_train_matrix, y_train, x_test_nodes, x_test_matrix, y_test = self.getDatasets(xTrain, yTrain, xTest, yTest)

        return x_train_nodes, x_train_matrix, y_train, x_test_nodes, x_test_matrix, y_test

            
    