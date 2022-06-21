from GraphInputLayer import GraphInputLayer

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


            
    