import dgl, torch
import readFiles as rf
import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as func
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn import GraphConv


x_train, y_train, x_test, y_test = rf.getParsedFiles()

class TrainingData(DGLDataset):
    def __init__(self):
        super().__init__(name='x_train')

    def process(self):
        self.graphs = []
        self.labels = []

        for i in range(len(x_train)):
            label = y_train[i]
            g = x_train[i]
            # g = dgl.graph((src, dst), num_nodes=numNodes)
            self.graphs.append(g)
            self.labels.append(label)
        self.labels = torch.LongTensor(self.labels)


    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

class TestData(DGLDataset):
    def __init__(self):
        super().__init__(name='x_test')

    def process(self):
        self.graphs = []
        self.labels = []

        for i in range(len(x_test)):
            label = y_test[i]
            g = x_test[i]
            self.graphs.append(g)
            self.labels.append(label)
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

trainingData = TrainingData()
x_train, y_train = trainingData[0:7]

testData = TestData()
x_test, y_test = testData[0:]

train_dataloader = GraphDataLoader(trainingData, batch_size=5, drop_last=False)
test_dataloader = GraphDataLoader(testData, batch_size=5, drop_last=False)

# it = iter(train_dataloader)
# batch = next(it)
# print(batch)

# batched_graph, label = batch
# print('Number of nodes for each graph element in the batch:', batched_graph.batch_num_nodes())
# print('Number of edges for each graph element in the batch:', batched_graph.batch_num_edges())
# graphs = dgl.unbatch(batched_graph)
# print('The original graphs in the minibatch:')
# print(graphs)
