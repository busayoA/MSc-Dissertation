import dgl
import readFiles as rf
import torch
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
            # g = dgl.graph((src, dst), num_nodes=numNodes)
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

it = iter(train_dataloader)
batch = next(it)
print(batch)

batched_graph, label = batch
print('Number of nodes for each graph element in the batch:', batched_graph.batch_num_nodes())
print('Number of edges for each graph element in the batch:', batched_graph.batch_num_edges())
graphs = dgl.unbatch(batched_graph)
print('The original graphs in the minibatch:')
print(graphs)

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = func.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')

model = GCN(0, 16, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(20):
    for batched_graph, labels in train_dataloader:
        pred = model(batched_graph, None)
        loss = func.cross_entropy(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()