import dgl
import networkx as nx
import numpy as np
# import tensorflow as tf
import readCodeFiles as rcf
import torch
import torch.nn as nn
import torch.nn.functional as func
from dgl.data import DGLDataset
from dgl.nn import GraphConv
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# x_train, y_train, x_test, y_test, x_train_matrix, x_test_matrix = rcf.readCodeFiles()
# x_train_i, x_train_j = np.nonzero(x_train_matrix)
# x_train_dgl = dgl.graph((x_train_i, x_train_j))

# print(x_train_dgl)


x_train, y_train, x_test, y_test, x_train_graph, x_test_graph = rcf.readCodeFiles()
# x_train_dgl = dgl.from_networkx(x_train_graph, node_attrs = ['yValue', 'nodeType'], edge_attrs = ['edgeType'])
# x_test_dgl = dgl.from_networkx(x_test_graph, node_attrs = ['yValue', 'nodeType'], edge_attrs = ['edgeType'])

x_train_dgl, x_test_dgl = [], []
# print(x_train_graph)
for i in range(len(x_train_graph)):
    x_train_dgl.append(dgl.from_networkx(x_train_graph[i], node_attrs = ['encoding', 'nodeType'], edge_attrs = ['edgeType']))

for i in range(len(x_test_graph)):
    x_test_dgl.append(dgl.from_networkx(x_test_graph[i], node_attrs = ['encoding', 'nodeType'], edge_attrs = ['edgeType']))


# print(x_test_graph)

# x_train_dgl = dgl.to_homogeneous(x_train_dgl)
# x_test_dgl = dgl.to_homogeneous(x_test_dgl)
# # print(x_train_dgl)
# # print(x_train_dgl.edges())

# trainingSampler = SubsetRandomSampler(torch.arange(len(x_train_dgl)))

trainingDataLoader = GraphDataLoader(x_train_dgl, batch_size = 5, drop_last = False)
testingDataLoader = GraphDataLoader(x_test_dgl, batch_size = 5, drop_last = False)


print(trainingDataLoader)

# for batched_graph in trainingDataLoader:
#     print('Number of nodes for each graph element in the batch:', batched_graph.batch_num_nodes())
#     print('Number of edges for each graph element in the batch:', batched_graph.batch_num_edges())


# print(x_test_dgl, y_test_labels)





# class GCN(nn.Module):
#     def __init__(self, in_feats, h_feats, num_classes):
#         super(GCN, self).__init__()
#         self.conv1 = GraphConv(in_feats, h_feats)
#         self.conv2 = GraphConv(h_feats, num_classes)

#     def forward(self, g, in_feat):
#         g = dgl.add_self_loop(g)
#         h = self.conv1(g, in_feat)
#         h = func.relu(h)
#         h = self.conv2(g, h)
#         g.ndata['h'] = h
#         return dgl.mean_nodes(g, 'h')




# it = iter(trainingDataLoader)
# batch = next(it)
# print(batch)

# for graph in trainingDataLoader:
#     print('Number of nodes for each graph element in the batch:', graph.batch_num_nodes())
#     print('Number of edges for each graph element in the batch:', graph.batch_num_edges())

# model = GCN(len(trainingDataLoader), 2, 2)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# batchSize = 10
# for graph in trainingDataLoader:
#     pred = model(graph, sum(graph.batch_num_nodes()))
#     loss = func.cross_entropy(pred, y_train_labels[i])
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()


# num_correct = 0
# num_tests = 0
# for i in range(len(x_test_dgl)):
#     pred = model(x_test_dgl[i], x_test_dgl[i].ndata['nodeType'].float())
#     num_correct += (pred.argmax(1) == y_test_labels[i]).sum().item()
#     num_tests += len(y_test_labels[i])


