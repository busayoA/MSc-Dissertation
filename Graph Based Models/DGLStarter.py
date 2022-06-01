import dgl
import networkx as nx
import numpy as np
# import tensorflow as tf
import readCodeFiles as rcf
import torch
import torch.nn as nn
import torch.nn.functional as func

# x_train, y_train, x_test, y_test, x_train_matrix, x_test_matrix = rcf.readCodeFiles()
# x_train_i, x_train_j = np.nonzero(x_train_matrix)
# x_train_dgl = dgl.graph((x_train_i, x_train_j))

# print(x_train_dgl)


x_train, y_train, x_test, y_test, x_train_graph, x_test_graph = rcf.readCodeFiles()
x_train_dgl = dgl.from_networkx(x_train_graph, node_attrs=['yValue'])
x_test_dgl = dgl.from_networkx(x_test_graph)

# print(x_train_dgl)

# print('Node feature dimensionality:', x_train_dgl.dim_nfeats)
print('Number of graph categories:', x_train_dgl.gclasses)
