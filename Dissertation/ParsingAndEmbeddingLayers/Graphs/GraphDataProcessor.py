import tensorflow as tf
from ParsingAndEmbeddingLayers.Graphs.GraphParser import GraphParser 
from ParsingAndEmbeddingLayers.Graphs.GraphEmbeddingLayer import GraphEmbeddingLayer
from os.path import dirname, join

class GraphDataProcessor:
    def __init__(self, hashed):
        self.hashed = hashed
        self.parser = GraphParser(self.hashed)
        self.segmentCount = 40
        
    def splitTrainTest(self, x, matrices, y):
            split = int(0.7 * len(x))

            x_train = x[:split]
            x_train_matrix = matrices[:split]
            y_train = y[:split]

            x_test = x[split:]
            x_test_matrix = matrices[split:]
            y_test = y[split:]
            
            return x_train, x_train_matrix, y_train, x_test, x_test_matrix, y_test

    def runSegmentation(self, nodeEmbeddings: tf.Tensor, numSegments: int):
        segments = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])
        segFunc = tf.math.unsorted_segment_mean(nodeEmbeddings, segments, num_segments = numSegments)
        return segFunc

    def getMaxLen(self, x):
        maxLen = 0
        for i in x:
            if len(i) > maxLen:
                maxLen = len(i)

        return maxLen

    def padGraphs1(self, x, maxLen):
        length = len(x)
        for i in range(length):
            if len(x[i]) < maxLen:
                padCount = maxLen - len(x[i])
                x[i] = list(x[i])
                for j in range(padCount):
                    x[i].append(0.0)
            x[i] = tf.convert_to_tensor(x[i])
        return x

    def padGraphs2(self, x, maxLen):
        length = len(x)
        for i in range(length):
            if len(x[i]) < maxLen:
                padCount = maxLen - len(x[i])
                for j in range(padCount):
                    x[i].append(0.0)
        return x

    def runParser(self, processor: int):
        x, matrices, labels = self.parser.readFiles()
        x_train_all, x_train_matrix, y_train, x_test_all, x_test_matrix, y_test = self.splitTrainTest(x, matrices, labels)

        x_train_nodes = []
        x_train_graph = []

        for i in range(len(x_train_all)):
            x_train_nodes.append(x_train_all[i][0])
            x_train_graph.append(x_train_all[i][1])

        x_test_nodes = []
        x_test_graph = []
        for i in range(len(x_test_all)):
            x_test_nodes.append(x_test_all[i][0])
            x_test_graph.append(x_test_all[i][1])

        if processor == 1:
            return x_train_graph, x_train_matrix, y_train, x_test_graph, x_test_matrix, y_test
        elif processor == 2:
            return x_train_nodes, y_train, x_test_nodes, y_test

    def runProcessor1(self):
        x_train, y_train, x_test, y_test = self.getData()

        total_x = x_train + x_test
        maxLen = self.getMaxLen(total_x)
        x_train = self.padGraphs1(x_train, maxLen)
        x_test = self.padGraphs1(x_test, maxLen)

        x_train = tf.convert_to_tensor(x_train)
        y_train = tf.keras.utils.to_categorical(y_train)
        x_test = tf.convert_to_tensor(x_test)
        y_test = tf.keras.utils.to_categorical(y_test)  

        return x_train, y_train, x_test, y_test

    def runProcessor2(self):
        xTrain, yTrain, xTest, yTest = self.getData()

        totalX = xTrain + xTest
        maxLen = self.getMaxLen(totalX)

        xTrain = self.padGraphs2(xTrain, maxLen)
        xTest = self.padGraphs2(xTest, maxLen)

        xTrain2, xTest2 = [], []
        for i in xTrain:
            xTrain2.append(i)

        for i in xTest:
            xTest2.append(i)
        return xTrain2, yTrain, xTest2, yTest

    def runProcessor3(self):
        xTrain, yTrain, xTest, yTest = self.getData()
        x_train2, x_test2 = [], []
        for x in xTrain:
            x = tf.convert_to_tensor(x)
            x = [x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, 
            x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x]
            x = self.runSegmentation(x, self.segmentCount)
            x = tf.reshape(x, (len(x[0]), self.segmentCount))
            x_train2.append(x[0])

        for x in xTest:
            x = tf.convert_to_tensor(x)
            x = [x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, 
            x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x]
            x = self.runSegmentation(x, self.segmentCount)
            x = tf.reshape(x, (len(x[0]), self.segmentCount))
            x_test2.append(x[0])

        x_train2 = tf.convert_to_tensor(x_train2)
        y_train = tf.keras.utils.to_categorical(yTrain)
        x_test2 = tf.convert_to_tensor(x_test2)
        y_test = tf.keras.utils.to_categorical(yTest)  
        
        return x_train2, y_train, x_test2, y_test

    def runProcessor4(self):
        xTrain, yTrain, xTest, yTest = self.getData()
        x_train2, x_test2 = [], []
        for x in xTrain:
            x = tf.convert_to_tensor(x)
            x = [x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, 
            x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x]
            x = self.runSegmentation(x, self.segmentCount)
            x = tf.reshape(x, (len(x[0]), self.segmentCount))
            x_train2.append(x[0])

        for x in xTest:
            x = tf.convert_to_tensor(x)
            x = [x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, 
            x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x]
            x = self.runSegmentation(x, self.segmentCount)
            x = tf.reshape(x, (len(x[0]), self.segmentCount))
            x_test2.append(x[0])

        xTrain2, xTest2 = [], []
        for i in x_train2:
            xTrain2.append(list(i.numpy()))

        for i in x_test2:
            xTest2.append(list(i.numpy()))
        return xTrain2, yTrain, xTest2, yTest

    def runEmbeddingLayer(self):
        index = 0
        gp = GraphParser(False)
        x, x_graph, matrices, labels = gp.readFiles()
        # print(x_graph)
        embeddings = []
        print("Collecting Graph Embeddings:")
        for graph in x_graph:
            if index % 5 == 0:
                print(end = ".")
            embed = GraphEmbeddingLayer(graph)
            embeddings.append(embed.vectors)
            index += 1

        x_train, y_train, x_test, y_test = self.splitTrainTest(embeddings, labels)
        
        self.writeToFiles(x_train, y_train, x_test, y_test)

    def runHashLayer(self):
        gp = GraphParser(True)
        x, x_graph, matrices, labels = gp.readFiles()

        embeddings = []
        for graph in x:
            g = graph[:-1]
            embeddings.append(g)

        x_train, y_train, x_test, y_test = self.splitTrainTest(embeddings, labels)
        
        self.writeToFiles(x_train, y_train, x_test, y_test)

        
    def splitTrainTest(self, x, y):
        split = int(0.7*len(x))

        x_train = x[:split]
        y_train = y[:split]

        x_test = x[split:]
        y_test = y[split:]

        return x_train, y_train, x_test, y_test

    
    def getFileNames(self):
        current_dir = dirname(__file__)
        if self.hashed is False:
            xTrain = join(current_dir, "./Graph Data/graph_x_train.txt")
            yTrain = join(current_dir, "./Graph Data/graph_y_train.txt")
            xTest = join(current_dir, "./Graph Data/graph_x_test.txt")
            yTest = join(current_dir, "./Graph Data/graph_y_test.txt")
        else:
            xTrain = join(current_dir, "./Graph Data/graph_x_train_hashed.txt")
            yTrain = join(current_dir, "./Graph Data/graph_y_train_hashed.txt")
            xTest = join(current_dir, "./Graph Data/graph_x_test_hashed.txt")
            yTest = join(current_dir, "./Graph Data/graph_y_test_hashed.txt")

        return xTrain, yTrain, xTest, yTest

    def writeToFiles(self, x_train, y_train, x_test, y_test):
        xTrain, yTrain, xTest, yTest = self.getFileNames()
        with open(xTrain, 'w') as writer:
            for i in x_train:
                writer.write(str(i) + "\n")

        with open(yTrain, 'w') as writer:
            for i in y_train:
                writer.write(str(i) + "\n")    
        
        with open(xTest, 'w') as writer:
            for i in x_test:
                writer.write(str(i) + "\n")
            
        with open(yTest, 'w') as writer:
            for i in y_test:
                writer.write(str(i) + "\n")

    def getData(self):
        xTrain, yTrain, xTest, yTest = self.getFileNames()
        
        x_train, y_train, x_test, y_test = [], [], [], []
        x_train = self.readFiles(xTrain, False)
        y_train = self.readFiles(yTrain, True)

        x_test = self.readFiles(xTest, False)
        y_test = self.readFiles(yTest, True)

        return x_train, y_train, x_test, y_test

    def readFiles(self, filePath, yFile: bool):
        with open(filePath, 'r') as reader:
            values = reader.readlines()

        if yFile is True:
            values = [int(i[0]) for i in values]
        else:
            for x in range(len(values)):
                values[x] = values[x].replace("[", "").replace("]", "").strip("\n")
                values[x] = values[x].split(",")
                values[x] = [float(i) for i in values[x]]

        return values

