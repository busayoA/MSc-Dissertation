import tensorflow as tf
from ParsingAndEmbeddingLayers.Graphs.GraphParser import GraphParser 
from ParsingAndEmbeddingLayers.Graphs.GraphEmbeddingLayer import GraphEmbeddingLayer
from os.path import dirname, join

class GraphDataProcessor:
    def __init__(self, hashed: bool):
        """
        A Graph Data Processor Class. This is where all the data for the graph 
        data structure is processed before the model is run on the data
        hashed: bool - Whether or not we are working with hashed data
        """
        self.hashed = hashed
        self.parser = GraphParser(self.hashed) #initialise the parser object
        self.segmentCount = 40 #the humber of segments for segmentation
        
    def splitTrainTest(self, x, matrices, y):
        """
        Split the data into training and testing
        x - The graph data
        matrices - The matrix representation of the graph data
        y - The graph data labels

        returns: 
        x_train - The training data
        x_train_matrix - The training data in matrix form
        y_train - The training data labels
        x_test - The testing data
        x_test_matrix - The testing data in matrix form
        y_test - The testing data labels
        """ 
        split = int(0.7 * len(x)) #split 70-30

        # The training data
        x_train = x[:split]
        x_train_matrix = matrices[:split]
        y_train = y[:split]

        # The testing data
        x_test = x[split:]
        x_test_matrix = matrices[split:]
        y_test = y[split:]
        
        return x_train, x_train_matrix, y_train, x_test, x_test_matrix, y_test

    def splitTrainTestNoMatrices(self, x, y):
        """
        An alternative to splitTrainTest that does not involve the use of matrices
        x - The graph data
        y - The graph data labels

        returns:
        x_train - The training data
        y_train - The training data labels
        x_test - The testing data
        y_test - The testing data labels
        """
        split = int(0.7*len(x))

        x_train = x[:split]
        y_train = y[:split]

        x_test = x[split:]
        y_test = y[split:]

        return x_train, y_train, x_test, y_test

    def runSegmentation(self, nodeEmbeddings: tf.Tensor, numSegments: int):
        """
        Run the unsorted segment mean algorithm on the node embeddings
        nodeEmbeddings: tf.Tensor - The set of embeddings from each graph
        numSegments: int - The number of segments to be used

        Returns:
        segFunc - The result of segmentation on the graph
        """
        segments = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])
        segFunc = tf.math.unsorted_segment_mean(nodeEmbeddings, segments, num_segments = numSegments)
        return segFunc

    def getMaxLen(self, x):
        """
        Get the maximum graph length for padding
        x - The list of all graphs in both the training and testing data

        Returns:
        maxLen - The number of nodes in the largest available graph
        """
        maxLen = 0
        for i in x:
            if len(i) > maxLen:
                maxLen = len(i)

        return maxLen

    def padGraphs1(self, x, maxLen: int):
        """
        A method to pad the graphs and return a tensor representation of the padded graphs
        This is intended for use with the deep learning models
        x - The list of graphs to be padded
        maxLen: int - The number of nodes in the largest graph

        Returns:
        x - The padded graphs
        """
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
        """
        A method to pad the graphs and return a list representation of the padded graphs
        This is intended for use with the non deep learning models
        x - The list of graphs to be padded
        maxLen: int - The number of nodes in the largest graph

        Returns:
        x - The padded graphs
        """
        length = len(x)
        for i in range(length):
            if len(x[i]) < maxLen:
                padCount = maxLen - len(x[i])
                for j in range(padCount):
                    x[i].append(0.0)
        return x

    def runProcessor1(self):
        """
        Processor 1: This processor is run when working with deep learning models and padded graphs

        Returns:
        x_train - The padded training data in tensor format
        y_train - The training data labels in categorical tensor format
        x_test - The padded testing data in tensor format
        y_test - The testing data labels in categorical tensor format
        """
        # Collect all the required data
        x_train, y_train, x_test, y_test = self.getData()

        total_x = x_train + x_test #combine training and testing to find the largest graph
        maxLen = self.getMaxLen(total_x)
        # Pad the training and testing graphs
        x_train = self.padGraphs1(x_train, maxLen)
        x_test = self.padGraphs1(x_test, maxLen)

        # Convert the padded graphs to tensors for use in the deep learning models
        x_train = tf.convert_to_tensor(x_train)
        y_train = tf.keras.utils.to_categorical(y_train)
        x_test = tf.convert_to_tensor(x_test)
        y_test = tf.keras.utils.to_categorical(y_test)  

        return x_train, y_train, x_test, y_test

    def runProcessor2(self):
        """
        Processor 2: This processor is run when working with non deep learning models and padded graphs

        Returns:
        xTrain2 - The padded training data in list format
        yTrain - The training data labels in list format
        xTest2 - The padded testing data in list format
        yTest - The testing data labels in list format
        """
        xTrain, yTrain, xTest, yTest = self.getData()

        totalX = xTrain + xTest
        maxLen = self.getMaxLen(totalX)

        xTrain = self.padGraphs2(xTrain, maxLen)
        xTest = self.padGraphs2(xTest, maxLen)

        xTrain2, xTest2 = [], []

        # Convert the tensors into Python list form
        for i in xTrain:
            xTrain2.append(i)

        for i in xTest:
            xTest2.append(i)
        return xTrain2, yTrain, xTest2, yTest

    def runProcessor3(self):
        """
        Processor 3: This processor is run when working with deep learning models and segmented graphs

        Returns:
        x_train2 - The segmented training data in tensor format
        y_train - The training data labels in categorical tensor format
        x_test2 - The segmented testing data in tensor format
        y_test - The testing data labels in categorical tensor format
        """
        xTrain, yTrain, xTest, yTest = self.getData()
        x_train2, x_test2 = [], [] #empty list to copy the segmented graphs into

        # Run segmentation on the training data
        for x in xTrain:
            x = tf.convert_to_tensor(x)
            x = [x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, 
            x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x]
            x = self.runSegmentation(x, self.segmentCount)
            x = tf.reshape(x, (len(x[0]), self.segmentCount))
            x_train2.append(x[0]) #add the segmented graph to a new list

        # Run segmentation on the testing data
        for x in xTest:
            x = tf.convert_to_tensor(x)
            x = [x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, 
            x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x]
            x = self.runSegmentation(x, self.segmentCount)
            x = tf.reshape(x, (len(x[0]), self.segmentCount))
            x_test2.append(x[0])

        # COnvert all the data and labels into tensors for use in the deep learning models
        x_train2 = tf.convert_to_tensor(x_train2)
        y_train = tf.keras.utils.to_categorical(yTrain)
        x_test2 = tf.convert_to_tensor(x_test2)
        y_test = tf.keras.utils.to_categorical(yTest)  
        
        return x_train2, y_train, x_test2, y_test

    def runProcessor4(self):
        """
        Processor 4: This processor is run when working with non deep learning models and segmented graphs

        Returns:
        xTrain2 - The segmented training data in list format
        yTrain - The training data labels in list format
        xTest2 - The segmented testing data in list format
        yTest - The testing data labels in list format
        """
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
            #get the actual value and add it to the list to be returned
            xTrain2.append(list(i.numpy()))

        for i in x_test2:
            xTest2.append(list(i.numpy()))
        return xTrain2, yTrain, xTest2, yTest

    def runEmbeddingLayer(self):
        """
        Run the Graph Embedding Layer on each graph/set of nodes
        This is the method that performs 'unhashing' on the nodes
        """
        index = 0 #tracker value
        gp = GraphParser(False) #Create a GraphParser object with hashed = False
        x, x_graph, matrices, labels = gp.readFiles()

        embeddings = [] #an empty list for embeddings
        print("Collecting Graph Embeddings:")
        for graph in x_graph:
            if index % 5 == 0:
                print(end = ".")
            embed = GraphEmbeddingLayer(graph)
            embeddings.append(embed.vectors)
            index += 1

        # split into training and testing data after embedding
        x_train, y_train, x_test, y_test = self.splitTrainTestNoMatrices(embeddings, labels)
        
        # write the data into files for easy retrieval
        self.writeToFiles(x_train, y_train, x_test, y_test)

    def runHashLayer(self):
        """
        The alternative to runEmbeddingLayer that runs the hashing algorithm on the nodes
        """
        gp = GraphParser(True) #Create a GraphParser object with hashed = True
        x, x_graph, matrices, labels = gp.readFiles()

        embeddings = []
        for graph in x:
            # Hashed = true returns the graph with the label in the final position so the actual graph is at :-1
            g = graph[:-1]
            embeddings.append(g)

        # split into training and testing data and write to files for easier processing
        x_train, y_train, x_test, y_test = self.splitTrainTestNoMatrices(embeddings, labels)
        self.writeToFiles(x_train, y_train, x_test, y_test)

    def runParser(self, processor: int):
        """
        Run the parser object on each processor for the padded graphs
        processor: int - The number of the processor to be selected

        Returns:
        (If processor is 1) 
        x_train_graph - The training data in graph form 
        x_train_matrix - The training data in matrix form  
        y_train - The training data labels
        x_test_graph - The testing data in graph form
        x_test_matrix - The testing data in matrix form
        y_test - The testing data labels

        OR 
        (If processor is 2)
        x_train_nodes - The list of nodes from the training data 
        y_train - The training data labels
        x_test_nodes - The list of nodes from the testing data 
        y_test - The testing data labels
        """

        x, matrices, labels = self.parser.readFiles() 
        x_train_all, x_train_matrix, y_train, x_test_all, x_test_matrix, y_test = self.splitTrainTest(x, matrices, labels)

        x_train_nodes = [] #empty list for the node representations
        x_train_graph = [] #empty list for the graph representations

        for i in range(len(x_train_all)):
            x_train_nodes.append(x_train_all[i][0]) #get the nodes
            x_train_graph.append(x_train_all[i][1]) #get the entire graph

        x_test_nodes = []
        x_test_graph = []
        for i in range(len(x_test_all)):
            x_test_nodes.append(x_test_all[i][0])
            x_test_graph.append(x_test_all[i][1])

        if processor == 1:
            return x_train_graph, x_train_matrix, y_train, x_test_graph, x_test_matrix, y_test
        elif processor == 2:
            return x_train_nodes, y_train, x_test_nodes, y_test

    def getFileNames(self):
        """
        Get the names of the files to be read

        Returns: 
        xTrain - The name of the file containing the appropriate training data
        yTrain - The name of the file containing the appropriate training data labels
        xTest - The name of the file containing the appropriate testing data
        yTest - The name of the file containing the appropriate testing data labels
        """
        current_dir = dirname(__file__)
        if self.hashed is False:
            # if hashed is false, return the unhashed file names
            xTrain = join(current_dir, "./Graph Data/graph_x_train.txt")
            yTrain = join(current_dir, "./Graph Data/graph_y_train.txt")
            xTest = join(current_dir, "./Graph Data/graph_x_test.txt")
            yTest = join(current_dir, "./Graph Data/graph_y_test.txt")
        else:
            # if hashed is true, return the hashed file names
            xTrain = join(current_dir, "./Graph Data/graph_x_train_hashed.txt")
            yTrain = join(current_dir, "./Graph Data/graph_y_train_hashed.txt")
            xTest = join(current_dir, "./Graph Data/graph_x_test_hashed.txt")
            yTest = join(current_dir, "./Graph Data/graph_y_test_hashed.txt")

        return xTrain, yTrain, xTest, yTest

    def writeToFiles(self, x_train, y_train, x_test, y_test):
        """
        Write the embeddings into files
        x_train - The training data to be written
        y_train - The training data labels to be written
        x_test - The testing data to be written
        y_test - The testing data labels to be written
        """
        xTrain, yTrain, xTest, yTest = self.getFileNames() #collect the file names
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

    def readFiles(self, filePath, yFile: bool):
        """
        Read the contents of a file from a given file path
        filePath - The path of the file to be read
        yFile: bool - Whether or not the file contains labels 

        Returns:
        values - The dformatted data read from the files
        """
        with open(filePath, 'r') as reader:
            values = reader.readlines()

        # If it is a file of labels, convert into integers form from string
        if yFile is True:
            values = [int(i[0]) for i in values]
        else:
            # if not a file of labels, remove brackets and commas and newlines and convert bact to floats
            for x in range(len(values)):
                values[x] = values[x].replace("[", "").replace("]", "").strip("\n")
                values[x] = values[x].split(",")
                values[x] = [float(i) for i in values[x]]

        return values

    def getData(self):
        """
        Get the data for running the models

        Returns:
        x_train - The training data that has been read
        y_train - The training data labels that have been read
        x_test - The testing data that has been read
        y_test - The testing data labels that have been read
        """

        xTrain, yTrain, xTest, yTest = self.getFileNames()
        
        x_train, y_train, x_test, y_test = [], [], [], []
        # Call the readfiles method to read all the files
        x_train = self.readFiles(xTrain, False)
        y_train = self.readFiles(yTrain, True)

        x_test = self.readFiles(xTest, False)
        y_test = self.readFiles(yTest, True)

        return x_train, y_train, x_test, y_test

