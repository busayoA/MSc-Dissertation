import Edge, Node

class Graph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def addNode(self, node):
        if node not in self.nodes:
            self.nodes.append(node)
        else:
            print("That node already exists")

    def addEgde(self, startNode, endNode):
        newEdge = Edge.Edge(len(self.edges)-1, startNode, endNode)
        edgeFound = False
        for edge in self.edges:
            if self.identicalEdges(edge, newEdge) is True:
                edgeFound = True

        if edgeFound is False:
            self.edges.append(Edge.Edge(len(self.edges)-1, startNode, endNode))
            startNode.addConnection(endNode)
        else:
            print("Edge between", startNode.nodeID, "&", endNode.nodeID, "already exists")

    def addRelationship(self, startNode, endNode):
        if startNode not in self.nodes:
            print("Invalid Operation: node", startNode.nodeID, "does not exist in this graph")
        elif endNode not in self.nodes:
            print("Invalid Operation: node", endNode.nodeID, "does not exist in this graph")
        else:
            self.addEgde(startNode, endNode)

    def removeNode(self, node):
        for endNode in node.connections:
            self.removeEdge(node, endNode)
        
        for startNode in self.nodes:
            self.removeEdge(startNode, node)

        self.nodes.remove(node)

    def removeEdge(self, startNode, endNode):
        if self.adjacentNodes(startNode, endNode) is True:
            placeholderEdge = Edge.Edge(len(self.edges)-1, startNode, endNode)
            for edge in self.edges:
                if self.identicalEdges(placeholderEdge, edge) is True:
                    placeholderEdge = edge
            
            self.edges.remove(placeholderEdge)
            startNode.connections.remove(endNode)

    def identicalEdges(self, edge1, edge2):
        if edge1.start == edge2.start and edge1.end == edge2.end:
            return True
        else:
            return False

    def adjacentNodes(self, firstNode, secondNode):
        adjacent = False
        if firstNode not in self.nodes or secondNode not in self.nodes:
            print("Invalid Operation: either node", firstNode.nodeID, "or", secondNode.nodeID, "does not exist in this graph")
        else:
            for node in firstNode.connections:
                if node.nodeID == secondNode.nodeID:
                    print("Edge from", firstNode.nodeID, "to", secondNode.nodeID)
                    adjacent = True
                    break
            
            for node in secondNode.connections:
                if node.nodeID == firstNode.nodeID:
                    print("Edge from", secondNode.nodeID, "to", firstNode.nodeID)
                    adjacent = True
                    break
        
        return adjacent

    def getConnectedNodes(self, node):
        if node not in self.nodes:
            print("Invalid Operation: node", node.nodeID, "does not exist in this graph")
        else:
            if len(node.connections) == 0:
                print("No conencted nodes")
            for node in node.connections:
                print(node.nodeID, end=" ")

    def printAdjList(self):
        for node in self.nodes:
            node.printNode()
    
nodes, edges = [], []
graph = Graph(nodes, edges)
nodeA = Node.Node("A")
nodeB = Node.Node("B")
nodeC = Node.Node("C")
nodeD = Node.Node("D")
nodeE = Node.Node("E")
nodeF = Node.Node("F")
graph.addNode(nodeA)
graph.addNode(nodeB)
graph.addNode(nodeC)
graph.addNode(nodeD)
graph.addNode(nodeE)

graph.addRelationship(nodeA, nodeB)
graph.addRelationship(nodeB, nodeA)
graph.addRelationship(nodeA, nodeC)
graph.addRelationship(nodeD, nodeB)
graph.addRelationship(nodeB, nodeE)
graph.addRelationship(nodeA, nodeF)

graph.addEgde(nodeA, nodeB)

graph.printAdjList()

graph.adjacentNodes(nodeA, nodeB)
graph.adjacentNodes(nodeA, nodeF)
graph.getConnectedNodes(nodeF)
graph.addNode(nodeD)
graph.removeEdge(nodeA, nodeB)
graph.removeNode(nodeA)
graph.printAdjList()
