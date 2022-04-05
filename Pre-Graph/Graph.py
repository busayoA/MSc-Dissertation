import Edge, Node

class Graph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self.edgeSets = []

    def addEgde(self, startNode, endNode):
        if Edge.Edge(startNode, endNode) not in self.edges:
            self.edges.append(Edge.Edge(startNode, endNode))
            self.addRelationship(startNode, endNode)
    
    def addNode(self, node, connections):
        node = Node.Node(len(self.nodes), connections)
        self.nodes.append(node)

        # for connection in connections:

    def addRelationship(self, startNode, endNode):
        self.edgeSets.append([startNode + "--->" + endNode])
    
