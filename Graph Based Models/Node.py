class Node(object):
    def __init__(self, nodeID):
        self.nodeID = nodeID
        self.connections = []
        self.traversed = False

    def addConnection(self, otherNode):
        self.connections.append(otherNode)

    def printNode(self):
        self.sortConnections()
        print(self.nodeID, [node.nodeID for node in self.connections])

    def sortConnections(self):
        self.connections.sort(key=lambda node: node.nodeID)