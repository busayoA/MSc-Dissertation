class Node(object):
    def __init__(self, nodeID):
        self.nodeID = nodeID
        self.connections = []

    def addConnection(self, otherNode):
        self.connections.append(otherNode)

    def printNode(self):
        print(self.nodeID, [node.nodeID for node in self.connections])