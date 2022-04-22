class Node(object):
    def __init__(self, contents):
        self.contents = contents
        self.children = []
        self.childIDs = []
        self.nodeID = contents['id']
        self.traversed = False

    def addChild(self, childNode):
        self.children.append(childNode)
        self.childIDs.append(childNode.nodeID)

    def printNode(self):
        self.sortConnections()
        print(self.nodeID, [node.nodeID for node in self.children])

    def sortConnections(self):
        self.children.sort(key=lambda node: node.nodeID)