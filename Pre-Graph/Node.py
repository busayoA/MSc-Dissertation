class Node:
    def __init__(self, node, info):
        self.node = node
        self.id = info[0]
        self.adjacent = info[1:]
        self.info = info
        self.connections = []

    def addConnection(self, position, otherNode):
        if position == "start":
            self.connections.append(["goes to", otherNode])
        elif position == "end":
            self.connections.append(["comes from", otherNode])
        else:
            print("Invalid operation")