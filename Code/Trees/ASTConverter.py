import ast, pprint
from anytree import Node, RenderTree

root = Node(10)

level_1_child_1 = Node(34, parent=root)
level_1_child_2 = Node(89, parent=root)
level_2_child_1 = Node(45, parent=level_1_child_1)
level_2_child_2 = Node(50, parent=level_1_child_2)

for pre, fill, node in RenderTree(root):
    print("%s%s" % (pre, node.name))

class ASTConverter():
    def __init__(self, inputFile):
        self.inputFile = inputFile
        self.astTree = []

    def readInputFile(self):
        exhausted = False
        with open(self.inputFile, "r") as source:
            self.astTree = source.readlines()


    def generateTree(self):
        root = Node(self.astTree[0])
        maxLen = len(self.astTree)
        spaces = list(set([len(i) - len(i.lstrip()) for i in self.astTree]))
        treeDepth = len(spaces)
        print(treeDepth)




        #     astTree = ast.parse(source.read()) #converts the code to a tree
        # # print(astTree._fields) #child nodes

        # # pprint.pprint(ast.dump(astTree))
        
        # l = str(ast.dump(astTree))
        # l = l.split("body")
        # [print(i, end="\n\n") for i in l]

testFile = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Code/Training Data/Bubble Sort/bs1.py"
astConverter = ASTConverter(testFile)
astConverter.readInputFile()
astConverter.generateTree()