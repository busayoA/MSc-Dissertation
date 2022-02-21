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
        treeLength = len(self.astTree)
        spaces = [len(i) - len(i.lstrip()) for i in self.astTree]
        treeIndexes, trees = [], []

        for i in range(len(self.astTree)):
            if spaces[i] == 0:
                treeIndexes.append(i)

        #Basic Implementation made to work with code containing only two methods for the purposes of this project
        for i in range(len(treeIndexes)-1):
            i = treeIndexes[i]
            j = treeIndexes[i+1]
            trees.append([self.astTree[i:j]])
            trees.append([self.astTree[j:]])

        treeLevels = []
        for i in range(len(trees)):
            root = trees[i][0][0]
            root = Node(root)
            treeSpaces = [len(i) - len(i.lstrip()) for i in trees[i][0]]
            nodes = []
            treeSet = list(sorted(set(treeSpaces)))
            treeDepth = len(treeSet)

            for j in treeSet:
                thisLevel = []
                for k in range(len(trees[i][0])):
                    innerLevel = []
                    if spaces[k] == j:
                        thirdLevel = []
                        thirdLevel.append(trees[i][0][k])
                        # print(trees[i][0][k])
            #         innerLevel = []
            #         if spaces[k] == j:
            #             innerLevel.append(spaces[k])

            #     thisLevel.append(innerLevel)
            # treeLevels.append(thisLevel)
            # for i in trees[i][0][1:]:
            #     count = max(treeSpaces)
            #     j = 4
            #     while j <= count:
            #         if spaces[i] == j :
            #             newNode = 


            # root = Node(root)

                print(thirdLevel)


        #     astTree = ast.parse(source.read()) #converts the code to a tree
        # # print(astTree._fields) #child nodes

        # # pprint.pprint(ast.dump(astTree))
        
        # l = str(ast.dump(astTree))
        # l = l.split("body")
        # [print(i, end="\n\n") for i in l]

testFile = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Code/Training Data/Quick Sort/qs1.py"
astConverter = ASTConverter(testFile)
astConverter.readInputFile()
astConverter.generateTree()
