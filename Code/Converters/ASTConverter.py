import ast, pprint
from anytree import Node, RenderTree

root = Node(10)

level_1_child_1 = Node(34, parent=root)
level_1_child_2 = Node(89, parent=root)
level_2_child_1 = Node(45, parent=level_1_child_1)
level_2_child_2 = Node(50, parent=level_1_child_2)

for pre, fill, node in RenderTree(root):
    print("%s%s" % (pre, node.name))

astTree = []

def runAST():
    with open("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Code/Training Data/Bubble Sort/bs1.py", "r") as source:
        astTree = ast.parse(source.read()) #converts the code to a tree

    # astAnalyser = ASTAnalyser()
    # astAnalyser.visit(astTree)
    # astAnalyser.printDetails()
    pprint.pprint(ast.dump(astTree))
    # pprint.pprint(astTree)

    l = [i for i in astTree]

    print(l)
runAST()