import ast
import astpretty

merge = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Merge Sort/1.py"
def readAST():
    with open (merge, "r") as file:
        return ast.parse(file.read())

def readASTString():
    with open (merge, "r") as file:
        return file.read()

programAST = readAST()
# print(ast.dump(programAST))
astpretty.pprint(programAST)

