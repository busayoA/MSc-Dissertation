import os, ast
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer

# =====================================================================================================================================================================================
# CODE 
merge = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Merge Sort/1.py"
quick = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Quick Sort"
other = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Other"
pythonASTs = []

# def assignLabels(filePath):
#     os.chdir(filePath)
#     i = 0
#     for file in os.listdir():
#         # Check whether file is in text format or not
#         if file.endswith(".py"):
#             with open (file, "r") as file:
#                 # pythonASTs.append(cst.parse_expression)
#                 pythonASTs.append(ast.parse(file.read()))
#                 # print(i, tree)
#                 # i += 1

def assignLabels():
    with open (merge, "r") as file:
        pythonASTs.append(ast.parse(file.read()))
    return pythonASTs

