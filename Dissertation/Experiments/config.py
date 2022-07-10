import sys, importlib.util
from os.path import dirname, join
import Networks

currentDirectory = dirname(__file__)
pathSplit = "/Experiments"
head = currentDirectory.split(pathSplit)
path = head[0]

# IMPORT MLP MODULE
mlp = join(path, './Networks/MLP.py')
location1 = importlib.util.spec_from_file_location("MLP", mlp)
mlp = importlib.util.module_from_spec(location1)
sys.modules["MLP"] = mlp
location1.loader.exec_module(mlp)


# IMPORT NAÏVE BAYES MODULE
nbc = join(path, './Networks/NaiveBayes.py')
location2 = importlib.util.spec_from_file_location("NBClassier", nbc)
nbc = importlib.util.module_from_spec(location2)
sys.modules["NBClassifier"] = nbc
location2.loader.exec_module(nbc)

# IMPORT RNN MODULE
rnn = join(path, './Networks/RNN.py')
location3 = importlib.util.spec_from_file_location("RNN", rnn)
rnn = importlib.util.module_from_spec(location3)
sys.modules["RNN"] = rnn
location3.loader.exec_module(rnn)

# IMPORT SKLEARN MODULE
scikit = join(path, './Networks/MLP.py')
location4 = importlib.util.spec_from_file_location("MLP", scikit)
scikit = importlib.util.module_from_spec(location4)
sys.modules["SKLearnClassifers"] = scikit
location4.loader.exec_module(scikit)

# IMPORT GRAPH DATA PROCESSOR 
gdp = join(path, './ParsingAndEmbeddingLayers/Graphs/GraphDataProcessor.py')
location5 = importlib.util.spec_from_file_location("GraphDataProcessor", gdp)
gdp = importlib.util.module_from_spec(location5)
sys.modules["GDP"] = gdp
location5.loader.exec_module(gdp)


