from GraphDataProcessor import GraphDataProcessor

hashed = True
gdp = GraphDataProcessor(hashed)
"""RUNNING ON PADDED GRAPHS"""
x_train, y_train, x_test, y_test = gdp.runProcessor4()


