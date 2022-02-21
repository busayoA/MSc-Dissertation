#!/usr/local/bin/python3

import tensorflow as tf
import numpy as numpy
print(tf.__version__)

class RNN:
    def __init__(self, layers):
        self.layers = layers