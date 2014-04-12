'''

Author: Brian Sinclair

Editted: March 2014

Version: Python 2.7


'''
import numpy as np
import pybrain as pb
from pybrain.datasets import SupervisedDataSet

net = buildNetwork(2, 3, 1) # network that has two inputs, three hidden and a single output neuron.
ds = SupervisedDataSet(2, 1) #create dataset 2 inputs, 1 output

#How to create a hidden layer 
net = buildNetwork(2, 3, 1, hiddenclass=TanhLayer)
net['hidden0']

#organize data s.t. ([inputs], [output]) -- ([1, 2, 0], [1]) 
