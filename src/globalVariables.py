# Import necessary global variables and modules to be used
import numpy as np
import pickle as pkl
import os
from tensorflow.keras import layers, models, Input, Model
import tensorflow as tf


#
simulationNames = np.array(['CDM','SIDM0.1','SIDM0.3','SIDM1'])
allRedshifts = np.array([0.000, 0.125, 0.250, 0.375])
