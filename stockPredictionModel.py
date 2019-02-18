from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


#Import and randomize data
aaplStockDataframe = pd.read_csv("AAPL_data.csv", sep=",") 
# aaplStockDataframe = aaplStockDataframe.reindex(np.random.permutation(aaplStockDataframe.index))

def preprocess_features(aaplStockDataframe):
	"""Prepares input features from aaplStocks data set.

	Args:
	aaplStockDataframe: A Pandas DataFrame expected to contain data
	  from the aaplStocks data set.
	Returns:
	A DataFrame that contains the features to be used for the model, including
	synthetic features.
	"""
	selectedFeatures = aaplStockDataframe[
		["date",
		"open",
		"high",
		"low",
		"close",
		"volume"]]

	processedFeatures = selectedFeatures.copy()

	#For each day in dataset, add the current 52-week high and low
	fiftyTwoWeekHigh = [processedFeatures["high"][0]]
	fiftyTwoWeekLow = [processedFeatures["low"][0]]

	for high in processedFeatures["high"]:
		if (high > fiftyTwoWeekHigh[-1]):
			fiftyTwoWeekHigh.append(high)
		else:
			fiftyTwoWeekHigh.append(fiftyTwoWeekHigh[-1])
	for low in processedFeatures["low"]:
		if (low <  fiftyTwoWeekLow[-1]):
			fiftyTwoWeekLow.append(low)
		else:
			fiftyTwoWeekLow.append(fiftyTwoWeekLow[-1])

	processedFeatures["fiftyTwoWeekHigh"] = pd.Series(fiftyTwoWeekHigh)
	processedFeatures["fiftyTwoWeekLow"] = pd.Series(fiftyTwoWeekLow)


