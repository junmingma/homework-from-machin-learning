import numpy as np
from typing import List
from classifier import Classifier

class DecisionStump(Classifier):
	def __init__(self, s:int, b:float, d:int):
		self.clf_name = "Decision_stump"
		self.s = s
		self.b = b
		self.d = d

	def train(self, features: List[List[float]], labels: List[int]):
		pass
		
	def predict(self, features: List[List[float]]) -> List[int]:
		'''
		Inputs:
		- features: the features of all test examples
   
		Returns:
		- the prediction (-1 or +1) for each example (in a list)
		'''
		##################################################
		# TODO: implement "predict"
		##################################################
		pred_label = []
		for example_n in features:
			if example_n[self.d] > self.b:
				pred_label.append(self.s)
			else:
				pred_label.append(-self.s)
		return pred_label