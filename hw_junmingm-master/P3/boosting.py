import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs      # set of weak classifiers to be considered
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		'''
		Inputs:
		- features: the features of all test examples
   
		Returns:
		- the prediction (-1 or +1) for each example (in a list)
		'''
		########################################################
		# TODO: implement "predict"
		########################################################
		pred_label = [0]*len(features)
		for t in range(self.T):
			pred_weak = self.clfs_picked[t].predict(features)
			for n in range(len(features)):
				pred_label[n] += self.betas[t] * pred_weak[n]
		for i in range(len(pred_label)):
			if pred_label[i] > 0:
				pred_label[i] = 1
			else:
				pred_label[i] = -1
		return pred_label

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		'''
		Inputs:
		- features: the features of all examples
		- labels: the label of all examples
   
		Require:
		- store what you learn in self.clfs_picked and self.betas
		'''
		############################################################
		# TODO: implement "train"
		############################################################
		N = len(features)
		D = [1 / float(N)] * N
		
		
		for t in range(self.T):
			epsi_t = 10000.0
			for h in self.clfs:
				pred_label = h.predict(features)
				epsi = 0
				for i in range(N):
					if labels[i] != pred_label[i]:
						epsi += D[i]
				if epsi < epsi_t:
					epsi_t = epsi
					h_t = h
			beta_t = 0.5 * np.log((1.0 - epsi_t) / epsi_t)
			self.clfs_picked.append(h_t)
			self.betas.append(beta_t)

			pred_labels = h_t.predict(features)
			for i in range(N):
				if labels[i] == pred_labels[i]:
					D[i] *= np.exp(-beta_t)
				else:
					D[i] *= np.exp(beta_t)
			S = float(sum(D))
			for i in range(N):
				D[i] = float(D[i])/S


		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)



	