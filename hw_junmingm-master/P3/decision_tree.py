import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert(len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels)+1

		# build the tree
		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()

		return
		
	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')
	 	
		string = ''
		for idx_cls in range(node.num_cls):
			string += str(node.labels.count(idx_cls)) + ' '
		print(indent + ' num of sample / cls: ' + string)

		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent+'}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label # majority of current node

		if len(np.unique(labels)) < 2:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None # the index of the feature to be split

		self.feature_uniq_split = None # the possible unique values of the feature to be split


	def split(self):
		def conditional_entropy(branches: List[List[int]]) -> float:
			'''
			branches: C x B array, 
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of 
					  corresponding training samples 
					  e.g.
					              ○ ○ ○ ○
					              ● ● ● ●
					            ┏━━━━┻━━━━┓
				               ○ ○       ○ ○
				               ● ● ● ●
				               
				      branches = [[2,2], [4,0]]
			'''
			########################################################
			# TODO: compute the conditional entropy
			########################################################
			cond_entropy = 0.0
			branches = np.array(branches).T
			num_branch = len(branches)
			num_examples = np.sum(branches)
			examples_perbranch = np.sum(branches,axis = 1)
			entro = []
			for i in range(num_branch):
				h_p = 0
				for classes in branches[i]:
					prob = classes / float(examples_perbranch[i])
					if prob > 0:
						h_p -= prob * np.log2(prob)
				entro.append(h_p)
			for i in range(num_branch):
				cond_entropy += examples_perbranch[i] / float(num_examples) * entro[i]
			return cond_entropy


		features = np.array(self.features)
		labels = np.array(self.labels)
		if features.shape[1] == 0:
			self.splittable = False
		C = np.unique(labels)
		min_entropy = 10000.0
		for idx_dim in range(len(self.features[0])):
		############################################################
		# TODO: compare each split using conditional entropy
		#       find the best split
		############################################################
			if self.splittable:
				B = np.unique(features[:, idx_dim])
				branches = np.zeros((C.shape[0], B.shape[0]))
			
				for n in range(len(features)):
					cn, = np.where(C == labels[n])
					bn, = np.where(B == features[n][idx_dim])
					branches[cn[0]][bn[0]] += 1
				cond_entropy_i = conditional_entropy(branches)
				if cond_entropy_i <= min_entropy:
					min_entropy = cond_entropy_i
					self.dim_split = idx_dim


		############################################################
		# TODO: split the node, add child nodes
		############################################################
		if (features.shape[1]>1):
			features_split = features[:, self.dim_split]
		elif (features.shape[1]==1):
			features_split = np.reshape(features,(features.shape[0],))
		else:
			features_split = np.array([])


		self.feature_uniq_split = np.unique(features_split).tolist()


		if self.splittable :
			features_del = np.delete(features, self.dim_split, axis=1)
			for k in range(len(self.feature_uniq_split)):
				child_idx = np.where(features_split == np.unique(features_split)[k])
				child_features = features_del[child_idx]
				child_labels = labels[child_idx]

				child_num_cls = np.max(child_labels) + 1
				child_node = TreeNode(child_features.tolist(), child_labels.tolist(), child_num_cls)
				
				self.children.append(child_node)



		# split the child nodes
		for child in self.children:
			if child.splittable:
				child.split()

		return

	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])
			feature = feature[:self.dim_split] + feature[self.dim_split+1:]
			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max


