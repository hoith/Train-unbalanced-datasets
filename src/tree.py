import numpy as np
import math

class Node():
	"""
	Node of decision tree

	Parameters:
	-----------
	prediction: int
		Class prediction at this node
	feature: int
		Index of feature used for splitting on
	split: int
		Categorical value for the threshold to split on for the feature
	left_tree: Node
		Left subtree
	right_tree: Node
		Right subtree
	"""
	def __init__(self, prediction, feature, split, left_tree, right_tree):
		self.prediction = prediction
		self.feature = feature
		self.split = split
		self.left_tree = left_tree
		self.right_tree = right_tree


class DecisionTreeClassifier():
	"""
	Decision Tree Classifier. Class for building the decision tree and making predictions

	Parameters:
	------------
	max_depth: int
		The maximum depth to build the tree. Root is at depth 0, a single split makes depth 1 (decision stump)
	"""

	def __init__(self, max_depth=None):
		self.max_depth = max_depth

	# take in features X and labels y
	# build a tree
	def fit(self, X, y):
		self.num_classes = len(set(y))
		self.root = self.build_tree(X, y, depth=1)


	# make prediction for each example of features X
	def predict(self, X):
		preds = [self._predict(example) for example in X]

		return preds

	# prediction for a given example
	# traverse tree by following splits at nodes
	def _predict(self, example):
		node = self.root
		while node.left_tree:
			if example[node.feature] < node.split:
				node = node.left_tree
			else:
				node = node.right_tree
		return node.prediction

	# accuracy
	def accuracy_score(self, X, y):
		preds = self.predict(X)
		accuracy = (preds == y).sum()/len(y)
		return accuracy

	# function to build a decision tree
	def build_tree(self, X, y, depth):
		num_samples, num_features = X.shape
		# which features we are considering for splitting on
		self.features_idx = np.arange(0, X.shape[1])

		# store data and information about best split
		# used when building subtrees recursively
		best_feature = None
		best_split = None
		best_gain = 0.0
		best_left_X = None
		best_left_y = None
		best_right_X = None
		best_right_y = None

		# what we would predict at this node if we had to
		# majority class
		num_samples_per_class = [np.sum(y == i) for i in range(self.num_classes)]
		prediction = np.argmax(num_samples_per_class)

		# if we haven't hit the maximum depth, keep building
		if depth <= self.max_depth:
			# consider each feature
			for feature in self.features_idx:
				# consider the set of all values for that feature to split on
				possible_splits = np.unique(X[:, feature])
				for split in possible_splits:
					# get the gain and the data on each side of the split
					# >= split goes on right, < goes on left
					gain, left_X, right_X, left_y, right_y = self.check_split(X, y, feature, split)
					# if we have a better gain, use this split and keep track of data
					if gain > best_gain:
						best_gain = gain
						best_feature = feature
						best_split = split
						best_left_X = left_X
						best_right_X = right_X
						best_left_y = left_y
						best_right_y = right_y
		
		# if we haven't hit a leaf node
		# add subtrees recursively
		if best_gain > 0.0:
			left_tree = self.build_tree(best_left_X, best_left_y, depth=depth+1)
			right_tree = self.build_tree(best_right_X, best_right_y, depth=depth+1)
			return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=left_tree, right_tree=right_tree)

		# if we did hit a leaf node
		return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=None, right_tree=None)


	# gets data corresponding to a split by using numpy indexing
	def check_split(self, X, y, feature, split):
		left_idx = np.where(X[:, feature] < split)
		right_idx = np.where(X[:, feature] >= split)
		left_X = X[left_idx]
		right_X = X[right_idx]
		left_y = y[left_idx]
		right_y = y[right_idx]

		# calculate gini impurity and gain for y, left_y, right_y
		gain = self.calculate_gini_gain(y, left_y, right_y)
		return gain, left_X, right_X, left_y, right_y

	def calculate_gini_gain(self, y, left_y, right_y):
		# not a leaf node
		# calculate gini impurity and gain
		gain = 0.
		yp,yn = 0,0
		lyp,lyn = 0,0
		ryp,ryn = 0,0
		if len(left_y) > 0 and len(right_y) > 0:
			for n in y:
				if n>0:
					yp+=1
				else:
					yn+=1
			for n in left_y:
				if n>0:
					lyp+=1
				else:
					lyn+=1
			for n in right_y:
				if n>0:
					ryp+=1
				else:
					ryn+=1
			ua = 1- np.power((yp/(yp+yn)),2) - np.power((yn/(yp+yn)),2) 
			ual = 1-np.power((lyp/(lyp+lyn)),2)  - np.power((lyn/(lyp+lyn)),2) 
			uar = 1-np.power((ryp/(ryp+ryn)),2) - np.power((ryn/(ryp+ryn)),2) 
			pl = (lyp+lyn)/(yp+yn)
			pr = (ryp+ryn)/(yp+yn)
			gain = ua - np.multiply(pl,ual) - np.multiply(pr,uar)

			return gain
		# we hit leaf node
		# don't have any gain, and don't want to divide by 0
		else:
			return 0
		
class RandomForestClassifier():
	"""
	Random Forest Classifier. Build a forest of decision trees.
	Use this forest for ensemble predictions

	YOU WILL NEED TO MODIFY THE DECISION TREE VERY SLIGHTLY TO HANDLE FEATURE BAGGING

	Parameters:
	-----------
	n_trees: int
		Number of trees in forest/ensemble
	max_features: int
		Maximum number of features to consider for a split when feature bagging
	max_depth: int
		Maximum depth of any decision tree in forest/ensemble
	"""
	def __init__(self, n_trees, max_features, max_depth):
		self.n_trees = n_trees
		self.max_features = max_features
		self.max_depth = max_depth

		##################
		# YOUR CODE HERE #
		##################

	# fit all trees
	def fit(self, X, y):
		bagged_X, bagged_y = self.bag_data(X, y)
		print('Fitting Random Forest...\n')
		for i in range(self.n_trees):
			print(i+1, end='\t\r')
			##################
			# YOUR CODE HERE #
			##################
		print()

	def bag_data(self, X, y, proportion=1.0):
		bagged_X = []
		bagged_y = []
		for i in range(self.n_trees):
			continue
			##################
			# YOUR CODE HERE #
			##################

		# ensure data is still numpy arrays
		return np.array(bagged_X), np.array(bagged_y)


	def predict(self, X):
		preds = []

		# remove this one \/
		preds = np.ones(len(X)).astype(int)
		# ^that line is only here so the code runs

		##################
		# YOUR CODE HERE #
		##################
		return preds


class AdaBoostClassifier():
	def __init__(self):
		pass

	def adaboost(self,X,y,Xt,yt,D):
		blf = DTC_For_Ada(max_depth=1,Dt=D)
		blf.fit(X, y)
		preds_train = blf.predict(X)
		preds_test = blf.predict(Xt)
		leni = len(X)
		we = 0
		y[y==0] = -1
		for i in range(leni):
			if preds_train[i] != y[i]:
				we+=D[i]
		a=0.5*np.log((1-we)/we)
		for i in range(leni):
			if preds_train[i] != y[i]:
				D[i] = D[i]*math.exp(a)
			else:
				D[i] = D[i]*math.exp(-a)
		Dsum= sum(D)
		for i in range(leni):
			D[i] = D[i]/Dsum
		return preds_train, preds_test, D,we


# A slightly changed DTC with weight changed.
class DTC_For_Ada():
	def __init__(self, max_depth, Dt):
		self.max_depth = max_depth
		self.Dt=Dt
		
	# take in features X and labels y
	# build a tree
	def fit(self, X, y):
		self.num_classes = len(set(y))
		self.root = self.build_tree(X, y, depth=1)


	# make prediction for each example of features X
	def predict(self, X):
		preds = [self._predict(example) for example in X]

		return preds

	# prediction for a given example
	# traverse tree by following splits at nodes
	def _predict(self, example):
		node = self.root
		while node.left_tree:
			if example[node.feature] < node.split:
				node = node.left_tree #expected true
			else:
				node = node.right_tree #expected false
		return node.prediction

	# accuracy
	def accuracy_score(self, X, y):
		preds = self.predict(X)
		accuracy = (preds == y).sum()/len(y)
		return accuracy
		# function to build a decision tree
	def build_tree(self, X, y, depth):
		num_samples, num_features = X.shape
		# which features we are considering for splitting on
		self.features_idx = np.arange(0, X.shape[1])

		# store data and information about best split
		# used when building subtrees recursively
		best_feature = None
		best_split = None
		best_error = 1.0
		best_left_X = None
		best_left_y = None
		best_right_X = None
		best_right_y = None
		# what we would predict at this node if we had to
		# majority class
		num_samples_per_class = [np.sum(y == i) for i in range(self.num_classes)]
		prediction = np.argmax(num_samples_per_class)
		if(prediction == 0):
			prediction = -1
		# if we haven't hit the maximum depth, keep building
		if depth <= self.max_depth:
			# consider each feature
			for feature in self.features_idx:
				# consider the set of all values for that feature to split on
				possible_splits = np.unique(X[:, feature])
				for split in possible_splits:
					# get the gain and the data on each side of the split
					# >= split goes on right, < goes on left
					error, left_X, right_X, left_y, right_y = self.check_split(X, y, feature, split)
					# if we have a better gain, use this split and keep track of data
					if error < best_error:
						best_error = error 
						best_feature = feature
						best_split = split
						best_left_X = left_X
						best_right_X = right_X
						best_left_y = left_y
						best_right_y = right_y
		# if we haven't hit a leaf node
		# add subtrees recursively
		if best_error < 1.0:
			left_tree = self.build_tree(best_left_X, best_left_y, depth=depth+1)
			right_tree = self.build_tree(best_right_X, best_right_y, depth=depth+1)
			return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=left_tree, right_tree=right_tree)

		# if we did hit a leaf node
		return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=None, right_tree=None)



	# gets data corresponding to a split by using numpy indexing
	def check_split(self, X, y, feature, split):
		left_idx = np.where(X[:, feature] < split)
		right_idx = np.where(X[:, feature] >= split)
		left_X = X[left_idx]
		right_X = X[right_idx]
		left_y = y[left_idx]
		right_y = y[right_idx]
		left_D = self.Dt[left_idx]
		right_D = self.Dt[right_idx]

		# calculate gini impurity and gain for y, left_y, right_y
		error = self.calculate_weighted_error(y, left_y, right_y,left_D,right_D)
		return error, left_X, right_X, left_y, right_y

	def calculate_weighted_error(self, y, left_y, right_y,left_D,right_D):
		# not a leaf node
		# calculate gini impurity and gain
		error = 1.0
		yp,yn = 0,0
		lyp,lyn = 0,0
		ryp,ryn = 0,0
		yd,lyd,ryd=0,0,0

		if len(left_y) > 0 and len(right_y) > 0:
			for n in y:
				if n>0:
					yp+=self.Dt[yd]
					yd+=1
				else:
					yn+=self.Dt[yd]
					yd+=1
			for n in left_y: #Expected positive
				if n>0:
					lyp+=left_D[lyd]
					lyd+=1
				else:
					lyn+=left_D[lyd]
					lyd+=1
			for n in right_y:  #Expected negative
				if n>0:
					ryp+=right_D[ryd]
					ryd+=1
				else:
					ryn+=right_D[ryd]
					ryd+=1
			error = ryp + lyn #weighted false nagative+weighted false positive
			return error
		# we hit leaf node
		# not choose it
		else:
			return 1

