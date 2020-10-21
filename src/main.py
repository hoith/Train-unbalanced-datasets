import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
sns.set()

import argparse

from utils import load_data, f1, accuracy_score, load_dictionary, dictionary_info
from tree import DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier

def load_args():

	parser = argparse.ArgumentParser(description='arguments')
	parser.add_argument('--county_dict', default=1, type=int)
	parser.add_argument('--decision_tree', default=1, type=int)
	parser.add_argument('--random_forest', default=1, type=int)
	parser.add_argument('--ada_boost', default=1, type=int)
	parser.add_argument('--root_dir', default='../data/', type=str)
	args = parser.parse_args()

	return args


def county_info(args):
	county_dict = load_dictionary(args.root_dir)
	dictionary_info(county_dict)

def decision_tree_testing(x_train, y_train, x_test, y_test):
	n = 1
	print('Decision Tree depth: ',n)
	clf = DecisionTreeClassifier(max_depth=n)
	clf.fit(x_train, y_train)
	preds_train = clf.predict(x_train)
	preds_test = clf.predict(x_test)
	train_accuracy = accuracy_score(preds_train, y_train)
	test_accuracy = accuracy_score(preds_test, y_test)
	print('Train {}'.format(train_accuracy))
	print('Test {}'.format(test_accuracy))
	preds = clf.predict(x_test)
	print('F1 Test {}'.format(f1(y_test, preds)))



def random_forest_testing(x_train, y_train, x_test, y_test):
	print('Random Forest\n\n')
	rclf = RandomForestClassifier(max_depth=7, max_features=11, n_trees=50)
	rclf.fit(x_train, y_train)
	preds_train = rclf.predict(x_train)
	preds_test = rclf.predict(x_test)
	train_accuracy = accuracy_score(preds_train, y_train)
	test_accuracy = accuracy_score(preds_test, y_test)
	print('Train {}'.format(train_accuracy))
	print('Test {}'.format(test_accuracy))
	preds = rclf.predict(x_test)
	print('F1 Test {}'.format(f1(y_test, preds)))

def ababoost(x_train, y_train, x_test, y_test):
	print('Ababoost\n\n')
	leni = len(x_train)
	L = 3
	D = np.array([1/leni]*leni) #The first D is 1/length of train set
	bclf = AdaBoostClassifier()
	for i in range(L):
		preds_train, preds_test, D,we = bclf.adaboost(x_train,y_train,x_test,y_test,D)
	y_train[y_train==0] = -1
	y_test[y_test==0] = -1
	train_accuracy = accuracy_score(preds_train, y_train)
	test_accuracy = accuracy_score(preds_test, y_test)
	print('L = ',L)
	print(D)
	print('Train {}'.format(train_accuracy))
	print('Test {}'.format(test_accuracy))
	print('F1 Train {}'.format(f1(y_train, preds_train)))
	print('F1 Test {}'.format(f1(y_test, preds_test)))
	print('we = ',we)


###################################################
# Modify for running your experiments accordingly #
###################################################
if __name__ == '__main__':
	args = load_args()
	x_train, y_train, x_test, y_test = load_data(args.root_dir)
	if args.county_dict == 1:
		county_info(args)
#	if args.decision_tree == 1:
#		decision_tree_testing(x_train, y_train, x_test, y_test)
#	if args.random_forest == 1:
#		random_forest_testing(x_train, y_train, x_test, y_test)
	if args.ada_boost == 1:
		ababoost(x_train, y_train, x_test, y_test)

	print('Done')
	
	





