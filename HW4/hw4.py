import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import FormatStrFormatter
import random
from numpy.linalg import inv
from math import log, exp

def standardize(array):
	m = np.mean(array)
	s = np.std(array)
	array[:] = [x - m for x in array]
	array[:] = [x / s for x in array]	
	return array, m, s

def extract_data(filename):
	data = np.loadtxt(filename, delimiter = ',', ndmin = 2)
	return data

def pick_train_test_data(data):
	np.random.shuffle(data)
	train_index = math.ceil(len(data)*2/3)
	train_data = data[range(train_index)]
	test_data = data[range(train_index, len(data))]
	return train_data, test_data

def prepare_data(train_data, test_data):
	train_mean = []
	train_stdev = []
	index = 0

	for column in train_data.T[:-1]:
		column, m, s = standardize(column)
		train_mean.append(m)
		train_stdev.append(s)
	i = 0

	if len(test_data) > 1:
		for column in test_data.T[:-1]:
			column[:] = [x - train_mean[i] for x in column]
			column[:] = [x / train_stdev[i] for x in column]
			i += 1
	else:
		test_data_std = []
		for element in test_data[0][:-1]:
			element = element - train_mean[i]
			element = element / train_stdev[i]
			test_data_std.append(element)
			i += 1
		test_data_std.append(test_data[0][-1])
		test_data = np.array(test_data_std).reshape(1,3) 
	return train_data, test_data

def extract_data_and_label(data):
	X = data[:,:-1]
	Y = data[:, -1].reshape(len(data), 1)
	return X, Y

def divide_class(data):
	spam = []
	non_spam = []
	for sample in data:
		if sample[-1] == 1:
			spam.append(sample)
		else:
			non_spam.append(sample)
	spam = np.array(spam)
	non_spam = np.array(non_spam)
	p_spam = len(spam)/len(data)
	p_non_spam = len(non_spam)/len(data)
	return spam, non_spam, p_spam, p_non_spam

def normal_model(data):
	mean = []
	stdev = []
	for column in data.T[:-1]:
		mean.append(np.mean(column))
		stdev.append(np.std(column, ddof = 1))
		# print(np.std(column, ddof = 1))
		# print(np.mean(column))
	mean = np.array(mean)
	stdev = np.array(stdev)
	return mean, stdev

def gaussian(x_k, x_mean, x_std):
	if x_std == 0:
		return 0
	result = 1/(x_std*2.506628275)
	p = -((x_k - x_mean)**2)/(2*(x_std**2))
	result *= math.exp(p)
	# print(result)
	# if result != 0:
	# 	result = math.log(result)
	return result

def bayes_prob(x, x_mean, x_std, p_class):
	P = p_class
	for i in range(len(x)):
		P *= gaussian(x[i], x_mean[i], x_std[i])
	return P

def bayes_prob_2(x, x_mean, x_std, p_class):
	P = log(p_class)
	for i in range(len(x)):
		g = gaussian(x[i], x_mean[i], x_std[i])
		if g != 0:
			g = log(g)
		P += g
	return P

def eval_result(Y_predict, Y_test):
	TP = 0
	TN = 0
	FP = 0
	FN = 0
	for i in range(len(Y_predict)):
		if Y_predict[i] == 1 and Y_test[i] == 1:
			TP += 1
		elif Y_predict[i] == 1 and Y_test[i] == 0:
			FP += 1
		elif Y_predict[i] == 0 and Y_test[i] == 0:
			TN += 1			
		elif Y_predict[i] == 0 and Y_test[i] == 1:
			FN += 1
	precision = TP/(TP+FP)
	recall = TP/(TP+FN)
	F = 2*precision*recall/(precision + recall)
	accuracy = (TP+TN)/(TP+TN+FP+FN)
	print("precision: %f\nRecall:    %f\nF-Measure: %f\nAccuracy:  %f" % (precision, recall, F, accuracy))	
	return precision, recall	


def naive_bayes():
	data = extract_data("spambase.data")
	while (1):
		pass
		train_data, test_data = pick_train_test_data(data)
		train_data, test_data = prepare_data(train_data, test_data)
		
		spam_data, non_spam_data, p_spam, p_non_spam = divide_class(train_data)
		mean_spam, stdev_spam = normal_model(spam_data)
		mean_non_spam, stdev_non_spam = normal_model(non_spam_data)

		X_test, Y_test = extract_data_and_label(test_data)
		Y_predict = []
		for sample in X_test:
			Ps = bayes_prob(sample, mean_spam, stdev_spam, p_spam)
			Pns = bayes_prob(sample, mean_non_spam, stdev_non_spam, p_non_spam)
			if Ps >= Pns:
				Y_predict.append(1)
			else:
				Y_predict.append(0)
		Y_predict = np.array(Y_predict).reshape(len(Y_predict), 1)
		precision, recall =  eval_result(Y_predict, Y_test)
		if precision - 0.69 < 0.5:
			break

def eval_result_recall_precision(Y_predict, Y_test):
	TP = 0
	TN = 0
	FP = 0
	FN = 0
	for i in range(len(Y_predict)):
		if Y_predict[i] == 1 and Y_test[i] == 1:
			TP += 1
		elif Y_predict[i] == 1 and Y_test[i] == 0:
			FP += 1
		elif Y_predict[i] == 0 and Y_test[i] == 0:
			TN += 1			
		elif Y_predict[i] == 0 and Y_test[i] == 1:
			FN += 1
	if TP+FP == 0:
		precision = 1
	else:
		precision = TP/(TP+FP)
	if TP + FN == 0:
		recall = 1
	else:
		recall = TP/(TP+FN)
	# print("precision: %f\nRecall:    %f\n" % (precision, recall))	
	return precision, recall

def extra_credit():
	data = extract_data("sex_h_w.csv")
	train_data, test_data = pick_train_test_data(data)
	train_data, test_data = prepare_data(train_data, test_data)
	
	spam_data, non_spam_data, p_spam, p_non_spam = divide_class(train_data)
	mean_spam, stdev_spam = normal_model(spam_data)
	mean_non_spam, stdev_non_spam = normal_model(non_spam_data)

	X_test, Y_test = extract_data_and_label(test_data)
	
	threshold = 0
	
	Ps = []
	Pns = []
	for sample in X_test:
		Ps.append((bayes_prob(sample, mean_spam, stdev_spam, p_spam)))
		Pns.append((bayes_prob(sample, mean_non_spam, stdev_non_spam, p_non_spam)))
	pre = []
	rec = []
	while threshold <= 1.05:	
		# print(Ps[0])
		Y_predict = []
		for i in range(len(Ps)):
			a = Ps[i]
			b = Pns[i]
			if (a + b) == 0 or a/(a + b) >= threshold:	
				Y_predict.append(1)
			else:
				Y_predict.append(0)
		Y_predict = np.array(Y_predict).reshape(len(Y_predict), 1)

		precision, recall = eval_result_recall_precision(Y_predict, Y_test)
		pre.append(precision)
		rec.append(recall)
		threshold += 0.05


	plt.figure()
	plt.plot(rec, pre, '-o')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	title = "Precision vs Recall"
	plt.title(title)
	plt.show()


def main():
	print("Perform naive_bayes")
	naive_bayes()
	print("Perform naive_bayes on new data set\nThe Dataset is weight (1st column), height (2nd column) and Gender (3rd column)")
	extra_credit()	


if __name__ == "__main__":
	main()