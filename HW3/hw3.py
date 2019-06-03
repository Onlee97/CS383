import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import FormatStrFormatter
import random
from numpy.linalg import inv

def G(x):
	# return (x - 1)**4
	return (x-1)*(x-1)*(x-1)*(x-1)

def gradient(x):
	# return 4*(x - 1)**3
	return 4*(x-1)*(x-1)*(x-1)

def fix_learning_rate(n):
	x = 0
	x_arr = []
	G_x_arr = []
	theta = 2**(-23)
	while True:
		old_x = x
		x_arr.append(x)
		G_x_arr.append(G(x))
		y = gradient(x)
		x -= n*y
		if x > 10:
			break
		if abs(x - old_x) < theta:
			break
	return x_arr, G_x_arr, n

def plot_fix_learning_rate(n):
	x_arr, G_x_arr, n = fix_learning_rate(0.1)
	plt.figure()
	plt.plot(x_arr)
	plt.xlabel('Iterations')
	plt.ylabel('X')
	title = "X vs Iterations, n = " + str(n)
	plt.title(title)

	
	plt.figure()
	plt.plot(G_x_arr)
	plt.xlabel('Iterations')
	plt.ylabel('g(x)')
	title = "g(x) vs Iterations, n = " + str(n)
	plt.title(title)
	plt.show()

def ad_learning_rate_2():
	x = 0
	x_arr = []
	G_x_arr = []
	theta = 2**(-23)
	n = 1.0
	g = 1
	# while True:
	for i in range(50):
		old_x = x
		old_g = g
		x_arr.append(x)
		G_x_arr.append(G(x))
		g = gradient(x)
		if g*old_g < 0:
			n = n/2
		x -= n*g
		if abs(x - old_x) < theta:
			break
	return x_arr, G_x_arr, n

def ad_learning_rate_2():
	x = 0
	x_arr = []
	G_x_arr = []
	theta = 2**(-23)
	n = 1.0
	g = 1
	# while True:
	for i in range(50):
		old_x = x
		old_g = g
		x_arr.append(x)
		G_x_arr.append(G(x))
		g = gradient(x)
		if g*old_g < 0:
			n = n/2
		x -= n*g
		if abs(x - old_x) < theta:
			break
	return x_arr, G_x_arr, n


def ad_learning_rate():
	x = 0
	x_arr = []
	G_x_arr = []
	theta = 2**(-23)
	n = 1.0
	g = 1
	# while True:
	while(True):
		old_x = x
		old_g = g
		x_arr.append(x)
		G_x_arr.append(G(x))

		g = gradient(x)
		if g*old_g < 0:
			n = n/2
		x -= n*g
		if abs(x - old_x) < theta:
			break
		# print(x)
	return x_arr, G_x_arr, n


def plot_adaptive_learning_rate():
	x_arr, G_x_arr, n = ad_learning_rate()
	plt.figure()
	plt.plot(x_arr)
	plt.xlabel('Iterations')
	plt.ylabel('X')
	title = "X vs Iterations, Adaptive Learning rate"
	plt.title(title)

	
	plt.figure()
	plt.plot(G_x_arr)
	plt.xlabel('Iterations')
	plt.ylabel('g(x)')
	title = "g(x) vs Iterations, Adaptive Learning Rate"
	plt.title(title)
	plt.show()

def standardize(array):
	m = np.mean(array)
	s = np.std(array)
	array[:] = [x - m for x in array]
	array[:] = [x / s for x in array]	
	return array, m, s

def extract_train_test_data(filename):
	with open(filename) as f:
		ncols = len(f.readline().split(','))
	data = np.loadtxt(filename, delimiter = ',', usecols = range(1,ncols), ndmin = 2, skiprows = 1)
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
	# print(train_mean, train_stdev)
	# print("test data: ", len(test_data))
	# print(test_data)
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
	# print(test_data)
	return train_data, test_data

def extract_data_and_label(data):
	X = np.ones((len(data), len(data[0])))
	X[:,1:] = data[:,:-1]
	Y = data[:, -1].reshape(len(data), 1)
	return X, Y


def close_form_sol(X, Y):
	XTX = np.matmul(X.T, X)
	inv_XTX = inv(XTX)
	inv_XTX_XT = np.matmul(inv_XTX, X.T)
	theta = np.matmul(inv_XTX_XT, Y)
	
	return theta

def RMSE(Y_test, Y_predict):
	error_sum = 0
	N = len(Y_predict)
	for i in range(N):
		error_sum += (Y_predict[i] - Y_test[i])**2
	rmse = math.sqrt(error_sum/N)
	return rmse

def SE(Y_test, Y_predict):
	error_sum = 0
	N = len(Y_predict)
	for i in range(N):
		error_sum += (Y_predict[i] - Y_test[i])**2
	return error_sum, len(Y_predict)

def RMSE_S_fold(error_list, N):
	s = np.sum(error_list)
	rmse = math.sqrt(s/N)
	return rmse


def linear_regression():
	data = extract_train_test_data("x06Simple.csv")
	train_data, test_data = pick_train_test_data(data)
	train_data, test_data = prepare_data(train_data, test_data)
	X, Y = extract_data_and_label(train_data)
	theta = close_form_sol(X, Y)
	print("Theta Matrix of the close form solution:")
	print(theta)

	X_test, Y_test = extract_data_and_label(test_data)
	Y_predict = np.matmul(X_test, theta)
	rmse = RMSE(Y_test, Y_predict)
	print("RMSE = ", rmse)

def pick_train_test_from_S_fold(data, s_num, s_index):
	N = len(data)
	if s_num >= N:
		s_num = N
		test_data = data[s_index].reshape(1,3)
		train_data = data[np.arange(N) != s_index]
		# print(train_data)
	else:
		fold_size = math.ceil(N/s_num)
		test_index = math.ceil(s_index*N/s_num)
	# test_last_index = test_index + fold_size
	# if test_last_index
		test_data = data[range(test_index, test_index + fold_size -1)]

		train_index = np.append(range(test_index), range(test_index + fold_size - 1, N))
		train_index.astype(int)
	
		train_data = []
		for i in range(test_index):
			train_data.append(data[i])
		for i in range(test_index + fold_size, N):
			train_data.append(data[i])
	
		train_data = np.array(train_data)
	return train_data, test_data

def S_fold(S):
	data = extract_train_test_data("x06Simple.csv")
	print("Number of Fold: ", S)
	rmse_list = []
	for i in range(20):
		s_data = data[:]
		np.random.shuffle(s_data)		
		error_list = []
		N = 0
		for i in range(S):
			train_data, test_data = pick_train_test_from_S_fold(s_data, S, i)
			train_data, test_data = prepare_data(train_data, test_data)
			X, Y = extract_data_and_label(train_data)
			theta = close_form_sol(X, Y)
			X_test, Y_test = extract_data_and_label(test_data)
			Y_predict = np.matmul(X_test, theta)
			error, num = SE(Y_test, Y_predict)
			error_list.append(error)
			N += num

		error_list = np.array(error_list)
		rmse_list.append(RMSE_S_fold(error_list, N))

	avg_RMSE = np.mean(rmse_list)
	std_RMSE = np.std(rmse_list)
	print("Average RMSE: ", avg_RMSE, ". Standard Deviation RMSE: ", std_RMSE)

def main():
	print("Gradient Descent With Fix Learning Rate")
	plot_fix_learning_rate(0.1)
	print("------------")
	print("Gradient Descent With Adaptive Learning Rate")
	plot_adaptive_learning_rate()
	print("------------")
	print("Close Form Linear Regression")
	linear_regression()
	print("------------")
	print("S_fold")
	S_fold(3)
	S_fold(5)
	S_fold(20)
	S_fold(44)
	

if __name__ == "__main__":
	main()
