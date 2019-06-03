import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numpy import linalg as LA
# from numpy import genfromtxt
import os


def get_imlist(path):
	return [os.path.join(path,f) for f in os.listdir(path) if f.startswith('subject')]

def standardize(array):
	m = np.mean(array)
	s = np.std(array)
	array[:] = [x - m for x in array]
	array[:] = [x / s for x in array]	
	return array, m, s

def unstandardize(array, m, s):
	array[:] = [x * s for x in array]
	array[:] = [x + m for x in array]	
	return array

def get_data_matrix():
	data_matrix = np.empty((154,1600))
	img_list = get_imlist('/home/duyle/Desktop/CS383 HW1/yalefaces')
	img_list.sort()
	index = 0
	for image in img_list: 
		img = Image.open(image)
		img = img.resize((40,40))
		arr = np.array(img)
		flatten_arr = arr.flatten()
		data_matrix[index,:] = flatten_arr
		index = index + 1

	mean = np.empty(1600)
	std = np.empty(1600)
	i = 0
	for column in data_matrix.T:
		column, m, s = standardize(column)
		mean[i] = m
		std[i] = s
		i = i + 1
	np.savetxt('mean_data_matrix.csv', mean, delimiter = ',')
	np.savetxt('std_data_matrix.csv', std, delimiter = ',')	
	np.savetxt('data_matrix.csv', data_matrix, delimiter = ',')
	
def get_eig():
	data_matrix = np.loadtxt('data_matrix.csv', delimiter = ',')
	cov = np.cov(data_matrix.T)
	print(data_matrix)
	w, v = LA.eig(cov)
	np.savetxt('eigvec.csv', v.view(float))
	np.savetxt('eigval.csv', w.view(float))

def pca_2d():
	data_matrix = np.loadtxt('data_matrix.csv', delimiter = ',')
	w = np.loadtxt('eigval.csv').view(complex)
	v = np.loadtxt('eigvec.csv').view(complex)
	print(len(w))
	print(len(v))
	max_indice = np.argpartition(w, -2)[-2:]
	max_eigvec = v[:, [max_indice[0], max_indice[1]]]
	Z = np.matmul(data_matrix, max_eigvec)
	plt.scatter(Z[:,1], Z[:,0])
	plt.show()
	return Z

def chose_eigvec_kd():
	w = np.loadtxt('eigval.csv').view(complex)
	idx = w.argsort()[::-1]
	wsum = np.sum(w)
	for i in range(len(w)-1, -1, -1):
		w = np.delete(w, i)
		if np.sum(w)/wsum <= 0.95:
			print("We need to use ", i, "eigenvectors to include 95% of information")
			return i


# def chose_eigvec_kd():
# 	w = np.loadtxt('eigval.csv').view(complex)
# 	idx = w.argsort()[::-1]
# 	wsum = np.sum(w)
# 	new_w = np.array([])
# 	k = 0
# 	for i in idx:
# 		if np.sum(new_w)/wsum >= 0.95:
# 			print("We need to use ", k, "eigenvectors to include 95% of information")
# 			return i
# 		new_w = np.append(new_w, w[i])
# 		k += 1

def pca_kd():
	data_matrix = np.loadtxt('data_matrix.csv', delimiter = ',')
	v = np.loadtxt('eigvec.csv').view(complex)
	index = chose_eigvec_kd()
	max_eigvec = v[:, np.arange(index)]
	print(max_eigvec)
	Z = np.matmul(data_matrix, max_eigvec)
	rc_data_matrix = np.matmul(Z, max_eigvec.T)
	return rc_data_matrix

def reconstruct_pca_kd():
	rc_data_matrix = pca_kd()
	# data_matrix = np.loadtxt('data_matrix.csv', delimiter = ',')
	mean = np.genfromtxt('mean_data_matrix.csv', delimiter = ',', filling_values = 0)
	std = np.genfromtxt('std_data_matrix.csv', delimiter = ',', filling_values = 0)
	index = 0
	for column in rc_data_matrix.T:
		column = unstandardize(column, mean[index], std[index])
		index = index + 1
	np.savetxt('rc_data_matrix.csv', rc_data_matrix.real, delimiter = ',')

def display_image_kd():
	data_matrix = np.loadtxt('data_matrix.csv', delimiter = ',')
	mean = np.genfromtxt('mean_data_matrix.csv', delimiter = ',', filling_values = 0)
	std = np.genfromtxt('std_data_matrix.csv', delimiter = ',', filling_values = 0)
	
	index = 0
	for column in data_matrix.T:
		column = unstandardize(column, mean[index], std[index])
		index = index + 1
	
	rc_data_matrix = np.loadtxt('rc_data_matrix.csv', delimiter = ',')
	second_pic = np.reshape(rc_data_matrix[0], (40,40))

	first_pic = np.reshape(data_matrix[0], (40,40))
	img = Image.fromarray(first_pic)
	img.show()

	img2 = Image.fromarray(second_pic)
	img2.show()

def chose_eigvec_1d():
	w = np.loadtxt('eigval.csv').view(complex)
	max_indice = np.argmax(w)
	print(max_indice)
	return max_indice

def pca_1d():
	data_matrix = np.loadtxt('data_matrix.csv', delimiter = ',')
	v = np.loadtxt('eigvec.csv').view(complex)
	index = chose_eigvec_1d()
	max_eigvec = v[:, index].reshape((1600,1))
	Z = np.matmul(data_matrix, max_eigvec).reshape((154,1))
	rc_data_matrix = np.matmul(Z, max_eigvec.T)
	return rc_data_matrix

def reconstruct_pca_1d():
	rc_data_matrix = pca_1d()
	# data_matrix = np.loadtxt('data_matrix.csv', delimiter = ',')
	mean = np.genfromtxt('mean_data_matrix.csv', delimiter = ',', filling_values = 0)
	std = np.genfromtxt('std_data_matrix.csv', delimiter = ',', filling_values = 0)
	index = 0
	for column in rc_data_matrix.T:
		column = unstandardize(column, mean[index], std[index])
		index = index + 1
	np.savetxt('rc_data_matrix_1d.csv', rc_data_matrix.real, delimiter = ',')

def display_image_1d():
	rc_data_matrix = np.loadtxt('rc_data_matrix_1d.csv', delimiter = ',')
	second_pic = np.reshape(rc_data_matrix[0], (40,40))
	img2 = Image.fromarray(second_pic)
	img2.show()

def main():
	# Homework 1.2
	# get_data_matrix()
	# get_eig()
	# pca_2d()
	#--------------------------------#
	
	# Homework 1.3	
	# pca_1d()
	# reconstruct_pca_1d()
	# display_image_1d()

	# pca_kd()
	# reconstruct_pca_kd()
	# display_image_kd()


	chose_eigvec_kd()
	#--------------------------------#

	#
if __name__ == "__main__":
	main()