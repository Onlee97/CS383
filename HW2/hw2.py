# Duy Le - CS383
# HW2 - Clustering

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn import decomposition
from mpl_toolkits.mplot3d import Axes3D
import random
import os
import math
import collections


class Cluster:
	def __init__(self, ref, dimension = None):
		self.center = ref
		self.data = []
		if dimension == None:
			self.dimension = 3
		else:
			self.dimension = dimension 

	def clear_data(self):
		self.data = []

	def add_data(self, point):
		self.data.append(point)

	def xyz(self):
		self.data = np.array(self.data)

	def re_center(self):
		self.xyz()
		x_mean = np.mean(self.data.T[0])
		y_mean = np.mean(self.data.T[1])
		z_mean = np.mean(self.data.T[2])
		self.center = [x_mean, y_mean, z_mean]
		self.clear_data()

	def print_data(self):
		print(self.data)

	def plot_cluster_and_return_purity(self, color, ax):
		self.xyz()
		xs = self.data.T[0]
		ys = self.data.T[1]
		zs = self.data.T[2]
		
		xc = self.center[0]
		yc = self.center[1]
		zc = self.center[2]

		#calculate purity	
		label = self.data.T[3]
		count = collections.Counter(label)
		purity = max(count.values())/len(self.data)

		if self.dimension == 3:	
			ax.scatter(xs, ys, zs, marker = "x", s = 10, linewidths = 1, c = color)
			ax.scatter(xc, yc, zc, marker = "o", s = 50,  edgecolors = 'k' ,c = color)		
		else:
			ax.scatter(xs, ys, marker = "x", c = color)
			ax.scatter(xc, yc, marker = "o", s = 50 ,c = color)	

		return purity				

def standardize(array):
	m = np.mean(array)
	s = np.std(array)
	array[:] = [x - m for x in array]
	array[:] = [x / s for x in array]	
	return array

def reduce_dimension(X):
	dimension = len(X[0])
	if (dimension <= 3):
		Y = np.zeros((len(X), 3))
		Y[:,:-1] = X
		return Y
	else: 
		pca = decomposition.PCA(n_components = 3)
		X = pca.fit_transform(X)
		return X

def prepare_data(filename):
	#Change the # of feature of matrix by change usecols = range(start_column, end_column)
	X = np.loadtxt(filename, delimiter = ',', usecols = range(1,9), ndmin = 2)
	print("Number of features: ", len(X[0]))
	label = np.loadtxt(filename, delimiter = ',', usecols = 0, ndmin = 2)
	for column in X.T:
		column = standardize(column)
	Y = np.append(reduce_dimension(X), label, 1)
	return Y

def distance(c, p):
	dx = c[0] - p[0]
	dy = c[1] - p[1]
	dz = c[2] - p[2]
	return dx*dx + dy*dy + dz*dz

def init_cluster_set(X, k):
	index = set()
	length = len(X)
	while len(index) < k:
		index.add(random.randrange(0,length))
	cluster_list = []
	
	#Check dimension type
	if X[0][2] == 0:
		dimension = 2
	else:
		dimension = 3
	#Append cluster to cluster list
	for i in index:
		cluster_list.append(Cluster(X[i], dimension))
	return cluster_list

def chose_cluster(point, cluster_list):
	distance_list = []
	for cluster in cluster_list:
		d = distance(cluster.center, point)
		distance_list.append(d)
	d_min = distance_list.index(min(distance_list))
	cluster_list[d_min].add_data(point)

def divide_data_into_clusters(X, cluster_list):
	for point in X:
		chose_cluster(point, cluster_list)

def plot_cluster_list(cluster_list, plt, filename = None):
	colors = ["r", "b", "y", "g", "k", "m", "c"]
	i = 0
	fig = plt.figure()
	if (cluster_list[0].dimension == 3):
		ax = Axes3D(fig)
	else:
		ax = fig.add_subplot(111)

	purity = 0
	for cluster in cluster_list:
		purity += cluster.plot_cluster_and_return_purity(colors[i], ax)	
		i += 1

	average_purity = purity/len(cluster_list)
	text = "Average purity = " + str(average_purity)[:6]

	if (cluster_list[0].dimension == 3):
		ax.text(1,2,3, text)
	else:
		ax.text(2,-2, text)
			# plt.savefig(pig_name)
	if filename != None:
		plt.savefig(filename)
	plt.show()
	plt.clf()
	plt.close('all')
	

def re_center_cluster_list(cluster_list):
	d_sum = 0
	for cluster in cluster_list:
		old_center = cluster.center
		cluster.re_center()
		d_sum += math.sqrt(distance(old_center, cluster.center))
	return d_sum

def average_purity(cluster_list):
	for cluster in cluster_list:
		cluster.purity()

def myKMeans(X, k):
	if k < 2 or k > 7:
		print("Number of cluster muster from 2 to 7")
		return
	cluster_list = init_cluster_set(X, k)
	print("Number of cluster: ", k)
	# Iterate until cluster reference does not change much
	E = math.pow(2, -23)
	i = 0
	while True:
		divide_data_into_clusters(X, cluster_list)
		
		#PLot and save fig for submission
		pig_name = "pic" + str(i) + ".png"
		i += 1
		plot_cluster_list(cluster_list, plt)
		

		
		d_sum = re_center_cluster_list(cluster_list)
		print("Sum of change = ", d_sum)
		if d_sum <= E:
			break


def main():
	#Initialize Datamatrix and cluster list, data_matrix is nx4, first 3 row are data, last row is label
	#To change the # of feature in datamatrix, adjust line 85 function prepre_data()
	X = prepare_data("diabetes.csv")
	myKMeans(X,4)	

		

if __name__ == "__main__":
	main()