# Required Packages
from sklearn import datasets		# To Get iris dataset
from sklearn import svm    			# To fit the svm classifier
import numpy as np
import matplotlib.pyplot as plt            # To visuvalizing the data


# import iris data to model Svm classifier
iris_dataset = datasets.load_iris()
#print("Iris data set Description :: ", iris_dataset['DESCR'])

#alize the Speal features and corresponding target classesPython
def visuvalize_sepal_data():
	iris = datasets.load_iris()
	X = iris.data[:, :2]  # we only take the first two features.
	y = iris.target
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
	plt.xlabel('Sepal length')
	plt.ylabel('Sepal width')
	plt.title('Sepal Width & Length')
	plt.show()


visuvalize_sepal_data()

