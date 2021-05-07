# import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_data():

	path_train = os.chdir('Dataset/train/')

	categories =[]
	for i in os.listdir(path_train):
		categories.append(i)

	train_data = []
	train_labels = []

	test_data =[]

	test_labels = []

	for category in categories:

		for files in os.listdir(category):
			image = cv2.imread(category+'/'+files)
			gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			gray_image = cv2.resize(gray_image,(32,32))
			image = np.array(gray_image,dtype=np.float32)				
			train_data.append(image)
			cat = category.split(' ')
			train_labels.append(int(cat[1])-1)

	os.chdir('./../../')
	path_test = os.chdir(os.getcwd()+'/Dataset/test/')
	
	for i in os.listdir(path_test):			
		image = cv2.imread(i)
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray_image = cv2.resize(gray_image,(32,32))
		image = np.array(gray_image)								
		image = np.array(image,dtype=np.float32)		
		test_data.append(image)
		cat = i.split(' ')				
		test_labels.append(int(cat[1][0])-1)
		
		
	return (train_data,train_labels),(test_data,test_labels)

def show_image(image,labels):
	plt.title(labels)
	plt.imshow(image,cmap='gray')
	plt.show()

if __name__ == "__main__":
    (train_data,train_labels),(test_data,test_labels) = load_data()
    # print(train_data)	







		
