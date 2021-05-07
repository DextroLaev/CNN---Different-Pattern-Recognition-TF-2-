import tensorflow as tf
import dataset
import matplotlib.pyplot as plt
import sys

class Neural_nets:
	
	def __init__(self,train_data,train_label,test_data,test_label):
		self.train_data = train_data
		self.test_data = test_data
		self.train_label = tf.cast(train_label,tf.int32)
		self.test_label = tf.cast(test_label,tf.int32)
		self.lr = 0.0002

	def model(self):
		self.conv1 = tf.keras.layers.Conv2D(filters=16,kernel_size=5,strides=1,padding='same',activation='relu')
		self.maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=None,padding='valid')
		self.conv2 = tf.keras.layers.Conv2D(filters=32,kernel_size=5,strides=1,padding='same',activation='relu')
		self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=None,padding='valid')
		self.conv3 = tf.keras.layers.Conv2D(filters=120,kernel_size=5,strides=1,padding='same',activation='relu')	
		self.flatten_data = tf.keras.layers.Flatten()
		self.hd1 = tf.keras.layers.Dense(240,activation='relu')		
		self.hd2 = tf.keras.layers.Dense(120,activation='relu')	
		self.hd3 = tf.keras.layers.Dense(4,activation='linear')	

		self.layers = [self.hd3,self.hd2,self.hd1,self.flatten_data,self.conv3,
		               self.maxpool2,self.conv2,self.maxpool1,self.conv1]			
             
		               
	def feed_forward(self,inputs):
		# self.model()
		model_input = tf.reshape(inputs,shape=[-1,inputs[0].shape[0],inputs[0].shape[0],1])
		output = model_input
		for i in range(1,len(self.layers)+1):
			output = self.layers[-i](output)
		return output

	def loss_fn(self,pred,label):
		return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label,logits=pred))

	def accuracy(self,pred,label):
		return  tf.reduce_mean(tf.cast(tf.equal(tf.argmax(label,1),tf.argmax(pred,1)),dtype=tf.float32))
	
	def update_parameter(self,layers,gradients):
		for i in range(len(layers)):
			for j in range(2):
				if i == 3 or i== 5 or i==7:
					pass
				else:
					layers[i].trainable_weights[j].assign_sub(self.lr*gradients[i][j])

	def train(self,epochs=500):
		loss = []
		self.model()		
		for i in range(epochs):
			with tf.GradientTape() as tape:
				predicted = self.feed_forward(self.train_data)
				loss_val = self.loss_fn(predicted,self.train_label)
			gradients = tape.gradient(loss_val,[layer.trainable_weights for layer in self.layers])
			self.update_parameter(self.layers,gradients)			
			loss.append(loss_val)
			if epochs%10 == 0:
				print('\r epochs = {}, loss = {}'.format(i+1,loss_val),end='')
				sys.stdout.flush()		
		plt.plot(loss)
		plt.xlabel('iterations')
		plt.ylabel('loss')
		plt.show()			

	def test(self):
		p = self.feed_forward(self.test_data)
		acc = self.accuracy(p,self.test_label)
		tf.print('testing acc : ',acc)

if __name__=='__main__':
	(train_data,train_label),(test_data,test_label) = dataset.load_data()
	train_label = tf.squeeze(tf.one_hot(train_label,depth=4))
	test_label = tf.squeeze(tf.one_hot(test_label,depth=4))
	n = Neural_nets(train_data,train_label,test_data,test_label)
	n.train(3000)
	n.test()