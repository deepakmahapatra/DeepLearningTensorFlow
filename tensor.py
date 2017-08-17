import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

print "Started"


mnist=input_data.read_data_sets(".",one_hot=True)

n_nodes_hidden1= 100
n_nodes_hidden2=100
n_nodes_hidden3=100

n_classes=10

batch_size=100

x=tf.placeholder('float',[None, 784])
y=tf.placeholder('float')

def neural_network_model(data):
	hidden_layer_1= {'weights':tf.Variable(tf.random_normal([784, n_nodes_hidden1])),
	'biases':tf.Variable(tf.random_normal(n_nodes_hidden1))}
	hidden_layer_2= {'weights':tf.Variable(tf.random_normal([n_nodes_hidden1, n_nodes_hidden2])),
	'biases':tf.Variable(tf.random_normal(n_nodes_hidden2))}
	hidden_layer_3= {'weights':tf.Variable(tf.random_normal([n_nodes_hidden2, n_nodes_hidden3])),
	'biases':tf.Variable(tf.random_normal(n_nodes_hidden3))}
	output_layer= {'weights':tf.Variable(tf.random_normal([n_nodes_hidden3, n_classes])),
	'biases':tf.Variable(tf.random_normal(n_nodes_hidden1))}

	layer1=tf.add(tf.matmul(data,hidden_layer_1['weights']+hidden_layer_1['biases']))
	layer1=tf.nn.relu(layer1)
	
	layer2=tf.add(tf.matmul(layer1,hidden_layer_2['weights']+hidden_layer_2['biases']))
	layer2=tf.nn.relu(layer2)
	
	layer3=tf.add(tf.matmul(layer2,hidden_layer_3['weights']+hidden_layer_3['biases']))
	layer3=tf.nn.relu(layer3)

	



