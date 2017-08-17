import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

print "Started"


mnist=input_data.read_data_sets(".",one_hot=True)

n_nodes_hidden1= 500
n_nodes_hidden2=1000
n_nodes_hidden3=500

n_classes=10

batch_size=100

x=tf.placeholder('float',[None, 784])
y=tf.placeholder('float')

def neural_network_model(data):
	hidden_layer_1= {'weights':tf.Variable(tf.random_normal([784, n_nodes_hidden1])),'biases':tf.Variable(tf.random_normal([n_nodes_hidden1]))}
	hidden_layer_2= {'weights':tf.Variable(tf.random_normal([n_nodes_hidden1, n_nodes_hidden2])),'biases':tf.Variable(tf.random_normal([n_nodes_hidden2]))}
	hidden_layer_3= {'weights':tf.Variable(tf.random_normal([n_nodes_hidden2, n_nodes_hidden3])),'biases':tf.Variable(tf.random_normal([n_nodes_hidden3]))}
	output_layer= {'weights':tf.Variable(tf.random_normal([n_nodes_hidden3, n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}

	layer1=tf.add(tf.matmul(data,hidden_layer_1['weights']),hidden_layer_1['biases'])
	layer1=tf.nn.relu(layer1)
	
	layer2=tf.add(tf.matmul(layer1,hidden_layer_2['weights']),hidden_layer_2['biases'])
	layer2=tf.nn.relu(layer2)
	
	layer3=tf.add(tf.matmul(layer2,hidden_layer_3['weights']),hidden_layer_3['biases'])
	layer3=tf.nn.relu(layer3)


	output=tf.matmul(layer3, output_layer['weights']) + output_layer['biases']

	return output


def train_neural_network(x):
	prediction=neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
	optimizer=tf.train.AdamOptimizer().minimize(cost)
	#number of cycles of feedforward and back propagation
	num_epochs = 20
	with tf.Session() as session:
		session.run(tf.global_variables_initializer())
		for epoch in range(num_epochs):
			epoch_loss=0
			for _ in range(num_epochs):
				train_1, test_1 =mnist.train.next_batch(batch_size)
				_, c= session.run([optimizer,cost],feed_dict={x:train_1,y:test_1})
				epoch_loss += c
			print 'Epoch is ', epoch, "completed out of ", num_epochs,"loss is ", epoch_loss

		correct=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))

		accuracy= tf.reduce_mean(tf.cast(correct,'float'))

		print "Accuracy: ",accuracy.eval({x:mnist.test.images, y:mnist.test.labels})


train_neural_network(x)
	



