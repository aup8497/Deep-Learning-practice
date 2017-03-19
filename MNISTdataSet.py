import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# these are the number of nodes at each layer
# they need not necessarily be same
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

# this is the number of output nodes , i.e number of classes/categories to which we divide
n_classes = 10
# number of images to be processed at once
batch_size = 100

# height x width
x=tf.placeholder('float', [ None , 784])
y=tf.placeholder('float')

def neural_network_model(data):

    # here we are creating hidden layers with the following specifications
    #           hidden_1_layer takes in 784 input data and puts out the 500 nodes output ( the number of nodes in the hidden layer 1 )
    #           hidden_2_layer takes in 500 nodes input data ( the number of nodes in the hidden layer 1 ) and puts out the 500 nodes output ( the number of nodes in the hidden layer 2 )
    #           hidden_3_layer takes in 500 nodes input data ( the number of nodes in the hidden layer 2 ) and puts out the 500 nodes output ( the number of nodes in the hidden layer 3 )

    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784 , n_nodes_hl1 ])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1 , n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2 , n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3 , n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}


    # OUR MODEL
    # (input data * weights) + biases

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1=tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']),hidden_2_layer['biases'] )
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']) , hidden_3_layer['biases'] )
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases'] 

    return output

def train_neural_network(x):
	prediction = neural_network_model(x)
	# this calculates the difference between the prediction that we got to the num label that we have
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
	# now we ave to minimize the cost that we have got here
	# AdamOptimizer - this is synonymous with stocastic gradient descent ,itergrad and so on
	# using adamOptimizer we are minimizing the cost
	optimizer = tf.train.AdamOptimizer().minimize(cost)


	# 1 epoch is 1 cycle i.e one ( feedforward + backprop )
	hm_epochs = 10

    #we are creating a session
	with tf.Session() as sess:
		sess.run( tf.initialize_all_variables() )


	# training starts here
		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range( int(mnist.train.num_examples/batch_size)):
				epoch_x , epoch_y = mnist.train.next_batch(batch_size)
				_ , c=sess.run( [optimizer,cost] ,feed_dict = { x: epoch_x , y: epoch_y } )
				epoch_loss += c

	        # this is basically the number of iterations i.e epochs ,
	        # this line gets printed after each iteration in the testing/training
			print('Epoch', epoch , 'completed out of ' , hm_epochs , 'loss:', epoch_loss )
	#training ends here

	# NOW WE HAVE OPTIMIZED THE WEIGHTS
	# TO KNOW THE ACCURACY WE RUN THESE STATEMENTS
	    # tf.argmax returns the index of the maximum value of x and y
		correct = tf.equal( tf.argmax(prediction,1) , tf.argmax(y,1) )

		accuracy = tf.reduce_mean(tf.cast( correct,'float'))
		print('Accuracy:' , accuracy.eval({x:mnist.test.images , y:mnist.test.labels}))

train_neural_network(x)







