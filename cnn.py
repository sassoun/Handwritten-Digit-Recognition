'''
In order to build a Convolutional Neural Network (CNN), 
MNIST database http://yann.lecun.com/exdb/mnist/
is downloaded to be used in this project to train, verify, and test our model.

After CNN is created, trained, and tested, each weight and bias arrays are saved into different
text files to reuse for feature uses such as to implement into FPGA by using Verilog

Also, cpkt files and checkpoints are created to save and restore all values to test the model with 
different upcoming input images for future uses

The Coder: Sassoun Gostantian
Project: https://github.com/sassoun/Handwritten-Digit-Recognition
'''

from __future__ import print_function
import tensorflow as tf # giving a shortcut fro tensorflow
import numpy as numm # giving a shortcut for numpy 

from tensorflow.examples.tutorials.mnist import input_data # download the MNIST database images from the link we mentioned above
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True) # this command downloads the values from the tensorflow tutorial to use for our project

# these are our parameters
decreasing_function_of_time = 0.001 # we want to make sure the learning iteration is enough to increase the accuracy of learning
total_steps = 200000 # we use 200,000 iterations for better accuracy
min_batch_size_needed = 128 # min batch size needed to make sure all models can be trained with the steps given
specified_step_to_check_for_loss = 10 # every 10 times from count, the loss will be checked and found and recorded

# parameters from network connection
n_input = 784 # 28*28 = 784 pixels for each MNIST images 
digit_classes = 10 # from 0 to 9 total classes
dropout = 0.75 # Dropout, probability to keep units

mnist_input = tf.placeholder(tf.float32, [None, n_input]) # tf Graph input
output_of_the_model = tf.placeholder(tf.float32, [None, digit_classes])
come_up_with_probability = tf.placeholder(tf.float32) #dropout (keep probability)


def Convolutional_module_wrapper(mnist_input, W, b, strides=1): # Create some wrappers to make our lives easier by putting activation function and layers all together in a function
    mnist_input = tf.nn.conv2d(mnist_input, W, strides=[1, strides, strides, 1], padding='SAME')    # Convolutional_module_wrapper wrapper, with bias and relu activation
    mnist_input = tf.nn.bias_add(mnist_input, b)
    return tf.nn.relu(mnist_input)


def Max_Pooling(mnist_input, k=2):
    # Max_Pooling wrapper
    return tf.nn.max_pool(mnist_input, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def conv_net(mnist_input, weights, biases, dropout): # Let's create the layers of our model

    mnist_input = tf.reshape(mnist_input, shape=[-1, 28, 28, 1])    # Reshape input picture

    conv1 = Convolutional_module_wrapper(mnist_input, weights['wc1'], biases['bc1'])    # Our first Convolution Layer
    conv1 = Max_Pooling(conv1, k=2)    # First time Max Pooling (down-sampling)

    conv2 = Convolutional_module_wrapper(conv1, weights['wc2'], biases['bc2'])    # Our second Convolution Layer
    conv2 = Max_Pooling(conv2, k=2)    # Second time Max Pooling (down-sampling)

    # Fully connected layer (we must rearrange the last layer of conv2 to fit into fully connected layer, we used only one fully connected layer)
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)    # Apply Dropout

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])    # Output layer
    return out
	
	
# we created a ndarray and inserted all weights from each layers
weights = { 
    # 5x5 conv, 1 input, 32 different weights of output (first convolution layer) [we have 1*32 = 32 amount of weights]
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 different weights of output (goes to second convolution layer) [we have 32*64 = 2048 amount of weights]
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 weights of output (connects all the layer to each other)
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, only 10 weights of output (class prediction, predicts the results on output layer)
    'out': tf.Variable(tf.random_normal([1024, digit_classes]))
}

# we created a ndarray and inserted all biases from each layers
biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([digit_classes]))
}

pred = conv_net(mnist_input, weights, biases, come_up_with_probability) # Build the model

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = output_of_the_model)) # Create entropy loss function
optimizer = tf.train.AdamOptimizer(learning_rate = decreasing_function_of_time).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(output_of_the_model, 1)) # Calculate the results for the model
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

f1 = open("C:/Users/sassoun77/weights1.txt", "w") # save each weights of arrays in a text file (of course the file location can be changed)
f2 = open("C:/Users/sassoun77/weights2.txt", "w") 
f3 = open("C:/Users/sassoun77/weights3.txt", "w") 
f4 = open("C:/Users/sassoun77/weights4.txt", "w") 
f5 = open("C:/Users/sassoun77/bias1.txt", "w") # save each biases of arrays in a text file
f6 = open("C:/Users/sassoun77/bias2.txt", "w") 
f7 = open("C:/Users/sassoun77/bias3.txt", "w") 
f8 = open("C:/Users/sassoun77/bias4.txt", "w") 

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer()) # Initialize the global variables with running session
    count = 1 
	
    # Keep training until reach max iterations
    while count * min_batch_size_needed < total_steps: # continously train until "total_steps"
        batch_x, batch_y = mnist.train.next_batch(min_batch_size_needed)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={mnist_input: batch_x, output_of_the_model: batch_y,
                                       come_up_with_probability: dropout})
        if count % specified_step_to_check_for_loss == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={mnist_input: batch_x,
                                                              output_of_the_model: batch_y,
                                                              come_up_with_probability: 1.})
            print("Number of Steps: " + str(count*min_batch_size_needed) + ", Cross Entropy Loss: " + \
                  "{:.6f}".format(loss) + ", Total Accuracy: " + \
                  "{:.5f}".format(acc))

        count += 1
		
    weights = sess.run(weights)   # preparing the session to be used for weights 
    numm.set_printoptions(threshold=numm.inf) # importing this numpy helps us to get our full arrays of weight and bias values
	
    weights1 = weights['wc2']
    numm.array(list(weights1[:, 1]), dtype=numm.float) # remove dtype=float32 from end of the wc2 array
    weights2 = weights['wd1']
    numm.array(list(weights2[:, 1]), dtype=numm.float) # remove dtype=float32 from end of the wd1 array
    weights3 = weights['out']
    numm.array(list(weights3[:, 1]), dtype=numm.float) # remove dtype=float32 from end of the out array

    f1.write(str(weights['wc1']) + "\n" ) # print wc1 array of weights in a text file 
    f2.write(str(weights1) + "\n" ) # print wc2 array of weights in a text file 
    f3.write(str(weights2) + "\n" ) # print wd1 array of weights in a text file 
    f4.write(str(weights3) + "\n" ) # print out array of weights in a text file 

    biases = sess.run(biases)  # preparing the session to be used for bias
    f5.write(str(biases['bc1']) + "\n" ) # print bc1 array of weights in a text file 
    f6.write(str(biases['bc1']) + "\n" ) # print bc2 array of weights in a text file 
    f7.write(str(biases['bd1']) + "\n" ) # print bd1 array of weights in a text file 
    f8.write(str(biases['out']) + "\n" ) # print out array of weights in a text file 
	
	
    print("It has finished all the iterations sucessfully :)")
	
    # Calculate the accuracy!!
    print("Let's Test Our Accuracy:", \
        sess.run(accuracy, feed_dict={mnist_input: mnist.test.images[:256],
                                      output_of_the_model: mnist.test.labels[:256],
                                      come_up_with_probability: 1.}))
    
						  
f1.close() # close the text file
f2.close() # close the text file
f3.close() # close the text file
f4.close() # close the text file