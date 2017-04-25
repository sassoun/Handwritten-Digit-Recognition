'''
In order to build a Convolutional Neural Network (CNN), 
MNIST database http://yann.lecun.com/exdb/mnist/
is downloaded to be used in this project to train, verify, and test our model.

After CNN is created, trained, and tested, each weight and bias arrays are saved into different
text files to reuse for feature uses such as to implement into FPGA by using Verilog

Also, cpkt files and checkpoints are created to save and restore all values to test the model with 
different upcoming input images for future uses

The Coder: Sassoun Gostantian
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function
import tensorflow as tf # giving a shortcut
import numpy as numm # giving a shortcut

from tensorflow.examples.tutorials.mnist import input_data # download the MNIST database images from the link we mentioned above
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True) # continue downloading

# let's define the parameters
learning_rate = 0.001 # we want to make sure the learning iteration is slow enough to increase the accuracy of learning
training_iters = 200 # we use 200,000 iterations for better accuracy
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28 pixels)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units
i = 0
# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

weights = { # we created a ndarray and inserted all weights from each layers
    # 5x5 conv, 1 input, 32 different weights of output (first convolution layer) [we have 1*32 = 32 amount of weights]
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 different weights of output (goes to second convolution layer) [we have 32*64 = 2048 amount of weights]
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 weights of output (connects all the layer to each other)
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, only 10 weights of output (class prediction, predicts the results on output layer)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

# Store layers weight & bias
biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
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
    step = 1 
	
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))

        step += 1
		
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
	
	
    print("It has finished all the iterations :)")
	
    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256],
                                      keep_prob: 1.}))
    
						  
f1.close() # close the text file
f2.close() # close the text file
f3.close() # close the text file
f4.close() # close the text file