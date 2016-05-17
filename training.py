"""
    Training Script
"""

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import preprocessing
import utils
from SentimentCNN import SentimentCNN

# Parameters
# ==================================================

# Flags are command-line arguments to our program
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs")
tf.flags.DEFINE_integer("test_every", 100, "Steps to test the trained model with the testing partition")

# Config Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparatopn
# ==================================================

x, y, vocabulary, trainingPartitionLength = preprocessing.load_corpus("./data/tweets2013_train.txt","./data/tweets2013_test.txt")

x_train, x_test = x[:trainingPartitionLength], x[trainingPartitionLength:]
y_train, y_test = y[:trainingPartitionLength], y[trainingPartitionLength:]

print("Number of Tweets Train: {:d}".format(x_train.shape[0]))
print("Vocabulary Size Train: {:d}".format(len(vocabulary)))
print("Number of Tweets Test: {:d}".format(x_test.shape[0]))

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = SentimentCNN(
            sequence_length=x_train.shape[1],
            #Positive, Neutral, Negative
            num_classes=3,
            vocab_size=len(vocabulary),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        #TODO: Experiment with several optimizers
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, loss, accuracy = sess.run(
                [train_op, global_step, cnn.loss, cnn.accuracy],feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

        def test_step(x_test,y_test):
            
            accuracy = sess.run(cnn.accuracy, {cnn.input_x: x_test, cnn.input_y: y_test, cnn.dropout_keep_prob: 1.0})
                
            # Print accuracy
            print("=======TESTING========")
            print("Total number of test examples: {}".format(len(y_test)))
            print("Accuracy: {:g}".format(accuracy)) 

        #Training Phase

        # Generate batches
        batches = utils.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        # Training loop. For each batch...
        i = 0
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)

            # Test with Testing Partition
            if i % FLAGS.test_every == 0:
                test_step(x_test,y_test)
            i +=1 

        #Testing Phase
        #test_step(x_test,y_test)


