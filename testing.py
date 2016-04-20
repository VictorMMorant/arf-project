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

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1461164218/checkpoints", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Load data
print("Loading Testing data...")
x, y, vocabulary = preprocessing.load_data('./data/tweets2013_test.txt')
y = np.argmax(y,axis=1)
print(x)
print("Number of Tweets: {:d}".format(x.shape[0]))
print("Vocabulary Size: {:d}".format(len(vocabulary)))

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        print(predictions)
        # Generate batches for one epoch
        #batches = utils.batch_iter(x, FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = sess.run(predictions, {input_x: x, dropout_keep_prob: 1.0})

        #for x_test_batch in batches:

        #   batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
        #   all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy
correct_predictions = float(sum(all_predictions == y))
print("Total number of test examples: {}".format(len(y)))
print("Accuracy: {:g}".format(correct_predictions/float(len(y))))