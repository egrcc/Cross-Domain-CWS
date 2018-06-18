import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import utils
import utils_data
from model import LSTMModel, LSTMConfig, LSTMLMModel, LSTMLMConfig

tf.flags.DEFINE_string("model", "lstm", "The model used to classification.")
tf.flags.DEFINE_string("source", "ctb", "The dataset for source domain.")
tf.flags.DEFINE_string("target", "zx", "The dataset for target domain.")
tf.flags.DEFINE_string("name", "", "The model name.")
tf.flags.DEFINE_boolean("pl", True, "Set to True for using partial label data.")

FLAGS = tf.flags.FLAGS

if FLAGS.model == "lstm":
	config = LSTMConfig()
elif FLAGS.model == "lstmlm":
	config = LSTMLMConfig()

if FLAGS.target == "zx":
	target_valid_X, target_valid_Y, target_valid_bi_X, target_test_X, target_test_Y, target_test_bi_X = utils_data.get_zx_data()

init_embedding = utils_data.get_embedding(FLAGS.target)
# init_bi_embedding = utils_data.get_bi_embedding(FLAGS.target)

with tf.Graph().as_default(), tf.Session() as sess:

	if FLAGS.target == "zx":
		vocab_size = 4704
		bi_vocab_size = 250734

	if FLAGS.model == "lstm":
		m = LSTMModel(config.hidden_size, config.max_grad_norm, config.num_layers, vocab_size, 
			config.embedding_size, config.num_classes, config.learning_rate, config.bi_direction, init_embedding)
	elif FLAGS.model == "lstmlm":
		m = LSTMLMModel(config.hidden_size, config.max_grad_norm, config.num_layers, vocab_size, 
			config.embedding_size, config.num_classes, config.learning_rate, config.bi_direction, init_embedding)
	
	model_path = "model/%s_%s_%s_%s_%s.ckpt" % (FLAGS.model, FLAGS.source, FLAGS.target, str(FLAGS.pl), FLAGS.name)
	saver = tf.train.Saver()

	load_path = saver.restore(sess, model_path)

	test_cost = 0.
	test_acc = 0.
	test_step = 0
	test_true_Y = []
	test_pred_Y = []

	# test on target domain test data
	for step, (X, Y, bi_X, true_Y, fw_Y, bw_Y, W) in enumerate(utils_data.data_iterator(target_test_X,
														target_test_Y, target_test_bi_X, 128, False, vocab_size, bi_vocab_size)):
		if FLAGS.model == "lstm":
			cost, length, pred_Y, accuracy = sess.run([m.loss, m.seq_length, m.pred, m.accuracy],
												feed_dict={m.x: X, m.y: Y, m.weights: W,
														m.dropout_keep_prob: 1., m.pl: False})
		elif FLAGS.model == "lstmlm":
			cost, length, pred_Y, accuracy = sess.run([m.total_loss, m.seq_length, m.pred, m.accuracy],
								feed_dict={m.x: X, m.y: Y, m.weights: W, m.fw_y: fw_Y, m.bw_y: bw_Y,
											m.dropout_keep_prob: 1., m.pl: False})

		test_cost += cost
		test_acc += accuracy
		test_step += 1
		test_true_Y.extend(true_Y)
		test_pred_Y.extend([pred_Y[i][:length[i]].tolist() for i in range(len(pred_Y))])

		if step % 1000 == 0:
			print("Target domain test. Step: %d Cost: %.5f Accuracy: %.5f" % (step, cost, accuracy))

	avg_cost = test_cost / test_step
	avg_acc = test_acc / test_step
	p, r, f1 = utils.evaluate_word_PRF(test_pred_Y, test_true_Y)
	print("Average cost: %.5f Average accuracy: %.5f" % (avg_cost, avg_acc))
	print("Precision: %.5f Recall: %.5f F1: %.5f" % (p, r, f1))

	model_name = "%s_%s_%s" % (FLAGS.model, FLAGS.source, str(FLAGS.pl))
	utils.get_seg_file(target_test_X, test_pred_Y, FLAGS.target, model_name)