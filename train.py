import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import utils
import utils_data
import datetime
from model import LSTMModel, LSTMConfig, LSTMLMModel, LSTMLMConfig


tf.flags.DEFINE_string("model", "lstm", "The model used to classification.")
tf.flags.DEFINE_string("source", "ctb", "The dataset for source domain.")
tf.flags.DEFINE_string("target", "zx", "The dataset for target domain.")
tf.flags.DEFINE_string("name", "", "The model name.")
tf.flags.DEFINE_boolean("pl", True, "Set to True for using partial label data.")
tf.flags.DEFINE_float("memory", 0.5, "Allowing GPU memory growth")

FLAGS = tf.flags.FLAGS

if FLAGS.model == "lstm":
	config = LSTMConfig()
elif FLAGS.model == "lstmlm":
	config = LSTMLMConfig()

if FLAGS.source == "ctb":
	source_train_X, source_train_Y, source_train_bi_X = utils_data.get_ctb_data()

if FLAGS.target == "zx":
	target_valid_X, target_valid_Y, target_valid_bi_X, target_test_X, target_test_Y, target_test_bi_X = utils_data.get_zx_data()
	if FLAGS.model == "lstmlm":
		target_ul_train_X = utils_data.get_zx_ul_data()
	if FLAGS.pl == True:
		target_pl_train_X, target_pl_train_Y, target_pl_train_bi_X, target_pl_train_weight = utils_data.get_zx_pl_data()

init_embedding = utils_data.get_embedding(FLAGS.target)
# init_bi_embedding = utils_data.get_bi_embedding(FLAGS.target)

tfConfig = tf.ConfigProto()
tfConfig.gpu_options.per_process_gpu_memory_fraction = FLAGS.memory

with tf.Graph().as_default(), tf.Session(config=tfConfig) as sess:

	if FLAGS.target == "zx":
		vocab_size = 4704
		bi_vocab_size = 250734

	if FLAGS.model == "lstm":
		m = LSTMModel(config.hidden_size, config.max_grad_norm, config.num_layers, vocab_size, 
			config.embedding_size, config.num_classes, config.learning_rate, config.bi_direction, init_embedding)
	elif FLAGS.model == "lstmlm":
		m = LSTMLMModel(config.hidden_size, config.max_grad_norm, config.num_layers, vocab_size, 
			config.embedding_size, config.num_classes, config.learning_rate, config.bi_direction, init_embedding)
	
	sess.run(tf.global_variables_initializer())

	best_valid_f1 = 0.

	model_path = "model/%s_%s_%s_%s_%s.ckpt" % (FLAGS.model, FLAGS.source, FLAGS.target, str(FLAGS.pl), FLAGS.name)
	saver = tf.train.Saver()

	# saver.restore(sess, model_path)

	for epoch in range(config.max_epochs):

		if epoch < 10:
			m.assign_lr(sess, config.learning_rate)
		if 10 <= epoch < 15:
			m.assign_lr(sess, config.learning_rate * config.lr_decay)
		if 15 <= epoch < 20:
			m.assign_lr(sess, config.learning_rate * (config.lr_decay)**2)
		if 20 <= epoch < 25:
			m.assign_lr(sess, config.learning_rate * (config.lr_decay)**3)
		if 25 <= epoch:
			m.assign_lr(sess, config.learning_rate * (config.lr_decay)**4)

		total_cost = 0.
		total_acc = 0.
		total_step = 0
		total_true_Y = []
		total_pred_Y = []

		# training on source domain labeled data
		for step, (X, Y, bi_X, true_Y, fw_Y, bw_Y, W) in enumerate(utils_data.data_iterator(source_train_X,
												source_train_Y, source_train_bi_X, config.batch_size, True, vocab_size, bi_vocab_size)):
			if FLAGS.model == "lstm":
				cost, length, pred_Y, accuracy, _ = sess.run([m.loss, m.seq_length, m.pred, m.accuracy, m.train_op],
															feed_dict={m.x: X, m.y: Y, m.weights: W,
																m.dropout_keep_prob: config.dropout_keep_prob, m.pl: False})
			elif FLAGS.model == "lstmlm":
				cost, length, pred_Y, accuracy, _ = sess.run([m.total_loss, m.seq_length, m.pred, m.accuracy, m.total_train_op],
														feed_dict={m.x: X, m.y: Y, m.weights: W, m.fw_y: fw_Y, m.bw_y: bw_Y,
																m.dropout_keep_prob: config.dropout_keep_prob, m.pl: False})

			total_cost += cost
			total_acc += accuracy
			total_step += 1
			total_true_Y.extend(true_Y)
			total_pred_Y.extend([pred_Y[i][:length[i]].tolist() for i in range(len(pred_Y))])
			

			if step % 100 == 0:
				print("Source domain training. Step: %d Cost: %.5f Accuracy: %.5f" % (step, cost, accuracy))

		avg_cost = total_cost / total_step
		avg_acc = total_acc / total_step
		p, r, f1 = utils.evaluate_word_PRF(total_pred_Y, total_true_Y)
		print("Training Epoch: %d Average cost: %.5f Average accuracy: %.5f" % (epoch, avg_cost, avg_acc))
		print("Training Epoch: %d Precision: %.5f Recall: %.5f F1: %.5f" % (epoch, p, r, f1))

		# training on target domain unlabeled data
		if FLAGS.model == "lstmlm":

			for step, (X, fw_Y, bw_Y) in enumerate(utils_data.ul_data_iterator(target_ul_train_X,
																config.batch_size, True, vocab_size)):
				lm_cost, _ = sess.run([m.lm_loss, m.lm_train_op],
								feed_dict={m.x: X, m.fw_y: fw_Y, m.bw_y: bw_Y,
											m.dropout_keep_prob: config.dropout_keep_prob})

				if step % 100 == 0:
					print("Target domain unlabeled training. Step: %d Cost: %.5f" % (step, lm_cost))

		# training on target domain partial labeled data
		if FLAGS.pl == True and epoch % 5 == 0:

			for step, (X, Y, bi_X, fw_Y, bw_Y, W) in enumerate(utils_data.pl_data_iterator(target_pl_train_X,
					target_pl_train_Y, target_pl_train_bi_X, target_pl_train_weight, config.batch_size, True, vocab_size, bi_vocab_size)):
				if FLAGS.model == "lstm":
					pl_cost, pl_accuracy, _ = sess.run([m.loss, m.accuracy, m.train_op],
												feed_dict={m.x: X, m.y: Y, m.weights: W,
													m.dropout_keep_prob: config.dropout_keep_prob, m.pl: True})
				elif FLAGS.model == "lstmlm":
					# pl_cost, pl_accuracy, _ = sess.run([m.total_loss, m.accuracy, m.total_train_op],
					pl_cost, pl_accuracy, _ = sess.run([m.loss, m.accuracy, m.train_op],
												feed_dict={m.x: X, m.y: Y, m.weights: W, m.fw_y: fw_Y, m.bw_y: bw_Y,
														m.dropout_keep_prob: config.dropout_keep_prob, m.pl: True})

				if step % 100 == 0:
					print("Target domain partial labeled training. Step: %d Cost: %.5f Accuracy: %.5f" % (step, pl_cost, pl_accuracy))

		valid_cost = 0.
		valid_acc = 0.
		valid_step = 0
		valid_true_Y = []
		valid_pred_Y = []

		# validation on target domain valid data
		for step, (X, Y, bi_X, true_Y, fw_Y, bw_Y, W) in enumerate(utils_data.data_iterator(target_valid_X,
															target_valid_Y, target_valid_bi_X, 64, False, vocab_size, bi_vocab_size)):
			if FLAGS.model == "lstm":
				cost, length, pred_Y, accuracy = sess.run([m.loss, m.seq_length, m.pred, m.accuracy],
													feed_dict={m.x: X, m.y: Y, m.weights: W,
														m.dropout_keep_prob: 1., m.pl: False})
			elif FLAGS.model == "lstmlm":
				cost, length, pred_Y, accuracy = sess.run([m.total_loss, m.seq_length, m.pred, m.accuracy],
									feed_dict={m.x: X, m.y: Y, m.weights: W, m.fw_y: fw_Y, m.bw_y: bw_Y,
												m.dropout_keep_prob: 1., m.pl: False})

			valid_cost += cost
			valid_acc += accuracy
			valid_step += 1
			valid_true_Y.extend(true_Y)
			valid_pred_Y.extend([pred_Y[i][:length[i]].tolist() for i in range(len(pred_Y))])
		
			if step % 100 == 0:
				print("Target domain validation. Step: %d Cost: %.5f Accuracy: %.5f" % (step, cost, accuracy))

		avg_cost = valid_cost / valid_step
		avg_acc = valid_acc / valid_step
		p, r, f1 = utils.evaluate_word_PRF(valid_pred_Y, valid_true_Y)
		print("Validation Epoch: %d Average cost: %.5f Average accuracy: %.5f" % (epoch, avg_cost, avg_acc))
		print("Validation Epoch: %d Precision: %.5f Recall: %.5f F1: %.5f" % (epoch, p, r, f1))
		print("Validation Epoch: %d Best F1: %.5f" % (epoch, best_valid_f1))

		if f1 > best_valid_f1:
			best_valid_f1 = f1
			save_path = saver.save(sess, model_path)
			print("save model.")

		test_cost = 0.
		test_acc = 0.
		test_step = 0
		test_true_Y = []
		test_pred_Y = []

		# test on target domain test data
		for step, (X, Y, bi_X, true_Y, fw_Y, bw_Y, W) in enumerate(utils_data.data_iterator(target_test_X,
															target_test_Y, target_test_bi_X, 64, False, vocab_size, bi_vocab_size)):
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

			if step % 100 == 0:
				print("Target domain test. Step: %d Cost: %.5f Accuracy: %.5f" % (step, cost, accuracy))

		avg_cost = test_cost / test_step
		avg_acc = test_acc / test_step
		p, r, f1 = utils.evaluate_word_PRF(test_pred_Y, test_true_Y)
		print("Test Epoch: %d Average cost: %.5f Average accuracy: %.5f" % (epoch, avg_cost, avg_acc))
		print("Test Epoch: %d Precision: %.5f Recall: %.5f F1: %.5f" % (epoch, p, r, f1))

		print config
		print FLAGS.target
		model_name = "%s_%s_%s" % (FLAGS.model, FLAGS.source, str(FLAGS.pl))
		print model_name
		print datetime.datetime.now()
		print FLAGS.name
