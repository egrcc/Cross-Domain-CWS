import tensorflow as tf


class LSTMConfig(object):

	def __init__(self):
		self.vocab_size = 20000
		self.embedding_size = 100
		self.batch_size = 20
		self.hidden_size = 300
		self.max_grad_norm = 5
		self.num_layers = 2
		self.num_classes = 4
		self.learning_rate = 1e-3
		self.max_epochs = 30
		self.lr_decay = 0.5
		self.dropout_keep_prob = 0.8
		self.bi_direction = True

	def __str__(self):
		return "vocab_size: %s batch_size: %s embedding_size: %s num_layers: %s hidden_size: %s learning_rate: %s dropout_keep_prob: %s" % \
				(str(self.vocab_size), str(self.batch_size), str(self.embedding_size), str(self.num_layers), \
				str(self.hidden_size), str(self.learning_rate), str(self.dropout_keep_prob))


class LSTMModel(object):

	def __init__(self, hidden_size, max_grad_norm, num_layers, vocab_size, 
				embedding_size, num_classes, learning_rate, bi_direction, init_embedding):

		self.x = tf.placeholder(dtype=tf.int32, shape=[None, None, 3], name='input_x')
		self.y = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_y')
		self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_keep_prob')
		self.weights = tf.placeholder(dtype=tf.float32, shape=[None, None], name='weights')
		self.pl = tf.placeholder(dtype=tf.bool, name="pl")

		self.pl_w1 = tf.multiply(self.weights, tf.cast(tf.less(self.weights, tf.ones_like(self.weights) * 0.8), tf.float32))
		self.pl_w2 = tf.cast(tf.greater(self.weights, tf.ones_like(self.weights) * 0.8), tf.float32)

		self.seq_length = tf.reduce_sum(tf.cast(tf.less(self.x[:, :, 1], tf.ones_like(self.x[:, :, 1]) * (vocab_size-1)), tf.int32), 1)
		self.batch_size = tf.shape(self.x)[0]

		with tf.variable_scope('embedding'):
			self.embedding = tf.Variable(init_embedding, dtype=tf.float32, name='embedding', trainable=True)
			x = tf.nn.embedding_lookup(self.embedding, self.x)
			x = tf.reshape(x, [self.batch_size, -1, 3*embedding_size])
			x = tf.nn.dropout(x, self.dropout_keep_prob)

		with tf.variable_scope('lstm'):

			def lstm_cell():
				return tf.contrib.rnn.BasicLSTMCell(hidden_size)

			def attn_cell():
				return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=self.dropout_keep_prob)

			self.fw_multi_cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(num_layers)])
			self.bw_multi_cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(num_layers)])

			if bi_direction == False:
				self.outputs, _ = tf.nn.dynamic_rnn(
					cell=self.fw_multi_cell,
					inputs=x,
					sequence_length=self.seq_length,
					dtype=tf.float32
				)

			else:
				(fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
					cell_fw=self.fw_multi_cell,
					cell_bw=self.bw_multi_cell,
					inputs=x,
					sequence_length=self.seq_length,
					dtype=tf.float32
				)

				self.outputs = tf.concat((fw_outputs, bw_outputs), 2)


		with tf.variable_scope('loss'):
			if bi_direction == False:
				softmax_w = tf.get_variable("softmax_w", [hidden_size, num_classes])
			else:
				softmax_w = tf.get_variable("softmax_w", [hidden_size * 2, num_classes])
			softmax_b = tf.get_variable("softmax_b", [num_classes])

			self.logits = tf.einsum("aij,jk->aik", self.outputs, softmax_w) + softmax_b
			self.pred = tf.cast(tf.argmax(self.logits, axis=2), tf.int32)
			correct_pred = tf.cast(tf.equal(self.pred, self.y), tf.float32)
			# self.accuracy = tf.reduce_sum(tf.multiply(self.weights, correct_pred)) / tf.reduce_sum(self.weights)
			accuracy_full = tf.reduce_sum(tf.multiply(self.weights, correct_pred)) / (tf.reduce_sum(self.weights) + 1e-12)
			accuracy_pl = tf.reduce_sum(tf.multiply(self.pl_w1, correct_pred)) / (tf.reduce_sum(self.pl_w1) + 1e-12)
			self.accuracy = tf.cond(self.pl, lambda: accuracy_pl, lambda: accuracy_full)

			# self.loss = tf.contrib.seq2seq.sequence_loss(
			# 	self.logits,
			# 	self.y,
			# 	self.weights,
			# 	average_across_timesteps=True,
			# 	average_across_batch=True
			# )

			loss_full = tf.contrib.seq2seq.sequence_loss(
				self.logits,
				self.y,
				self.weights,
				average_across_timesteps=True,
				average_across_batch=True
			)

			loss_pl1 = tf.contrib.seq2seq.sequence_loss(
				self.logits,
				self.y,
				self.pl_w1,
				average_across_timesteps=True,
				average_across_batch=True
			)

			loss_pl2 = tf.contrib.seq2seq.sequence_loss(
				self.logits,
				self.y,
				tf.cast(tf.ones_like(self.y), tf.float32),
				average_across_timesteps=False,
				average_across_batch=False
			)
			loss_pl2 = tf.reduce_sum(tf.multiply(self.pl_w2, -tf.log(1 - tf.exp(-loss_pl2) + 1e-12))) / (tf.reduce_sum(self.pl_w2) + 1e-12)

			self.loss = tf.cond(self.pl, lambda: loss_pl1 + loss_pl2, lambda: loss_full)
			

		with tf.variable_scope('train_op'):
			tvars = tf.trainable_variables()
			grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), max_grad_norm)

			self.lr = tf.Variable(learning_rate, trainable=False)
			self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
			self.lr_update = tf.assign(self.lr, self.new_lr)

			# self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr)
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
			self.train_op = self.optimizer.apply_gradients(zip(grads, tvars))

	def assign_lr(self, session, lr_value):
		session.run(self.lr_update, feed_dict={self.new_lr: lr_value})


class LSTMLMConfig(object):

	def __init__(self):
		self.vocab_size = 20000
		self.embedding_size = 100
		self.batch_size = 20
		self.hidden_size = 300
		self.max_grad_norm = 5
		self.num_layers = 2
		self.num_classes = 4
		self.learning_rate = 1e-3
		self.max_epochs = 30
		self.lr_decay = 0.5
		self.dropout_keep_prob = 0.8
		self.bi_direction = True

	def __str__(self):
		return "vocab_size: %s batch_size: %s embedding_size: %s num_layers: %s hidden_size: %s learning_rate: %s dropout_keep_prob: %s" % \
				(str(self.vocab_size), str(self.batch_size), str(self.embedding_size), str(self.num_layers), \
				str(self.hidden_size), str(self.learning_rate), str(self.dropout_keep_prob))


class LSTMLMModel(object):

	def __init__(self, hidden_size, max_grad_norm, num_layers, vocab_size, 
				embedding_size, num_classes, learning_rate, bi_direction, init_embedding):

		self.x = tf.placeholder(dtype=tf.int32, shape=[None, None, 3], name='input_x')
		self.y = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_y')
		self.fw_y = tf.placeholder(dtype=tf.int32, shape=[None, None], name='forward_y')
		self.bw_y = tf.placeholder(dtype=tf.int32, shape=[None, None], name='backward_y')
		self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_keep_prob')
		self.weights = tf.placeholder(dtype=tf.float32, shape=[None, None], name='weights')
		self.pl = tf.placeholder(dtype=tf.bool, name="pl")

		self.pl_w1 = tf.multiply(self.weights, tf.cast(tf.less(self.weights, tf.ones_like(self.weights) * 0.8), tf.float32))
		self.pl_w2 = tf.cast(tf.greater(self.weights, tf.ones_like(self.weights) * 0.8), tf.float32)

		self.seq_length = tf.reduce_sum(tf.cast(tf.less(self.x[:, :, 1], tf.ones_like(self.x[:, :, 1]) * (vocab_size-1)), tf.int32), 1)
		self.lm_weights = tf.cast(tf.less(self.x[:, :, 1], tf.ones_like(self.x[:, :, 1]) * (vocab_size-1)), tf.float32)
		self.batch_size = tf.shape(self.x)[0]

		with tf.variable_scope('embedding'):
			self.embedding = tf.Variable(init_embedding, dtype=tf.float32, name='embedding', trainable=True)
			x = tf.nn.embedding_lookup(self.embedding, self.x)
			x = tf.reshape(x, [self.batch_size, -1, 3*embedding_size])
			x = tf.nn.dropout(x, self.dropout_keep_prob)

		def lstm_cell():
			return tf.contrib.rnn.BasicLSTMCell(hidden_size)

		def attn_cell():
			return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=self.dropout_keep_prob)

		with tf.variable_scope('lm-lstm'):

			self.lm_fw_multi_cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(num_layers)])
			self.lm_bw_multi_cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(num_layers)])

			(self.lm_fw_outputs, self.lm_bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
				cell_fw=self.lm_fw_multi_cell,
				cell_bw=self.lm_bw_multi_cell,
				inputs=x[:, :, embedding_size:2*embedding_size],
				sequence_length=self.seq_length,
				dtype=tf.float32
			)

		with tf.variable_scope('bi-lstm'):

			self.fw_multi_cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(num_layers)])
			self.bw_multi_cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(num_layers)])

			if bi_direction == False:
				self.bi_outputs, _ = tf.nn.dynamic_rnn(
					cell=self.fw_multi_cell,
					inputs=x,
					sequence_length=self.seq_length,
					dtype=tf.float32
				)

			else:
				(fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
					cell_fw=self.fw_multi_cell,
					cell_bw=self.bw_multi_cell,
					inputs=x,
					sequence_length=self.seq_length,
					dtype=tf.float32
				)

				self.bi_outputs = tf.concat((fw_outputs, bw_outputs), 2)


		with tf.variable_scope('loss'):

			fw_softmax_w = tf.get_variable("fw_softmax_w", [hidden_size, vocab_size])
			fw_softmax_b = tf.get_variable("fw_softmax_b", [vocab_size])
			bw_softmax_w = tf.get_variable("bw_softmax_w", [hidden_size, vocab_size])
			bw_softmax_b = tf.get_variable("bw_softmax_b", [vocab_size])

			self.fw_logits = tf.einsum("aij,jk->aik", self.lm_fw_outputs, fw_softmax_w) + fw_softmax_b
			self.bw_logits = tf.einsum("aij,jk->aik", self.lm_bw_outputs, bw_softmax_w) + bw_softmax_b

			W1_z = tf.get_variable("W1_z", [hidden_size, 2*hidden_size])
			W2_z = tf.get_variable("W2_z", [hidden_size, 2*hidden_size])
			U_z = tf.get_variable("U_z", [2*hidden_size, 2*hidden_size])
			b_z = tf.get_variable("b_z", [2*hidden_size])

			W1_r = tf.get_variable("W1_r", [hidden_size, 2*hidden_size])
			W2_r = tf.get_variable("W2_r", [hidden_size, 2*hidden_size])
			U_r = tf.get_variable("U_r", [2*hidden_size, 2*hidden_size])
			b_r = tf.get_variable("b_r", [2*hidden_size])

			W1_h = tf.get_variable("W1_h", [hidden_size, 2*hidden_size])
			W2_h = tf.get_variable("W2_h", [hidden_size, 2*hidden_size])
			U_h = tf.get_variable("U_h", [2*hidden_size, 2*hidden_size])
			b_h = tf.get_variable("b_h", [2*hidden_size])

			z = tf.sigmoid(tf.einsum("aij,jk->aik", self.lm_fw_outputs, W1_z) \
						+ tf.einsum("aij,jk->aik", self.lm_bw_outputs, W2_z) \
						+ tf.einsum("aij,jk->aik", self.bi_outputs, U_z) \
						+ b_z)

			r = tf.sigmoid(tf.einsum("aij,jk->aik", self.lm_fw_outputs, W1_r) \
						+ tf.einsum("aij,jk->aik", self.lm_bw_outputs, W2_r) \
						+ tf.einsum("aij,jk->aik", self.bi_outputs, U_r) \
						+ b_r)

			h_hat = tf.sigmoid(tf.einsum("aij,jk->aik", self.lm_fw_outputs, W1_h) \
							+ tf.einsum("aij,jk->aik", self.lm_bw_outputs, W2_h) \
							+ tf.einsum("aij,jk->aik", tf.multiply(r, self.bi_outputs), U_h) \
							+ b_h)

			self.outputs = tf.multiply((1 - z), self.bi_outputs) + tf.multiply(z, h_hat)
		
			softmax_w = tf.get_variable("softmax_w", [2*hidden_size, num_classes])
			softmax_b = tf.get_variable("softmax_b", [num_classes])

			self.logits = tf.einsum("aij,jk->aik", self.outputs, softmax_w) + softmax_b
			
			self.pred = tf.cast(tf.argmax(self.logits, axis=2), tf.int32)
			correct_pred = tf.cast(tf.equal(self.pred, self.y), tf.float32)
			# self.accuracy = tf.reduce_sum(tf.multiply(self.weights, correct_pred)) / tf.reduce_sum(self.weights)
			accuracy_full = tf.reduce_sum(tf.multiply(self.weights, correct_pred)) / (tf.reduce_sum(self.weights) + 1e-12)
			accuracy_pl = tf.reduce_sum(tf.multiply(self.pl_w1, correct_pred)) / (tf.reduce_sum(self.pl_w1) + 1e-12)
			self.accuracy = tf.cond(self.pl, lambda: accuracy_pl, lambda: accuracy_full)

			self.fw_loss = tf.contrib.seq2seq.sequence_loss(
				self.fw_logits,
				self.fw_y,
				self.lm_weights,
				average_across_timesteps=True,
				average_across_batch=True
			)

			self.bw_loss = tf.contrib.seq2seq.sequence_loss(
				self.bw_logits,
				self.bw_y,
				self.lm_weights,
				average_across_timesteps=True,
				average_across_batch=True
			)

			loss_full = tf.contrib.seq2seq.sequence_loss(
				self.logits,
				self.y,
				self.weights,
				average_across_timesteps=True,
				average_across_batch=True
			)

			loss_pl1 = tf.contrib.seq2seq.sequence_loss(
				self.logits,
				self.y,
				self.pl_w1,
				average_across_timesteps=True,
				average_across_batch=True
			)

			loss_pl2 = tf.contrib.seq2seq.sequence_loss(
				self.logits,
				self.y,
				tf.cast(tf.ones_like(self.y), tf.float32),
				average_across_timesteps=False,
				average_across_batch=False
			)
			loss_pl2 = tf.reduce_sum(tf.multiply(self.pl_w2, -tf.log(1 - tf.exp(-loss_pl2) + 1e-12))) / (tf.reduce_sum(self.pl_w2) + 1e-12)

			self.loss = tf.cond(self.pl, lambda: loss_pl1 + loss_pl2, lambda: loss_full)

			self.lm_loss = self.fw_loss + self.bw_loss
			self.total_loss = self.lm_loss + self.loss

		with tf.variable_scope('train_op'):
			tvars = tf.trainable_variables()
			grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), max_grad_norm)
			lm_grads, _ = tf.clip_by_global_norm(tf.gradients(self.lm_loss, tvars), max_grad_norm)
			total_grads, _ = tf.clip_by_global_norm(tf.gradients(self.total_loss, tvars), max_grad_norm)

			self.lr = tf.Variable(learning_rate, trainable=False)
			self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
			self.lr_update = tf.assign(self.lr, self.new_lr)

			# self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr)
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
			self.train_op = self.optimizer.apply_gradients(zip(grads, tvars))
			self.lm_train_op = self.optimizer.apply_gradients(zip(lm_grads, tvars))
			self.total_train_op = self.optimizer.apply_gradients(zip(total_grads, tvars))

	def assign_lr(self, session, lr_value):
		session.run(self.lr_update, feed_dict={self.new_lr: lr_value})


# class LSTMLMConfig(object):

# 	def __init__(self):
# 		self.vocab_size = 20000
# 		self.embedding_size = 100
# 		self.batch_size = 20
# 		self.hidden_size = 300
# 		self.max_grad_norm = 5
# 		self.num_layers = 2
# 		self.num_classes = 4
# 		self.learning_rate = 1e-3
# 		self.max_epochs = 30
# 		self.lr_decay = 0.5
# 		self.dropout_keep_prob = 0.55
# 		self.bi_direction = True

# 	def __str__(self):
# 		return "vocab_size: %s batch_size: %s embedding_size: %s num_layers: %s hidden_size: %s learning_rate: %s dropout_keep_prob: %s" % \
# 				(str(self.vocab_size), str(self.batch_size), str(self.embedding_size), str(self.num_layers), \
# 				str(self.hidden_size), str(self.learning_rate), str(self.dropout_keep_prob))


# class LSTMLMModel(object):

# 	def __init__(self, hidden_size, max_grad_norm, num_layers, vocab_size, 
# 				embedding_size, num_classes, learning_rate, bi_direction, init_embedding, init_bi_embedding):

# 		self.x = tf.placeholder(dtype=tf.int32, shape=[None, None, 3], name='input_x')
# 		self.y = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_y')
# 		self.fw_y = tf.placeholder(dtype=tf.int32, shape=[None, None], name='forward_y')
# 		self.bw_y = tf.placeholder(dtype=tf.int32, shape=[None, None], name='backward_y')
# 		self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_keep_prob')
# 		self.weights = tf.placeholder(dtype=tf.float32, shape=[None, None], name='weights')
# 		self.pl = tf.placeholder(dtype=tf.bool, name="pl")

# 		self.bi_x = tf.placeholder(dtype=tf.int32, shape=[None, None, 2], name='input_bi_x')

# 		self.pl_w1 = tf.multiply(self.weights, tf.cast(tf.less(self.weights, tf.ones_like(self.weights) * 0.8), tf.float32))
# 		self.pl_w2 = tf.cast(tf.greater(self.weights, tf.ones_like(self.weights) * 0.8), tf.float32)

# 		self.seq_length = tf.reduce_sum(tf.cast(tf.less(self.x[:, :, 1], tf.ones_like(self.x[:, :, 1]) * (vocab_size-1)), tf.int32), 1)
# 		self.lm_weights = tf.cast(tf.less(self.x[:, :, 1], tf.ones_like(self.x[:, :, 1]) * (vocab_size-1)), tf.float32)
# 		self.batch_size = tf.shape(self.x)[0]

# 		with tf.variable_scope('bi_embedding'):
# 			self.bi_embedding = tf.Variable(init_bi_embedding, dtype=tf.float32, name='bi_embedding', trainable=True)
# 			bi_x = tf.nn.embedding_lookup(self.bi_embedding, self.bi_x)
# 			bi_x = tf.reshape(bi_x, [self.batch_size, -1, 2*embedding_size])
# 			bi_x = tf.nn.dropout(bi_x, self.dropout_keep_prob)

# 		with tf.variable_scope('embedding'):
# 			self.embedding = tf.Variable(init_embedding, dtype=tf.float32, name='embedding', trainable=True)
# 			x = tf.nn.embedding_lookup(self.embedding, self.x)
# 			x = tf.reshape(x, [self.batch_size, -1, 3*embedding_size])
# 			x = tf.nn.dropout(x, self.dropout_keep_prob)

# 			combine_x = tf.concat([x, bi_x], 2)

# 		def lstm_cell():
# 			return tf.contrib.rnn.BasicLSTMCell(hidden_size)

# 		def attn_cell():
# 			return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=self.dropout_keep_prob)

# 		with tf.variable_scope('lm-lstm'):

# 			self.lm_fw_multi_cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(num_layers)])
# 			self.lm_bw_multi_cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(num_layers)])

# 			(self.lm_fw_outputs, self.lm_bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
# 				cell_fw=self.lm_fw_multi_cell,
# 				cell_bw=self.lm_bw_multi_cell,
# 				inputs=x[:, :, embedding_size:2*embedding_size],
# 				sequence_length=self.seq_length,
# 				dtype=tf.float32
# 			)

# 		with tf.variable_scope('bi-lstm'):

# 			self.fw_multi_cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(num_layers)])
# 			self.bw_multi_cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(num_layers)])

# 			if bi_direction == False:
# 				self.bi_outputs, _ = tf.nn.dynamic_rnn(
# 					cell=self.fw_multi_cell,
# 					inputs=combine_x,
# 					sequence_length=self.seq_length,
# 					dtype=tf.float32
# 				)

# 			else:
# 				(fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
# 					cell_fw=self.fw_multi_cell,
# 					cell_bw=self.bw_multi_cell,
# 					inputs=combine_x,
# 					sequence_length=self.seq_length,
# 					dtype=tf.float32
# 				)

# 				self.bi_outputs = tf.concat((fw_outputs, bw_outputs), 2)


# 		with tf.variable_scope('loss'):

# 			fw_softmax_w = tf.get_variable("fw_softmax_w", [hidden_size, vocab_size])
# 			fw_softmax_b = tf.get_variable("fw_softmax_b", [vocab_size])
# 			bw_softmax_w = tf.get_variable("bw_softmax_w", [hidden_size, vocab_size])
# 			bw_softmax_b = tf.get_variable("bw_softmax_b", [vocab_size])

# 			self.fw_logits = tf.einsum("aij,jk->aik", self.lm_fw_outputs, fw_softmax_w) + fw_softmax_b
# 			self.bw_logits = tf.einsum("aij,jk->aik", self.lm_bw_outputs, bw_softmax_w) + bw_softmax_b

# 			W1_z = tf.get_variable("W1_z", [hidden_size, 2*hidden_size])
# 			W2_z = tf.get_variable("W2_z", [hidden_size, 2*hidden_size])
# 			U_z = tf.get_variable("U_z", [2*hidden_size, 2*hidden_size])
# 			b_z = tf.get_variable("b_z", [2*hidden_size])

# 			W1_r = tf.get_variable("W1_r", [hidden_size, 2*hidden_size])
# 			W2_r = tf.get_variable("W2_r", [hidden_size, 2*hidden_size])
# 			U_r = tf.get_variable("U_r", [2*hidden_size, 2*hidden_size])
# 			b_r = tf.get_variable("b_r", [2*hidden_size])

# 			W1_h = tf.get_variable("W1_h", [hidden_size, 2*hidden_size])
# 			W2_h = tf.get_variable("W2_h", [hidden_size, 2*hidden_size])
# 			U_h = tf.get_variable("U_h", [2*hidden_size, 2*hidden_size])
# 			b_h = tf.get_variable("b_h", [2*hidden_size])

# 			z = tf.sigmoid(tf.einsum("aij,jk->aik", self.lm_fw_outputs, W1_z) \
# 						+ tf.einsum("aij,jk->aik", self.lm_bw_outputs, W2_z) \
# 						+ tf.einsum("aij,jk->aik", self.bi_outputs, U_z) \
# 						+ b_z)

# 			r = tf.sigmoid(tf.einsum("aij,jk->aik", self.lm_fw_outputs, W1_r) \
# 						+ tf.einsum("aij,jk->aik", self.lm_bw_outputs, W2_r) \
# 						+ tf.einsum("aij,jk->aik", self.bi_outputs, U_r) \
# 						+ b_r)

# 			h_hat = tf.sigmoid(tf.einsum("aij,jk->aik", self.lm_fw_outputs, W1_h) \
# 							+ tf.einsum("aij,jk->aik", self.lm_bw_outputs, W2_h) \
# 							+ tf.einsum("aij,jk->aik", tf.multiply(r, self.bi_outputs), U_h) \
# 							+ b_h)

# 			self.outputs = tf.multiply((1 - z), self.bi_outputs) + tf.multiply(z, h_hat)
		
# 			softmax_w = tf.get_variable("softmax_w", [2*hidden_size, num_classes])
# 			softmax_b = tf.get_variable("softmax_b", [num_classes])

# 			self.logits = tf.einsum("aij,jk->aik", self.outputs, softmax_w) + softmax_b
			
# 			self.pred = tf.cast(tf.argmax(self.logits, axis=2), tf.int32)
# 			correct_pred = tf.cast(tf.equal(self.pred, self.y), tf.float32)
# 			# self.accuracy = tf.reduce_sum(tf.multiply(self.weights, correct_pred)) / tf.reduce_sum(self.weights)
# 			accuracy_full = tf.reduce_sum(tf.multiply(self.weights, correct_pred)) / (tf.reduce_sum(self.weights) + 1e-12)
# 			accuracy_pl = tf.reduce_sum(tf.multiply(self.pl_w1, correct_pred)) / (tf.reduce_sum(self.pl_w1) + 1e-12)
# 			self.accuracy = tf.cond(self.pl, lambda: accuracy_pl, lambda: accuracy_full)

# 			self.fw_loss = tf.contrib.seq2seq.sequence_loss(
# 				self.fw_logits,
# 				self.fw_y,
# 				self.lm_weights,
# 				average_across_timesteps=True,
# 				average_across_batch=True
# 			)

# 			self.bw_loss = tf.contrib.seq2seq.sequence_loss(
# 				self.bw_logits,
# 				self.bw_y,
# 				self.lm_weights,
# 				average_across_timesteps=True,
# 				average_across_batch=True
# 			)

# 			loss_full = tf.contrib.seq2seq.sequence_loss(
# 				self.logits,
# 				self.y,
# 				self.weights,
# 				average_across_timesteps=True,
# 				average_across_batch=True
# 			)

# 			loss_pl1 = tf.contrib.seq2seq.sequence_loss(
# 				self.logits,
# 				self.y,
# 				self.pl_w1,
# 				average_across_timesteps=True,
# 				average_across_batch=True
# 			)

# 			loss_pl2 = tf.contrib.seq2seq.sequence_loss(
# 				self.logits,
# 				self.y,
# 				tf.cast(tf.ones_like(self.y), tf.float32),
# 				average_across_timesteps=False,
# 				average_across_batch=False
# 			)
# 			loss_pl2 = tf.reduce_sum(tf.multiply(self.pl_w2, -tf.log(1 - tf.exp(-loss_pl2) + 1e-12))) / (tf.reduce_sum(self.pl_w2) + 1e-12)

# 			tvars = tf.trainable_variables()
# 			l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])

# 			# self.loss = tf.cond(self.pl, lambda: loss_pl1 + loss_pl2, lambda: loss_full) + 0.0001 * l2_loss
# 			self.loss = tf.cond(self.pl, lambda: loss_pl1 + loss_pl2, lambda: loss_full)

# 			self.lm_loss = self.fw_loss + self.bw_loss
# 			self.total_loss = self.lm_loss + self.loss

# 		with tf.variable_scope('train_op'):
# 			grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), max_grad_norm)
# 			lm_grads, _ = tf.clip_by_global_norm(tf.gradients(self.lm_loss, tvars), max_grad_norm)
# 			total_grads, _ = tf.clip_by_global_norm(tf.gradients(self.total_loss, tvars), max_grad_norm)

# 			self.lr = tf.Variable(learning_rate, trainable=False)
# 			self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
# 			self.lr_update = tf.assign(self.lr, self.new_lr)

# 			# self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr)
# 			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
# 			self.train_op = self.optimizer.apply_gradients(zip(grads, tvars))
# 			self.lm_train_op = self.optimizer.apply_gradients(zip(lm_grads, tvars))
# 			self.total_train_op = self.optimizer.apply_gradients(zip(total_grads, tvars))

# 	def assign_lr(self, session, lr_value):
# 		session.run(self.lr_update, feed_dict={self.new_lr: lr_value})


# class LSTMConfig(object):

# 	def __init__(self):
# 		self.vocab_size = 20000
# 		self.embedding_size = 100
# 		self.batch_size = 20
# 		self.hidden_size = 300
# 		self.max_grad_norm = 5
# 		self.num_layers = 2
# 		self.num_classes = 4
# 		self.learning_rate = 1e-3
# 		self.max_epochs = 40
# 		self.lr_decay = 0.5
# 		self.dropout_keep_prob = 0.8
# 		self.bi_direction = True

# 	def __str__(self):
# 		return "vocab_size: %s batch_size: %s embedding_size: %s num_layers: %s hidden_size: %s learning_rate: %s dropout_keep_prob: %s" % \
# 				(str(self.vocab_size), str(self.batch_size), str(self.embedding_size), str(self.num_layers), \
# 				str(self.hidden_size), str(self.learning_rate), str(self.dropout_keep_prob))


# class LSTMModel(object):

# 	def __init__(self, hidden_size, max_grad_norm, num_layers, vocab_size, 
# 				embedding_size, num_classes, learning_rate, bi_direction, init_embedding, init_bi_embedding):

# 		self.x = tf.placeholder(dtype=tf.int32, shape=[None, None, 3], name='input_x')
# 		self.y = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_y')
# 		self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_keep_prob')
# 		self.weights = tf.placeholder(dtype=tf.float32, shape=[None, None], name='weights')
# 		self.pl = tf.placeholder(dtype=tf.bool, name="pl")

# 		self.bi_x = tf.placeholder(dtype=tf.int32, shape=[None, None, 2], name='input_bi_x')

# 		self.pl_w1 = tf.multiply(self.weights, tf.cast(tf.less(self.weights, tf.ones_like(self.weights) * 0.8), tf.float32))
# 		self.pl_w2 = tf.cast(tf.greater(self.weights, tf.ones_like(self.weights) * 0.8), tf.float32)

# 		self.seq_length = tf.reduce_sum(tf.cast(tf.less(self.x[:, :, 1], tf.ones_like(self.x[:, :, 1]) * (vocab_size-1)), tf.int32), 1)
# 		self.batch_size = tf.shape(self.x)[0]

# 		with tf.variable_scope('bi_embedding'):
# 			# self.bi_embedding = tf.Variable(init_bi_embedding, dtype=tf.float32, name='bi_embedding', trainable=True)
# 			self.bi_embedding = tf.Variable(init_bi_embedding, dtype=tf.float32, name='bi_embedding', trainable=False)
# 			bi_x = tf.nn.embedding_lookup(self.bi_embedding, self.bi_x)
# 			bi_x = tf.reshape(bi_x, [self.batch_size, -1, 2*embedding_size])
# 			bi_x = tf.nn.dropout(bi_x, self.dropout_keep_prob)

# 		with tf.variable_scope('embedding'):
# 			self.embedding = tf.Variable(init_embedding, dtype=tf.float32, name='embedding', trainable=True)
# 			x = tf.nn.embedding_lookup(self.embedding, self.x)
# 			x = tf.reshape(x, [self.batch_size, -1, 3*embedding_size])
# 			x = tf.nn.dropout(x, self.dropout_keep_prob)

# 			combine_x = tf.concat([x, bi_x], 2)

# 		with tf.variable_scope('lstm'):

# 			def lstm_cell():
# 				return tf.contrib.rnn.BasicLSTMCell(hidden_size)

# 			def attn_cell():
# 				return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=self.dropout_keep_prob)

# 			self.fw_multi_cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(num_layers)])
# 			self.bw_multi_cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(num_layers)])

# 			if bi_direction == False:
# 				self.outputs, _ = tf.nn.dynamic_rnn(
# 					cell=self.fw_multi_cell,
# 					inputs=combine_x,
# 					sequence_length=self.seq_length,
# 					dtype=tf.float32
# 				)

# 			else:
# 				(fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
# 					cell_fw=self.fw_multi_cell,
# 					cell_bw=self.bw_multi_cell,
# 					inputs=combine_x,
# 					sequence_length=self.seq_length,
# 					dtype=tf.float32
# 				)

# 				self.outputs = tf.concat((fw_outputs, bw_outputs), 2)


# 		with tf.variable_scope('loss'):
# 			if bi_direction == False:
# 				softmax_w = tf.get_variable("softmax_w", [hidden_size, num_classes])
# 			else:
# 				softmax_w = tf.get_variable("softmax_w", [hidden_size * 2, num_classes])
# 			softmax_b = tf.get_variable("softmax_b", [num_classes])

# 			self.logits = tf.einsum("aij,jk->aik", self.outputs, softmax_w) + softmax_b
# 			self.pred = tf.cast(tf.argmax(self.logits, axis=2), tf.int32)
# 			correct_pred = tf.cast(tf.equal(self.pred, self.y), tf.float32)
# 			accuracy_full = tf.reduce_sum(tf.multiply(self.weights, correct_pred)) / (tf.reduce_sum(self.weights) + 1e-12)
# 			accuracy_pl = tf.reduce_sum(tf.multiply(self.pl_w1, correct_pred)) / (tf.reduce_sum(self.pl_w1) + 1e-12)
# 			self.accuracy = tf.cond(self.pl, lambda: accuracy_pl, lambda: accuracy_full)

# 			loss_full = tf.contrib.seq2seq.sequence_loss(
# 				self.logits,
# 				self.y,
# 				self.weights,
# 				average_across_timesteps=True,
# 				average_across_batch=True
# 			)

# 			loss_pl1 = tf.contrib.seq2seq.sequence_loss(
# 				self.logits,
# 				self.y,
# 				self.pl_w1,
# 				average_across_timesteps=True,
# 				average_across_batch=True
# 			)

# 			loss_pl2 = tf.contrib.seq2seq.sequence_loss(
# 				self.logits,
# 				self.y,
# 				tf.cast(tf.ones_like(self.y), tf.float32),
# 				average_across_timesteps=False,
# 				average_across_batch=False
# 			)
# 			loss_pl2 = tf.reduce_sum(tf.multiply(self.pl_w2, -tf.log(1 - tf.exp(-loss_pl2) + 1e-12))) / (tf.reduce_sum(self.pl_w2) + 1e-12)

# 			tvars = tf.trainable_variables()
# 			l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])

# 			# self.loss = tf.cond(self.pl, lambda: loss_pl1 + loss_pl2, lambda: loss_full) + 0.0001 * l2_loss
# 			self.loss = tf.cond(self.pl, lambda: loss_pl1 + loss_pl2, lambda: loss_full)
			

# 		with tf.variable_scope('train_op'):
# 			grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), max_grad_norm)

# 			self.lr = tf.Variable(learning_rate, trainable=False)
# 			self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
# 			self.lr_update = tf.assign(self.lr, self.new_lr)

# 			# self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr)
# 			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
# 			self.train_op = self.optimizer.apply_gradients(zip(grads, tvars))

# 	def assign_lr(self, session, lr_value):
# 		session.run(self.lr_update, feed_dict={self.new_lr: lr_value})