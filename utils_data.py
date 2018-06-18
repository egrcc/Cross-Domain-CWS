import numpy as np
import codecs
import cPickle as pickle
import random
import os
import re
import collections
from gensim.models import word2vec


def build_zx_vocab():
	f1 = codecs.open("dataset/preprocess_data/ctb/ctb_train.txt", "r", 'utf-8')
	f2 = codecs.open("dataset/preprocess_data/zx/zx_test.txt", "r", 'utf-8')
	f3 = codecs.open("dataset/preprocess_data/zx/zx_valid.txt", "r", 'utf-8')
	f4 = codecs.open("dataset/preprocess_data/zx/zx_ul_train.txt", "r", 'utf-8')
	# f5 = codecs.open("dataset/preprocess_data/zx/zx_pl_train.txt", "r", 'utf-8')

	data = []
	data.extend(f1.read().split())
	data.extend(f2.read().split())
	data.extend(f3.read().split())
	data.extend(f4.read().split())
	# data.extend(f5.read().split())

	counter = collections.Counter("".join(data))
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

	id2char, _ = list(zip(*count_pairs))
	char2id = dict(zip(id2char, range(len(id2char))))

	pickle.dump( id2char, open("data/zx_id2char.p", "wb") )
	pickle.dump( char2id, open("data/zx_char2id.p", "wb") )


def build_zx_bi_vocab():
	f1 = codecs.open("dataset/preprocess_data/ctb/ctb_train.txt", "r", 'utf-8')
	f2 = codecs.open("dataset/preprocess_data/zx/zx_test.txt", "r", 'utf-8')
	f3 = codecs.open("dataset/preprocess_data/zx/zx_valid.txt", "r", 'utf-8')
	# f4 = codecs.open("dataset/preprocess_data/zx/zx_ul_train.txt", "r", 'utf-8')
	f5 = codecs.open("dataset/preprocess_data/zx/zx_pl_train.txt", "r", 'utf-8')

	data = []
	data.extend(f1.read().split())
	data.extend(f2.read().split())
	data.extend(f3.read().split())
	# data.extend(f4.read().split())
	data.extend(f5.read().split())
	data = "".join(data)

	bi_grams = []
	for i in range(len(data)):
		if i != len(data) - 1:
			bi_grams.append(data[i:i+2])


	counter = collections.Counter(bi_grams)
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

	id2char, _ = list(zip(*count_pairs))
	char2id = dict(zip(id2char, range(len(id2char))))

	pickle.dump( id2char, open("data/zx_bi_id2char.p", "wb") )
	pickle.dump( char2id, open("data/zx_bi_char2id.p", "wb") )


def get_embedding(target):
	if os.path.exists("data/%s_embedding.p" % (target, )) == False:
		if target == "zx":
			f1 = codecs.open("dataset/preprocess_data/ctb/ctb_train.txt", "r", 'utf-8')
		elif target == "msr":
			f1 = codecs.open("dataset/preprocess_data/msr/msr_train.txt", "r", 'utf-8')
		elif target == "pku":
			f1 = codecs.open("dataset/preprocess_data/pku/pku_train.txt", "r", 'utf-8')
		elif target == "ctb6":
			f1 = codecs.open("dataset/preprocess_data/ctb6/ctb6_train.txt", "r", 'utf-8')
		else:
			f1 = codecs.open("dataset/preprocess_data/pd/pd_train.txt", "r", 'utf-8')

		f2 = codecs.open("dataset/preprocess_data/%s/%s_test.txt" % (target, target), "r", 'utf-8')
		f3 = codecs.open("dataset/preprocess_data/%s/%s_valid.txt" % (target, target), "r", 'utf-8')
		f4 = codecs.open("dataset/preprocess_data/%s/%s_ul_train.txt" % (target, target), "r", 'utf-8')
		f5 = codecs.open("dataset/preprocess_data/%s/%s_pl_train.txt" % (target, target), "r", 'utf-8')

		data = []
		data.extend([list("".join(line.split())) for line in f1.readlines()])
		data.extend([list("".join(line.split())) for line in f2.readlines()])
		data.extend([list("".join(line.split())) for line in f3.readlines()])
		data.extend([list("".join(line.split())) for line in f4.readlines()])

		if target == "zx":
			data.extend([list("".join(line.split())) for line in f5.readlines()])
		else:
			p = u'\[\[(.*?)\]\]'
			data.extend([list(re.sub(p, r'\1', "".join(line.split()))) for line in f5.readlines()])

		model = word2vec.Word2Vec(data, size=100, workers=16, min_count=1, window=8, iter=20)

		id2char = pickle.load( open("data/%s_id2char.p" % (target, ), "rb") )

		embedding = np.zeros((len(id2char) + 1, 100))

		for i in range(len(id2char)):
			embedding[i] = model[id2char[i]]

		pickle.dump( embedding, open("data/%s_embedding.p" % (target, ), "wb") )
	else:
		embedding = pickle.load( open("data/%s_embedding.p" % (target, ), "rb") )

	return embedding


def get_bi_embedding(target):
	if os.path.exists("data/%s_bi_embedding.p" % (target, )) == False:
		
		bi_id2char = pickle.load( open("data/%s_bi_id2char.p" % (target, ), "rb") )
		bi_embedding = np.zeros((len(bi_id2char) + 1, 100))

		embedding = get_embedding(target)
		char2id = pickle.load( open("data/%s_char2id.p" % (target, ), "rb") )

		for i in range(len(bi_id2char)):
			bi_embedding[i] = (embedding[char2id[bi_id2char[i][0]]] + embedding[char2id[bi_id2char[i][1]]]) / 2

		pickle.dump( bi_embedding, open("data/%s_bi_embedding.p" % (target, ), "wb") )
	else:
		bi_embedding = pickle.load( open("data/%s_bi_embedding.p" % (target, ), "rb") )

	return bi_embedding


def _vectorize(data, char2id, bi_char2id, bi_vocab_size):
	X = []
	Y = []
	bi_X = []
	for line in data:
		words = line.split()

		str_ = "".join(words)

		x = map(lambda k: char2id[k], str_)
		X.append(x)

		bi_X.append([bi_vocab_size-1] + [bi_char2id[str_[i:i+2]] for i in range(len(str_)-1)] + [bi_vocab_size-1])
		
		y = []
		for w in words:
			if len(w) == 1:
				y.append(3)
			elif len(w) == 2:
				y.extend([0, 2])
			elif len(w) > 2:
				y.extend([0] + [1]*(len(w)-2) + [2])
		Y.append(y)

		assert len(x) == len(y)

	return X, Y, bi_X


def _vectorize_ul(data, char2id, max_len=20000, max_num=50000000):
	X = []
	for line in data:
		if len(line) > max_len:
			continue

		words = line.split()

		x = map(lambda k: char2id[k], "".join(words))
		X.append(x)

	if len(X) > max_num:
		dummy1 = [0]*len(X)
		dummy2 = [0]*len(X)
		X, _, _ =_shuffle_list(X, dummy1, dummy2)

		return X[:max_num]
	
	return X


def get_ctb_data():
	if os.path.exists("data/ctb_train_X.p") == False:
		data = codecs.open("dataset/preprocess_data/ctb/ctb_train.txt", "r", 'utf-8').readlines()
		char2id = pickle.load( open("data/zx_char2id.p", "rb") )
		bi_char2id = pickle.load( open("data/zx_bi_char2id.p", "rb") )
		
		train_X, train_Y, train_bi_X = _vectorize(data, char2id, bi_char2id, 250734)

		pickle.dump( train_X, open("data/ctb_train_X.p", "wb") )
		pickle.dump( train_Y, open("data/ctb_train_Y.p", "wb") )
		pickle.dump( train_bi_X, open("data/ctb_train_bi_X.p", "wb") )
	else:
		train_X = pickle.load( open("data/ctb_train_X.p", "rb") )
		train_Y = pickle.load( open("data/ctb_train_Y.p", "rb") )
		train_bi_X = pickle.load( open("data/ctb_train_bi_X.p", "rb") )

	return train_X, train_Y, train_bi_X


def get_zx_data():
	if os.path.exists("data/zx_valid_X.p") == False:
		data_valid = codecs.open("dataset/preprocess_data/zx/zx_valid.txt", "r", 'utf-8').readlines()
		data_test = codecs.open("dataset/preprocess_data/zx/zx_test.txt", "r", 'utf-8').readlines()
		char2id = pickle.load( open("data/zx_char2id.p", "rb") )
		bi_char2id = pickle.load( open("data/zx_bi_char2id.p", "rb") )
		
		valid_X, valid_Y, valid_bi_X = _vectorize(data_valid, char2id, bi_char2id, 250734)
		test_X, test_Y, test_bi_X = _vectorize(data_test, char2id, bi_char2id, 250734)

		pickle.dump( valid_X, open("data/zx_valid_X.p", "wb") )
		pickle.dump( valid_Y, open("data/zx_valid_Y.p", "wb") )
		pickle.dump( test_X, open("data/zx_test_X.p", "wb") )
		pickle.dump( test_Y, open("data/zx_test_Y.p", "wb") )
		pickle.dump( valid_bi_X, open("data/zx_valid_bi_X.p", "wb") )
		pickle.dump( test_bi_X, open("data/zx_test_bi_X.p", "wb") )
	else:
		valid_X = pickle.load( open("data/zx_valid_X.p", "rb") )
		valid_Y = pickle.load( open("data/zx_valid_Y.p", "rb") )
		test_X = pickle.load( open("data/zx_test_X.p", "rb") )
		test_Y = pickle.load( open("data/zx_test_Y.p", "rb") )
		valid_bi_X = pickle.load( open("data/zx_valid_bi_X.p", "rb") )
		test_bi_X = pickle.load( open("data/zx_test_bi_X.p", "rb") )

	return valid_X, valid_Y, valid_bi_X, test_X, test_Y, test_bi_X


def get_zx_ul_data():
	if os.path.exists("data/zx_ul_train_X.p") == False:
		data = codecs.open("dataset/preprocess_data/zx/zx_ul_train.txt", "r", 'utf-8').readlines()
		char2id = pickle.load( open("data/zx_char2id.p", "rb") )

		ul_train_X = _vectorize_ul(data, char2id)

		pickle.dump( ul_train_X, open("data/zx_ul_train_X.p", "wb") )
	else:
		ul_train_X = pickle.load( open("data/zx_ul_train_X.p", "rb") )

	random.shuffle(ul_train_X)
	ul_train_X = ul_train_X[:len(ul_train_X)/3]

	return ul_train_X


def get_zx_pl_data():
	if os.path.exists("data/zx_pl_train_X.p") == False:
		data = codecs.open("dataset/preprocess_data/zx/zx_pl_train.txt", "r", 'utf-8').readlines()
		dict_ = codecs.open("dataset/preprocess_data/zx/zx_dict.txt", "r", 'utf-8').read().split()
		char2id = pickle.load( open("data/zx_char2id.p", "rb") )
		bi_char2id = pickle.load( open("data/zx_bi_char2id.p", "rb") )

		pl_train_X = []
		pl_train_Y = []
		pl_train_W = []
		pl_train_bi_X = []
		for line in data:
			line = line.strip()

			x = map(lambda k: char2id[k], line)
			pl_train_X.append(x)
			pl_train_X.append(x)

			bi_x = [250734-1] + [bi_char2id.get(line[j:j+2], 250734-1) for j in range(len(line)-1)] + [250734-1]
			pl_train_bi_X.append(bi_x)
			pl_train_bi_X.append(bi_x)

			y = [3] * len(x)
			y2 = [3] * len(x)
			weight = [0] * len(x)
			for w in dict_:
				index = line.find(w)
				if index != -1:
					if len(w) == 1:
						y[index] = 3
						y2[index] = 3
						weight[index] = 0.5
						if index != 0:
							y[index-1] = 0
							y2[index-1] = 1
							weight[index-1] = 1
						if index != len(x) - 1:
							y[index+1] = 1
							y2[index+1] = 2
							weight[index+1] = 1
					elif len(w) == 2:
						y[index] = 0
						y[index+1] = 2
						y2[index] = 0
						y2[index+1] = 2
						weight[index] = 0.5
						weight[index+1] = 0.5
						if index != 0:
							y[index-1] = 0
							y2[index-1] = 1
							weight[index-1] = 1
						if (index+1) != len(x) - 1:
							y[index+2] = 1
							y2[index+2] = 2
							weight[index+2] = 1
					elif len(w) > 2:
						y[index] = 0
						y[index+len(w)-1] = 2
						y2[index] = 0
						y2[index+len(w)-1] = 2
						weight[index] = 0.5
						weight[index+len(w)-1] = 0.5
						for i in range(len(w)-2):
							y[index+i+1] = 1
							y2[index+i+1] = 1
							weight[index+i+1] = 0.5
						if index != 0:
							y[index-1] = 0
							y2[index-1] = 1
							weight[index-1] = 1
						if (index+len(w)-1) != len(x) - 1:
							y[index+len(w)] = 1
							y2[index+len(w)] = 2
							weight[index+len(w)] = 1

			pl_train_Y.append(y)
			pl_train_Y.append(y2)
			pl_train_W.append(weight)
			pl_train_W.append(weight)

		pickle.dump( pl_train_X, open("data/zx_pl_train_X.p", "wb") )
		pickle.dump( pl_train_Y, open("data/zx_pl_train_Y.p", "wb") )
		pickle.dump( pl_train_W, open("data/zx_pl_train_W.p", "wb") )
		pickle.dump( pl_train_bi_X, open("data/zx_pl_train_bi_X.p", "wb") )
	else:
		pl_train_X = pickle.load( open("data/zx_pl_train_X.p", "rb") )
		pl_train_Y = pickle.load( open("data/zx_pl_train_Y.p", "rb") )
		pl_train_W = pickle.load( open("data/zx_pl_train_W.p", "rb") )
		pl_train_bi_X = pickle.load( open("data/zx_pl_train_bi_X.p", "rb") )

	return pl_train_X, pl_train_Y, pl_train_bi_X, pl_train_W


def _shuffle_list(a, b, c, d):
	"""
	shuffle a, b, c, d simultaneously
	"""
	x = list(zip(a, b, c, d))
	random.shuffle(x)
	a, b, c, d = zip(*x)

	return a, b, c, d


def _padding(X, value, weight=False):
	max_len = 0

	for x in X:
		if len(x) > max_len:
			max_len = len(x)

	if weight == False:
		padded_X = np.ones((len(X), max_len), dtype=np.int32) * value
	else:
		padded_X = np.ones((len(X), max_len), dtype=np.float32) * value

	for i in range(len(X)):
		for j in range(len(X[i])):
			padded_X[i, j] = X[i][j]

	return padded_X


def data_iterator(X, Y, bi_X, batch_size, shuffle, vocab_size, bi_vocab_size):
	if shuffle == True:
		dummy = [0]*len(X)
		X, Y, bi_X, _ = _shuffle_list(X, Y, bi_X, dummy)

	data_len = len(X)
	batch_len = data_len / batch_size

	for i in range(batch_len):
		batch_X = X[i*batch_size:(i+1)*batch_size]
		batch_Y = Y[i*batch_size:(i+1)*batch_size]
		batch_bi_X = bi_X[i*batch_size:(i+1)*batch_size]

		padded_X = _padding(batch_X, vocab_size-1)
		padded_Y = _padding(batch_Y, 3)
		padded_bi_X = _padding(batch_bi_X, bi_vocab_size-1)
		true_Y = batch_Y

		padded = np.ones((len(padded_X), 1), dtype=np.int32) * (vocab_size-1)
		fw_Y = np.concatenate((padded_X[:, 1:], padded), axis=1)
		bw_Y = np.concatenate((padded, padded_X[:, :-1]), axis=1)

		W = np.less(padded_X, np.ones_like(padded_X) * (vocab_size-1)).astype(np.float32)

		padded_X = np.concatenate((np.expand_dims(bw_Y, axis=2),
								np.expand_dims(padded_X, axis=2),
								np.expand_dims(fw_Y, axis=2)), axis=2)

		fw_bi_X = np.expand_dims(padded_bi_X[:, :-1], axis=2)
		bw_bi_X = np.expand_dims(padded_bi_X[:, 1:], axis=2)

		padded_bi_X = np.concatenate((fw_bi_X, bw_bi_X), axis=2)

		yield padded_X, padded_Y, padded_bi_X, true_Y, fw_Y, bw_Y, W

	padded_X = _padding(X[batch_len*batch_size:], vocab_size-1)
	padded_Y = _padding(Y[batch_len*batch_size:], 3)
	padded_bi_X = _padding(bi_X[batch_len*batch_size:], bi_vocab_size-1)
	true_Y = Y[batch_len*batch_size:]

	padded = np.ones((len(padded_X), 1), dtype=np.int32) * (vocab_size-1)
	fw_Y = np.concatenate((padded_X[:, 1:], padded), axis=1)
	bw_Y = np.concatenate((padded, padded_X[:, :-1]), axis=1)

	W = np.less(padded_X, np.ones_like(padded_X) * (vocab_size-1)).astype(np.float32)

	padded_X = np.concatenate((np.expand_dims(bw_Y, axis=2),
								np.expand_dims(padded_X, axis=2),
								np.expand_dims(fw_Y, axis=2)), axis=2)

	fw_bi_X = np.expand_dims(padded_bi_X[:, :-1], axis=2)
	bw_bi_X = np.expand_dims(padded_bi_X[:, 1:], axis=2)

	padded_bi_X = np.concatenate((fw_bi_X, bw_bi_X), axis=2)
	
	yield padded_X, padded_Y, padded_bi_X, true_Y, fw_Y, bw_Y, W


def ul_data_iterator(X, batch_size, shuffle, vocab_size):
	if shuffle == True:
		dummy1 = [0]*len(X)
		dummy2 = [0]*len(X)
		dummy3 = [0]*len(X)
		X, _, _, _ = _shuffle_list(X, dummy1, dummy2, dummy3)

	data_len = len(X)
	batch_len = data_len / batch_size

	for i in range(batch_len):
		batch_X = X[i*batch_size:(i+1)*batch_size]

		padded_X = _padding(batch_X, vocab_size-1)

		padded = np.ones((len(padded_X), 1), dtype=np.int32) * (vocab_size-1)
		fw_Y = np.concatenate((padded_X[:, 1:], padded), axis=1)
		bw_Y = np.concatenate((padded, padded_X[:, :-1]), axis=1)

		padded_X = np.concatenate((np.expand_dims(bw_Y, axis=2),
								np.expand_dims(padded_X, axis=2),
								np.expand_dims(fw_Y, axis=2)), axis=2)

		yield padded_X, fw_Y, bw_Y

	padded_X = _padding(X[batch_len*batch_size:], vocab_size-1)

	padded = np.ones((len(padded_X), 1), dtype=np.int32) * (vocab_size-1)
	fw_Y = np.concatenate((padded_X[:, 1:], padded), axis=1)
	bw_Y = np.concatenate((padded, padded_X[:, :-1]), axis=1)

	padded_X = np.concatenate((np.expand_dims(bw_Y, axis=2),
								np.expand_dims(padded_X, axis=2),
								np.expand_dims(fw_Y, axis=2)), axis=2)

	yield padded_X, fw_Y, bw_Y


def pl_data_iterator(X, Y, bi_X, W, batch_size, shuffle, vocab_size, bi_vocab_size):
	if shuffle == True:
		X, Y, bi_X, W = _shuffle_list(X, Y, bi_X, W)

	data_len = len(X)
	batch_len = data_len / batch_size

	for i in range(batch_len):
		batch_X = X[i*batch_size:(i+1)*batch_size]
		batch_Y = Y[i*batch_size:(i+1)*batch_size]
		batch_W = W[i*batch_size:(i+1)*batch_size]
		batch_bi_X = bi_X[i*batch_size:(i+1)*batch_size]

		padded_X = _padding(batch_X, vocab_size-1)
		padded_Y = _padding(batch_Y, 3)
		padded_W = _padding(batch_W, 0, weight=True)
		padded_bi_X = _padding(batch_bi_X, bi_vocab_size-1)

		padded = np.ones((len(padded_X), 1), dtype=np.int32) * (vocab_size-1)
		fw_Y = np.concatenate((padded_X[:, 1:], padded), axis=1)
		bw_Y = np.concatenate((padded, padded_X[:, :-1]), axis=1)

		padded_X = np.concatenate((np.expand_dims(bw_Y, axis=2),
								np.expand_dims(padded_X, axis=2),
								np.expand_dims(fw_Y, axis=2)), axis=2)

		fw_bi_X = np.expand_dims(padded_bi_X[:, :-1], axis=2)
		bw_bi_X = np.expand_dims(padded_bi_X[:, 1:], axis=2)

		padded_bi_X = np.concatenate((fw_bi_X, bw_bi_X), axis=2)

		yield padded_X, padded_Y, padded_bi_X, fw_Y, bw_Y, padded_W

	padded_X = _padding(X[batch_len*batch_size:], vocab_size-1)
	padded_Y = _padding(Y[batch_len*batch_size:], 3)
	padded_W = _padding(W[batch_len*batch_size:], 0, weight=True)
	padded_bi_X = _padding(bi_X[batch_len*batch_size:], bi_vocab_size-1)

	padded = np.ones((len(padded_X), 1), dtype=np.int32) * (vocab_size-1)
	fw_Y = np.concatenate((padded_X[:, 1:], padded), axis=1)
	bw_Y = np.concatenate((padded, padded_X[:, :-1]), axis=1)

	padded_X = np.concatenate((np.expand_dims(bw_Y, axis=2),
								np.expand_dims(padded_X, axis=2),
								np.expand_dims(fw_Y, axis=2)), axis=2)

	fw_bi_X = np.expand_dims(padded_bi_X[:, :-1], axis=2)
	bw_bi_X = np.expand_dims(padded_bi_X[:, 1:], axis=2)

	padded_bi_X = np.concatenate((fw_bi_X, bw_bi_X), axis=2)

	yield padded_X, padded_Y, padded_bi_X, fw_Y, bw_Y, padded_W

