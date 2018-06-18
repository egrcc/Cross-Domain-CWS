# -*- coding: utf-8 -*-
import cPickle as pickle
import itertools
import codecs
import re


def evaluate_word_PRF(y_pred, y):
	y_pred=list(itertools.chain.from_iterable(y_pred))
	y=list(itertools.chain.from_iterable(y))
	assert len(y_pred)==len(y)
	cor_num = 0
	yp_wordnum = y_pred.count(2) + y_pred.count(3)
	yt_wordnum = y.count(2) + y.count(3)
	start = 0
	for i in xrange(len(y)):
		if y[i] == 2 or y[i] == 3:
			flag = True
			for j in xrange(start, i + 1):
				if y[j] != y_pred[j]:
					flag = False
			if flag == True:
				cor_num += 1
			start = i + 1

	P = cor_num / (float(yp_wordnum) + 1e-12)
	R = cor_num / float(yt_wordnum)
	F = 2 * P * R / (P + R + 1e-12)
	return P, R, F


def get_seg_file(X, pred_Y, target, model_name):
	id2char = pickle.load( open("data/%s_id2char.p" % (target, ), "rb") )
	seg_f = codecs.open("output/%s_test_%s.txt" % (target, model_name), "w", 'utf-8')

	for i in range(len(X)):
		assert len(X[i]) == len(pred_Y[i])

		line = []
		for j in range(len(X[i])):
			if pred_Y[i][j] == 0:
				line.append(" ")
				line.append(id2char[X[i][j]])
			elif pred_Y[i][j] == 1:
				line.append(id2char[X[i][j]])
			elif pred_Y[i][j] == 2:
				line.append(id2char[X[i][j]])
				line.append(" ")
			elif pred_Y[i][j] == 3:
				line.append(" ")
				line.append(id2char[X[i][j]])
				line.append(" ")

		seg_f.write("".join(line) + "\n")

	seg_f.close()


def strQ2B(ustring):
	rstring = ""
	for uchar in ustring:
		inside_code = ord(uchar)
		if inside_code == 12288:
			inside_code = 32
		elif (inside_code >= 65281 and inside_code <= 65374):
			inside_code -= 65248

		rstring += unichr(inside_code)
	return rstring


def preprocess(in_filename, out_filename):
	f_in = codecs.open(in_filename, 'r', 'utf-8')
	f_out = codecs.open(out_filename, 'w', 'utf-8')
	
	# rNUM = u'(-|\+)?\d+((\.|·)\d+)?%?'
	# rENG = u'[A-Za-z_.]+'
	rENGNUM =  u'[A-Za-z_.\d%#+@-]+'
		
	for text in f_in.readlines():
		
		text = strQ2B(text)

		# text=re.sub(rNUM, u'0', text)
		# text=re.sub(rENG, u'X', text)
		text=re.sub(rENGNUM, u'X', text)

		f_out.write(text)

	f_in.close()
	f_out.close()


def preprocess_datasets():
	datasets = ["pd/pd_dict.txt",
				"pd/pd_train.txt",
				"ctb/ctb_dict.txt",
				"ctb/ctb_train.txt",
				"zx/zx_test.txt",
				"zx/zx_ul_train.txt",
				"zx/zx_pl_train.txt",
				"zx/zx_valid.txt",
				"zx/zx_dict.txt",
				"com/com_test.txt",
				"com/com_ul_train.txt",
				"com/com_pl_train.txt",
				"com/com_pl_train_new.txt",
				"com/com_valid.txt",
				"lit/lit_test.txt",
				"lit/lit_ul_train.txt",
				"lit/lit_pl_train.txt",
				"lit/lit_pl_train_new.txt",
				"lit/lit_valid.txt",
				"fin/fin_test.txt",
				"fin/fin_ul_train.txt",
				"fin/fin_pl_train.txt",
				"fin/fin_pl_train_new.txt",
				"med/med_test.txt",
				"med/med_ul_train.txt",
				"med/med_pl_train.txt",
				"med/med_pl_train_new.txt"]

	for d in datasets:
		print d
		preprocess("dataset/raw_data/" + d, "dataset/preprocess_data/" + d)
