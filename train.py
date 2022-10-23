import numpy as np
import random
import itertools
import pickle

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
from tensorflow.keras import Model

def fifty_fifty():
    if random.random() < .5:
        return True
    return False

# following five functions inspired by https://github.com/oripress/DeepChess

pieces = {
	'p': 1,
	'P': -1,
	'n': 2,
	'N': -2,
	'b': 3,
	'B': -3,
	'r': 4,
	'R': -4,
	'q': 5,
	'Q': -5,
	'k': 6,
	'K': -6
		}

def shortenString(s):
	s = s[:s.rfind(" ")]
	return s;

def beautifyFEN(f):
	for i in range(4):
		f = shortenString(f)
	toMove = f[-1]
	if toMove == 'w':
		toMove = 7
	else:
		toMove = -7
	f = shortenString(f)
	newf = []
	for char in f:
		if char.isdigit():
			for i in range(int(char)):
				newf.append(0)
		elif char != '/':
			newf.append(pieces[char])	
	newf.append(toMove)
	return	newf

def make_bitboard(f):
	arrs = []
	result = []
	s = 	{
		'1' : 0,
		'2' : 1,
		'3' : 2,
		'4' : 3,
		'5' : 4,
		'6' : 5,
		'-1' : 6,
		'-2' : 7,
		'-3' : 8,
		'-4' : 9,
		'-5' : 10,
		'-6' : 11,
		}	
	for i in range(12):
		arrs.append(np.zeros(64))
	for i in range(64):
		c = str(int(f[i]))
		if c != '0':
			c = int(s[c])
			arrs[c][i] = 1
	for i in range(12):
		result.append(arrs[i])
	result = list(itertools.chain.from_iterable(result))
	if f[64] == -7:
		result.append(1)
	else:
		result.append(0)
	return result

def getTrain(input_size, total, batch_size):
	WWinning = np.zeros((total, input_size))
	BWinning = np.zeros((total, input_size))
	range1 = int(total/batch_size)
	for i in range(range1):
		print (i)
		path = 'game_data/volume' + str(i) + '.p'
		f = open(path, 'rb')
		full_data = pickle.load(f, encoding="bytes")
		curX = full_data[b'x']
		curX = np.array(curX)
		curL = full_data[b'x_labels']
		curL = np.array(curL)
		f.close()
		for j in range(batch_size):
			if curL[j][0] == 1:
				first = make_bitboard(curX[j][0])	
				second = make_bitboard(curX[j][1])
			else:
				first = make_bitboard(curX[j][1])	
				second = make_bitboard(curX[j][0])
			WWinning[i*batch_size+j] = first
			BWinning[i*batch_size+j] = second 
	return (WWinning, BWinning)

def get_batch(begin, amount, test):
	global WWinning_t
	global BWinning_t
	global WWinning_test
	global BWinning_test

	if test:
		X_test = []
		y_test = []
		for i in range(begin,begin+amount):
			random = fifty_fifty()
			if random:
				pos = [WWinning_test[i], BWinning_test[i]]
				y = 1
			else:
				pos = [BWinning_test[i], WWinning_test[i]]
				y = 0
			X_test.append(pos)
			y_test.append(y)
		return (X_test, y_test)
	else:
		X_train = []
		y_train = []
		for i in range(begin,begin+amount):
			random = fifty_fifty()
			if random:
				pos = [WWinning_t[i], BWinning_t[i]]
				y = 1
			else:
				pos = [BWinning_t[i], WWinning_t[i]]
				y = 0
			X_train.append(pos)
			y_train.append(y)
		return (X_train, y_train)

def train_test_split(WWinning, BWinning):
    WWinning_t = WWinning[:1000000]
    WWinning_test = WWinning[1000000:1050000]
    BWinning_t = BWinning[:1000000]
    BWinning_test = BWinning[1000000:1050000]
    return WWinning_t, BWinning_t, WWinning_test, BWinning_test

def make_model():
	first_game = tf.keras.layers.Input(shape=(769))
	first_dense = tf.keras.layers.Dense(200, activation="relu", kernel_regularizer=regularizers.l2(0.0001))(first_game)
	first_drop = Dropout(0.5)(first_dense)

	second_game = tf.keras.layers.Input(shape=(769))
	second_dense = tf.keras.layers.Dense(200, activation="relu", kernel_regularizer=regularizers.l2(0.0001))(second_game)
	second_drop = Dropout(0.5)(second_dense)

	merge = tf.keras.layers.concatenate([first_drop, second_drop])
	hidden1 = tf.keras.layers.Dense(400, activation="relu", kernel_regularizer=regularizers.l2(0.0001))(merge)
	dropout1 = Dropout(0.5)(hidden1)
	hidden2 = tf.keras.layers.Dense(100, activation="relu", kernel_regularizer=regularizers.l2(0.0001))(dropout1)
	dropout2 = Dropout(0.5)(hidden2)
	output = tf.keras.layers.Dense(1, activation="sigmoid")(dropout2)

	model = Model(inputs=[first_game, second_game], outputs=output)

	model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
	return model

def save_tf_small_model(model, filename):
	# save model as tf lite model for performance improvement
	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	tflite_model = converter.convert()
	with open(filename, 'wb') as f:
		f.write(tflite_model)

def train_model(model, epochs, starting_lr, lr_mp):

	global WWinning_t
	global WWinning_test
	global BWinning_t
	global BWinning_test

	lr = starting_lr

	for epoch in range(epochs):
		# shuffle training data
		WWinning_t = np.random.permutation(WWinning_t)
		BWinning_t = np.random.permutation(BWinning_t)
		WWinning_test = np.random.permutation(WWinning_test)
		BWinning_test = np.random.permutation(BWinning_test)
		# set learning rate
		tf.keras.backend.set_value(model.optimizer.lr, lr)
		lr = lr * lr_mp
		print (model.optimizer.lr)
		# split training into 39 * 25,000 datapoints
		for j in range(0, 39, 1):
			print ("Outer Epoch #", epoch, "\nInner Epoch #", j)
			# get training data
			X, y = get_batch(j*25000, 25000, False)
			# get test data
			X_test, y_test = get_batch(j*1000, 1000, True)
			# transform to array
			X_test = np.array(X_test)
			y_test = np.array(y_test)
			X = np.array(X)
			y = np.array(y)
			first_board, second_board = tf.unstack(X, axis=1)
			# fit model
			model.fit(x=[first_board, second_board], y=y, epochs=1, verbose=1, validation_data=([X_test[:,0], X_test[:,1]], y_test), batch_size=25000)

WWinning, BWinning = getTrain(769, 1075000, 25000)
WWinning_t, BWinning_t, WWinning_test, BWinning_test = train_test_split(WWinning, BWinning)

K.clear_session()
model = make_model()

train_model(model, epochs=300, starting_lr=0.001, lr_mp=0.99)
save_tf_small_model(model, "model.tflite")



