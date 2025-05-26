import numpy as np
import chess
import chess.pgn
import random
import itertools
import pickle

# https://github.com/oripress/DeepChess

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
			#newf.append(('pPnNbBrRqQkK'.find(char)+1))
			newf.append(pieces[char])
	
	newf.append(toMove)
#	print(f)
#	print(newf)
	return	newf

def bitifyFEN(f):
	arrs = []
	result = []
	s = 	{
		1 : 0,
		2 : 1,
		3 : 2,
		4 : 3,
		5 : 4,
		6 : 5,
		-1 : 6,
		-2 : 7,
		-3 : 8,
		-4 : 9,
		-5 : 10,
		-6 : 11,
		}
		 	
	for i in range(12):
		arrs.append(np.zeros(64))

	for i in range(64):
		piece_value = f[i]
		if piece_value != 0:
			piece_type = s[piece_value]
			arrs[piece_type][i] = 1

	for i in range(12):
		result.append(arrs[i])
	
	result = list(itertools.chain.from_iterable(result))
	
	if f[64] == -7:
		result.append(1)
	else:
		result.append(0)
	
	return result

#convert()
#bitifyFEN(beautifyFEN('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e3 0 1'))