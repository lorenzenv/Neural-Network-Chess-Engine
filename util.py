import numpy as np
import chess
import itertools

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