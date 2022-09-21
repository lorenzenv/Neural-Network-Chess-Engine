import chess
import numpy as np
import tensorflow as tf
from util import *

# load model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# create X
def make_x(first,second):
	x_1 = make_bitboard(beautifyFEN(first))
	x_2 = make_bitboard(beautifyFEN(second))
	x_1 = np.array(x_1, dtype=np.float32).reshape(1,769)
	x_2 = np.array(x_2, dtype=np.float32).reshape(1,769)
	return x_1, x_2

# evaluate two input positions
def evaluate_pos(first, second):
	x_1, x_2 = make_x(first,second)
	interpreter.set_tensor(input_details[0]['index'], x_1)
	interpreter.set_tensor(input_details[1]['index'], x_2)
	interpreter.invoke()
	evaluation = interpreter.get_tensor(output_details[0]['index'])[0][0]
	return evaluation

# create Engine class
class Engine:
	def __init__(self, fen):
		self.board = chess.Board()
		self.board.set_fen(fen)

	# is called in flask app
	def get_move(self):
		best_move = self.make_move()
		return str(best_move)

	# get best move for black
	def make_move(self):
		all_pos = {}
		black_legal_moves = self.board.legal_moves
		current_fen_x = self.board.fen()
		black_response = {}
		print ("calculating\n")
		for black_move in black_legal_moves:
			print (".")
			white_response = {}
			self.board.push(black_move)
			if self.board.is_checkmate():
				return str(black_move)
			white_legal_moves = self.board.legal_moves
			for white_move in white_legal_moves:
				self.board.push(white_move)
				if self.board.is_checkmate():
					white_response[white_move] = 0
					self.board.pop()
					break
				black_legal_moves_depth_2 = self.board.legal_moves
				black_response_depth_2 = {}
				for black_move_depth_2 in black_legal_moves_depth_2:
					self.board.push(black_move_depth_2)
					if self.board.is_checkmate():
						black_response_depth_2[black_move_depth_2] = 1
						self.board.pop()
						break
					next_fen_x = self.board.fen()
					if next_fen_x in all_pos:
						prediction_number = all_pos[next_fen_x]
					else:
						prediction_number = evaluate_pos(current_fen_x, next_fen_x)
						all_pos[next_fen_x] = prediction_number
					if len(white_response) > 0:
						if prediction_number > white_response[max(white_response, key=white_response.get)]:
							black_response_depth_2[black_move_depth_2] = prediction_number
							self.board.pop()
							break
						else:
							black_response_depth_2[black_move_depth_2] = prediction_number
							self.board.pop()
					else:
						black_response_depth_2[black_move_depth_2] = prediction_number
						self.board.pop()
				if len(black_response) > 0 and len(white_response) > 0:
					if white_response[min(white_response, key=white_response.get)] < black_response[max(black_response, key=black_response.get)]:
						white_response[white_move] = black_response_depth_2[max(black_response_depth_2, key=black_response_depth_2.get)]
						self.board.pop()
						break
					else:
						if len(black_response_depth_2) > 0:
							white_response[white_move] = black_response_depth_2[max(black_response_depth_2, key=black_response_depth_2.get)]
						self.board.pop()
				else:
					if len(black_response_depth_2) > 0:
						white_response[white_move] = black_response_depth_2[max(black_response_depth_2, key=black_response_depth_2.get)]
					self.board.pop()
			if len(white_response) > 0:
				black_response[black_move] = white_response[min(white_response, key=white_response.get)]
			self.board.pop()
		if len(black_response) > 0:
			best_move = max(black_response, key=black_response.get)
		else:
			print ("CHECKMATE")
			return str("checkmate")

		print ("best move found: ", best_move)
		return str(best_move)