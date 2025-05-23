import chess
import numpy as np
import tflite_runtime.interpreter as tflite
from util import *

# load model
interpreter = tflite.Interpreter(model_path="model.tflite")
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

# get raw model evaluation for two input positions
def get_raw_model_evaluation(first_fen, second_fen):
    x_1, x_2 = make_x(first_fen,second_fen)
    interpreter.set_tensor(input_details[0]['index'], x_1)
    interpreter.set_tensor(input_details[1]['index'], x_2)
    interpreter.invoke()
    evaluation = interpreter.get_tensor(output_details[0]['index'])[0][0]
    return evaluation

# evaluate two input positions, adjusting score based on whose turn it was in the first FEN
def evaluate_pos(first_fen, second_fen):
    raw_score = get_raw_model_evaluation(first_fen, second_fen)
    board = chess.Board(first_fen)
    if board.turn == chess.BLACK:
        return -raw_score
    return raw_score

class Engine:
    def __init__(self, fen):
        self.board = chess.Board()
        self.board.set_fen(fen)
        self.position_cache = {}

    # is called in flask app
    def get_move(self):
        best_move = self.make_move()
        return str(best_move)

    # get best move for black (AI)
    def make_move(self):
        current_fen = self.board.fen()
        print("calculating\n")
        
        black_legal_moves = list(self.board.legal_moves)
        if not black_legal_moves:
            return "no legal moves"
        
        best_move = None
        best_value = float('-inf')  # Black wants to maximize its score
        
        for black_move in black_legal_moves:
            print(".")
            self.board.push(black_move)
            
            # Check for immediate checkmate
            if self.board.is_checkmate():
                self.board.pop()
                return str(black_move)
            
            # Get White's best response (minimize Black's score)
            white_value = self.get_white_response(current_fen, float('-inf'), float('inf'))
            self.board.pop()
            
            # Black wants the move that gives the best result after White's best response
            if white_value > best_value:
                best_value = white_value
                best_move = black_move
        
        if best_move is None:
            print("CHECKMATE or no moves")
            return "checkmate"
        
        print("best move found:", best_move)
        return str(best_move)
    
    def get_white_response(self, original_fen, alpha, beta):
        """Get White's best response to Black's move"""
        white_legal_moves = list(self.board.legal_moves)
        if not white_legal_moves:
            return 0  # No moves available
        
        min_value = float('inf')  # White tries to minimize Black's advantage
        
        for white_move in white_legal_moves:
            self.board.push(white_move)
            
            # Check for White checkmate (bad for Black)
            if self.board.is_checkmate():
                self.board.pop()
                return -10000  # Very bad for Black
            
            # Get Black's best counter-response
            black_value = self.get_black_counter_response(original_fen, alpha, beta)
            self.board.pop()
            
            min_value = min(min_value, black_value)
            beta = min(beta, black_value)
            
            # Alpha-beta pruning
            if beta <= alpha:
                break
                
        return min_value
    
    def get_black_counter_response(self, original_fen, alpha, beta):
        """Get Black's best counter-response at depth 3"""
        black_legal_moves = list(self.board.legal_moves)
        if not black_legal_moves:
            return 0
        
        max_value = float('-inf')  # Black tries to maximize
        
        for black_move in black_legal_moves:
            self.board.push(black_move)
            
            # Check for Black checkmate (good for Black)
            if self.board.is_checkmate():
                self.board.pop()
                return 10000  # Very good for Black
            
            # Evaluate position using neural network
            current_fen = self.board.fen()
            if current_fen in self.position_cache:
                evaluation = self.position_cache[current_fen]
            else:
                evaluation = evaluate_pos(original_fen, current_fen)
                self.position_cache[current_fen] = evaluation
                
            self.board.pop()
            
            max_value = max(max_value, evaluation)
            alpha = max(alpha, evaluation)
            
            # Alpha-beta pruning
            if beta <= alpha:
                break
                
        return max_value