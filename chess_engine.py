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

# evaluate two input positions
def evaluate_pos(first, second):
    x_1, x_2 = make_x(first,second)
    interpreter.set_tensor(input_details[0]['index'], x_1)
    interpreter.set_tensor(input_details[1]['index'], x_2)
    interpreter.invoke()
    evaluation = interpreter.get_tensor(output_details[0]['index'])[0][0]
    return evaluation

def order_moves(board, moves):
    """Order moves to search better moves first (for alpha-beta pruning effectiveness)"""
    move_scores = []
    
    for move in moves:
        score = 0
        
        # Prioritize captures
        if board.piece_at(move.to_square) is not None:
            captured_piece = board.piece_at(move.to_square)
            moving_piece = board.piece_at(move.from_square)
            
            piece_values = {
                chess.PAWN: 1,
                chess.KNIGHT: 3,
                chess.BISHOP: 3,
                chess.ROOK: 5,
                chess.QUEEN: 9,
                chess.KING: 0
            }
            
            # MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
            if captured_piece and moving_piece:
                score += piece_values.get(captured_piece.piece_type, 0) * 10
                score -= piece_values.get(moving_piece.piece_type, 0)
        
        # Prioritize checks
        board.push(move)
        if board.is_check():
            score += 50
        board.pop()
        
        # Prioritize central moves
        if move.to_square in [chess.E4, chess.E5, chess.D4, chess.D5]:
            score += 20
        
        # Prioritize piece development (off back rank)
        if board.piece_at(move.from_square):
            piece = board.piece_at(move.from_square)
            if piece.color == chess.BLACK:
                if chess.square_rank(move.from_square) == 7:  # Moving from back rank
                    score += 15
        
        move_scores.append((score, move))
    
    # Sort by score descending (highest score first)
    move_scores.sort(key=lambda x: x[0], reverse=True)
    return [move for score, move in move_scores]

# create Engine class
class Engine:
    def __init__(self, fen):
        self.board = chess.Board()
        self.board.set_fen(fen)
        self.position_cache = {}

    # is called in flask app
    def get_move(self):
        best_move = self.make_move()
        return str(best_move)

    # get best move for black
    def make_move(self):
        current_fen_x = self.board.fen()
        black_response = {}
        print ("calculating\n")
        
        # Order moves for better search
        black_legal_moves = order_moves(self.board, list(self.board.legal_moves))
        
        for black_move in black_legal_moves:
            print (".")
            white_response = {}
            self.board.push(black_move)
            if self.board.is_checkmate():
                self.board.pop()
                return str(black_move)
            
            # Order white moves too
            white_legal_moves = order_moves(self.board, list(self.board.legal_moves))
            
            for white_move in white_legal_moves:
                self.board.push(white_move)
                if self.board.is_checkmate():
                    white_response[white_move] = 0
                    self.board.pop()
                    break
                
                black_legal_moves_depth_2 = order_moves(self.board, list(self.board.legal_moves))
                black_response_depth_2 = {}
                
                for black_move_depth_2 in black_legal_moves_depth_2:
                    self.board.push(black_move_depth_2)
                    if self.board.is_checkmate():
                        black_response_depth_2[black_move_depth_2] = 1
                        self.board.pop()
                        break
                    
                    next_fen_x = self.board.fen()
                    if next_fen_x in self.position_cache:
                        prediction_number = self.position_cache[next_fen_x]
                    else:
                        prediction_number = evaluate_pos(current_fen_x, next_fen_x)
                        self.position_cache[next_fen_x] = prediction_number
                    
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