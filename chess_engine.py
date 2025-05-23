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

CHECKMATE_POSITIVE_SCORE = 9999
CHECKMATE_NEGATIVE_SCORE = -9999
DEFAULT_SEARCH_DEPTH = 3

class Engine:
    def __init__(self, fen):
        self.board = chess.Board()
        self.board.set_fen(fen)
        self.search_depth = DEFAULT_SEARCH_DEPTH
        # For future transposition table:
        # self.transposition_table = {} 

    def get_move(self):
        # Helper to get the best move as a string
        # The flask app passes a 'depth' parameter but it's not used by make_move yet.
        # We can adapt this later if we want dynamic depth from the API.
        best_move = self.make_move_alpha_beta()
        if best_move:
            return str(best_move)
        # Fallback if no move found (should not happen in normal play)
        if list(self.board.legal_moves):
            return str(list(self.board.legal_moves)[0])
        return "" # Should indicate game over or error

    def make_move_alpha_beta(self):
        original_fen_for_eval = self.board.fen()
        legal_moves = list(self.board.legal_moves)

        if not legal_moves:
            return None # No legal moves

        best_move_found = None
        
        # Alpha and Beta initialized for the root node
        alpha = float('-inf')
        beta = float('inf')

        if self.board.turn == chess.WHITE: # White is the maximizing player for the evaluate_pos score
            max_eval = float('-inf')
            for move in legal_moves:
                self.board.push(move)
                # Next player (Black) is minimizing. Depth is current_depth - 1.
                eval_score = self.alpha_beta_search(self.board, self.search_depth - 1, alpha, beta, False, original_fen_for_eval)
                self.board.pop()
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move_found = move
                alpha = max(alpha, eval_score)
                # No beta cutoff at root for all moves, but alpha helps ordering for future.
            return best_move_found
        else: # Black is the minimizing player for the evaluate_pos score
            min_eval = float('inf')
            for move in legal_moves:
                self.board.push(move)
                # Next player (White) is maximizing. Depth is current_depth - 1.
                eval_score = self.alpha_beta_search(self.board, self.search_depth - 1, alpha, beta, True, original_fen_for_eval)
                self.board.pop()

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move_found = move
                beta = min(beta, eval_score) 
                # No alpha cutoff at root for all moves, but beta helps ordering for future.
            return best_move_found

    def alpha_beta_search(self, current_board, depth, alpha, beta, is_maximizing_player_for_eval, parent_fen_for_eval):
        if current_board.is_game_over():
            if current_board.is_checkmate():
                # current_board.turn is the player who IS mated.
                if current_board.turn == chess.WHITE: # White is mated (Black delivered mate)
                    return CHECKMATE_NEGATIVE_SCORE 
                else: # Black is mated (White delivered mate)
                    return CHECKMATE_POSITIVE_SCORE
            # For other game over conditions (stalemate, etc.), let the model evaluate.
            # The evaluate_pos function compares parent_fen_for_eval to current_board.fen()
            # This will give a score from White's perspective.
            return evaluate_pos(parent_fen_for_eval, current_board.fen())

        if depth == 0:
            return evaluate_pos(parent_fen_for_eval, current_board.fen())

        legal_moves_ab = list(current_board.legal_moves)
        if not legal_moves_ab: # Should be caught by is_game_over, but as a safeguard
             return evaluate_pos(parent_fen_for_eval, current_board.fen())

        fen_of_board_at_this_level = current_board.fen()

        if is_maximizing_player_for_eval: # White's turn in search tree (wants to maximize White's score)
            current_max_eval = float('-inf')
            for move in legal_moves_ab:
                current_board.push(move)
                # Next player (Black) is minimizing. Depth is current_depth - 1.
                eval_score = self.alpha_beta_search(current_board, depth - 1, alpha, beta, False, fen_of_board_at_this_level)
                current_board.pop()
                current_max_eval = max(current_max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break # Beta cut-off
            return current_max_eval
        else: # Black's turn in search tree (wants to minimize White's score)
            current_min_eval = float('inf')
            for move in legal_moves_ab:
                current_board.push(move)
                # Next player (White) is maximizing. Depth is current_depth - 1.
                eval_score = self.alpha_beta_search(current_board, depth - 1, alpha, beta, True, fen_of_board_at_this_level)
                current_board.pop()
                current_min_eval = min(current_min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break # Alpha cut-off
            return current_min_eval