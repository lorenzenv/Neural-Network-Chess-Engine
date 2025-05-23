import chess
import numpy as np
from util import make_bitboard, beautifyFEN # Assuming these are in util.py

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite
import time

ENGINE_VERSION = "1.7-AlphaBetaFixedDepthNNRootCompare"
ENGINE_NAME = "AlphaBeta Fixed Depth NN Root Comparison"
ENGINE_FEATURES = [
    "🧠 NN Direct Evaluation (Future Pos vs Root Pos)",
    "♟️ Alpha-Beta Fixed-Depth Search (e.g., 3-ply)",
    "🚫 No complex TT or accumulated scores",
]

# --- Configuration ---
FIXED_SEARCH_PLY = 4 # How many half-moves (ply) to look ahead. Increased to show AB benefit.
# --- End Configuration ---


interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def make_x(first_fen, second_fen):
    x_1 = make_bitboard(beautifyFEN(first_fen))
    x_2 = make_bitboard(beautifyFEN(second_fen))
    x_1 = np.array(x_1, dtype=np.float32).reshape(1, 769)
    x_2 = np.array(x_2, dtype=np.float32).reshape(1, 769)
    return x_1, x_2

def evaluate_nn_comparative(fen_to_evaluate, fen_reference):
    x_1, x_2 = make_x(fen_to_evaluate, fen_reference)
    interpreter.set_tensor(input_details[0]['index'], x_1)
    interpreter.set_tensor(input_details[1]['index'], x_2)
    interpreter.invoke()
    raw_evaluation = interpreter.get_tensor(output_details[0]['index'])[0][0]
    return raw_evaluation

class Engine:
    def __init__(self, fen):
        self.board = chess.Board()
        self.board.set_fen(fen)
        self.name = ENGINE_NAME
        self.version = ENGINE_VERSION
        self.nodes_evaluated_this_turn = 0
        self.ab_prunes = 0

    def get_version_info(self):
        return {
            "version": self.version,
            "name": self.name,
            "features": ENGINE_FEATURES
        }

    def get_move(self):
        self.nodes_evaluated_this_turn = 0
        self.ab_prunes = 0
        start_time = time.time()

        best_move_uci = self.find_best_move_fixed_depth_ab()
        
        duration = time.time() - start_time
        print(f"Search for best move took {duration:.2f}s. Nodes evaluated: {self.nodes_evaluated_this_turn}. AB Prunes: {self.ab_prunes}")

        if best_move_uci is None:
            if self.board.is_checkmate(): return "checkmate"
            if self.board.is_stalemate() or self.board.is_insufficient_material() or \
               self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition():
                return "draw"
            return "no_legal_moves_or_error"
        return best_move_uci

    def find_best_move_fixed_depth_ab(self):
        root_fen = self.board.fen()
        legal_moves = list(self.board.legal_moves)

        if not legal_moves:
            return None

        best_move_for_root_player = None
        
        is_white_at_root = (self.board.turn == chess.WHITE)

        alpha = float('-inf')
        beta = float('inf')

        if is_white_at_root:
            best_score_seen_by_root = float('-inf') 
        else:
            best_score_seen_by_root = float('inf')

        # Simple move ordering: try captures first, then checks, then others.
        # This can improve AB pruning effectiveness.
        # For "pure NN", we might skip this, but it's a search enhancement not an eval heuristic.
        # For now, let's keep it simple and not add explicit move ordering here.

        for move in legal_moves:
            self.board.push(move)
            score_for_this_path = self.minimax_ab_search(FIXED_SEARCH_PLY - 1, root_fen, not is_white_at_root, alpha, beta)
            self.board.pop()

            if is_white_at_root:
                if score_for_this_path > best_score_seen_by_root:
                    best_score_seen_by_root = score_for_this_path
                    best_move_for_root_player = move
                alpha = max(alpha, best_score_seen_by_root) 
            else: # Black at root
                if score_for_this_path < best_score_seen_by_root:
                    best_score_seen_by_root = score_for_this_path
                    best_move_for_root_player = move
                beta = min(beta, best_score_seen_by_root)
            
            # No pruning at root itself, but alpha/beta are updated for subsequent calls if we had siblings at root
            # (which we don't in this setup, each top-level move is independent to start).

        if best_move_for_root_player:
            print(f"Selected best move: {best_move_for_root_player.uci()} with AB score (vs root, White's P.O.V.): {best_score_seen_by_root:.4f}")
            return str(best_move_for_root_player.uci())
        elif legal_moves:
            print("Warning: No distinct best move from AB search, picking first legal move.")
            return str(legal_moves[0].uci())
        return None


    def minimax_ab_search(self, ply_remaining, original_root_fen, maximizing_player_is_white, alpha, beta):
        self.nodes_evaluated_this_turn +=1 # Counts internal nodes too, not just leaf evaluations.

        if self.board.is_checkmate():
            # self.nodes_evaluated_this_turn +=1 # Only count leaf evaluations if preferred
            if not maximizing_player_is_white: 
                return 10000.0 # White delivered mate (score from White's POV)
            else: 
                return -10000.0 # Black delivered mate
        
        if self.board.is_stalemate() or self.board.is_insufficient_material() or \
           self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition() or \
           self.board.can_claim_threefold_repetition():
            # self.nodes_evaluated_this_turn +=1
            current_fen = self.board.fen()
            return evaluate_nn_comparative(current_fen, original_root_fen) # Or simply 0.0

        if ply_remaining == 0:
            # self.nodes_evaluated_this_turn +=1
            current_fen = self.board.fen()
            return evaluate_nn_comparative(current_fen, original_root_fen)

        legal_moves = list(self.board.legal_moves)
        if not legal_moves: # Should be caught by checkmate/stalemate
            # self.nodes_evaluated_this_turn +=1
            current_fen = self.board.fen() 
            return evaluate_nn_comparative(current_fen, original_root_fen)

        # Simple move ordering can help here too, but keeping it minimal for now.

        if maximizing_player_is_white:
            max_eval = float('-inf')
            for move in legal_moves:
                self.board.push(move)
                eval_score = self.minimax_ab_search(ply_remaining - 1, original_root_fen, False, alpha, beta)
                self.board.pop()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    self.ab_prunes +=1
                    break 
            return max_eval
        else: # Minimizing player (Black seeks to minimize White's score)
            min_eval = float('inf')
            for move in legal_moves:
                self.board.push(move)
                eval_score = self.minimax_ab_search(ply_remaining - 1, original_root_fen, True, alpha, beta)
                self.board.pop()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    self.ab_prunes += 1
                    break
            return min_eval