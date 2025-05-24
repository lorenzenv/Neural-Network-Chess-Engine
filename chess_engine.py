import chess
import numpy as np
import functools
import time
import threading
from util import * # Changed from specific imports to wildcard

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

from util import *  # or your specific imports

# global model loading with thread safety
interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Thread lock for interpreter access
interpreter_lock = threading.Lock()

# ------- Engine Metadata -------
ENGINE_NAME = "valibot"
ENGINE_VERSION = "2.2.0"
ENGINE_FEATURES = ["Corrected Neural Network Evaluation", "Proper Bitboard Conversion", "Hybrid NN-Classical Evaluation", "Alpha-Beta Search"]

# ------- Configuration -------
class Config:
    # Speed Mode: "fast", "balanced", "strong"
    SPEED_MODE = "fast"  # Default to fast for Lichess play
    
    NN_SCALING_FACTOR = 1000.0
    MAX_PLY_FOR_KILLERS = 30
    QUIESCENCE_MAX_DEPTH_RELATIVE = 4  # Increased from 3 for better tactical vision
    LMR_MIN_MOVES_TRIED = 3 # Number of full-depth/PV moves to try before LMR can activate
    LMR_REDUCTION = 1       # Depth reduction for LMR moves
    NMP_REDUCTION = 3
    TT_SIZE_POWER = 20 # Reduced from 22 for memory efficiency
    FEN_CACHE_SIZE = 2048  # Reduced from 4096 for faster lookups
    
    # Speed-dependent settings - Improved for better performance
    if SPEED_MODE == "fast":
        MAX_SEARCH_DEPTH_ID = 8  # Increased from 6 for better tactical play
        ITERATIVE_DEEPENING_TIME_LIMIT_PER_MOVE = 4.0  # Slightly increased for depth
        NN_EVALUATION_BLEND = 0.8  # More balanced approach
        SEARCH_TIME_EXTENSION_FACTOR = 1.5  # Allow extension for critical positions
    elif SPEED_MODE == "balanced":
        MAX_SEARCH_DEPTH_ID = 10
        ITERATIVE_DEEPENING_TIME_LIMIT_PER_MOVE = 8.0
        NN_EVALUATION_BLEND = 0.75
        SEARCH_TIME_EXTENSION_FACTOR = 2.0
    else:  # "strong"
        MAX_SEARCH_DEPTH_ID = 12
        ITERATIVE_DEEPENING_TIME_LIMIT_PER_MOVE = 20.0
        NN_EVALUATION_BLEND = 0.7
        SEARCH_TIME_EXTENSION_FACTOR = 3.0

# ------- Piece values for MVV-LVA -------
PIECE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 320,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000
}

# ------- Zobrist Hashing -------
class ZobristHash:
    def __init__(self):
        # np.random.seed(42) # For deterministic hashes during testing
        self.piece_keys = np.random.randint(0, np.iinfo(np.uint64).max, size=(12, 64), dtype=np.uint64) # 12 piece types, 64 squares
        self.black_to_move_key = np.random.randint(0, np.iinfo(np.uint64).max, dtype=np.uint64)
        self.castling_keys = np.random.randint(0, np.iinfo(np.uint64).max, size=16, dtype=np.uint64) # For 2^4 castling rights combinations
        self.ep_keys = np.random.randint(0, np.iinfo(np.uint64).max, size=64, dtype=np.uint64) # For EP square

    def get_piece_id(self, piece: chess.Piece) -> int:
        # P=0, N=1, B=2, R=3, Q=4, K=5 (for white)
        # P=6, N=7, B=8, R=9, Q=10, K=11 (for black)
        piece_id = piece.piece_type - 1 # 0-5
        if piece.color == chess.BLACK:
            piece_id += 6
        return piece_id

    def hash(self, board: chess.Board) -> np.uint64:
        h = np.uint64(0)
        for sq_idx in range(64):
            piece = board.piece_at(sq_idx)
            if piece:
                h ^= self.piece_keys[self.get_piece_id(piece), sq_idx]
        
        if board.turn == chess.BLACK:
            h ^= self.black_to_move_key
        
        h ^= self.castling_keys[board.castling_rights & 15] # Use lower 4 bits for castling rights

        if board.ep_square is not None:
            h ^= self.ep_keys[board.ep_square]
            
        return h

# ------- Fixed-size Transposition Table -------
class FixedTT:
    def __init__(self):
        size = 1 << Config.TT_SIZE_POWER
        self.mask = np.uint64(size - 1)
        self.keys = np.zeros(size, dtype=np.uint64)
        self.entries = [None] * size 
    
    def index(self, key: np.uint64) -> int:
        return int(key & self.mask)

    def store(self, key: np.uint64, entry: tuple): # depth, score_wpov, best_move_uci, flag
        i = self.index(key)
        # Simple always-replace strategy. Could add generation/depth preference.
        self.keys[i] = key 
        self.entries[i] = entry

    def lookup(self, key: np.uint64):
        i = self.index(key)
        if self.keys[i] == key:
            return self.entries[i]
        return None

# ------- Cached FEN to Bitboard Conversion -------
@functools.lru_cache(maxsize=Config.FEN_CACHE_SIZE)
def make_x_cached(fen_1: str, fen_2: str): # Changed param names for clarity
    b1 = bitifyFEN(beautifyFEN(fen_1))
    b2 = bitifyFEN(beautifyFEN(fen_2))
    return (
        np.array(b1, dtype=np.float32).reshape(1, 769),
        np.array(b2, dtype=np.float32).reshape(1, 769)
    )

# ------- NN Evaluator -------
class NNEvaluator:
    def __init__(self):
        # Interpreter is now global, no need to initialize here
        pass

    def compare_positions(self, fen1: str, fen2: str) -> float:
        """Compare two positions. Returns value between 0 and 1.
        1.0 means fen1 is more winning for white than fen2
        0.0 means fen2 is more winning for white than fen1
        """
        try:
            x1_np, x2_np = make_x_cached(fen1, fen2)
            x1_copy = np.copy(x1_np)
            x2_copy = np.copy(x2_np)
            
            with interpreter_lock:
                interpreter.set_tensor(input_details[0]['index'], x1_copy) 
                interpreter.set_tensor(input_details[1]['index'], x2_copy)
                interpreter.invoke()
                raw_comparison = float(interpreter.get_tensor(output_details[0]['index'])[0][0])
            
            return raw_comparison
            
        except Exception as e:
            print(f"NN Comparison error: {e}")
            return 0.5  # Neutral if NN fails

    def compare_positions_ensemble(self, fen1: str, fen2: str) -> float:
        """
        Improved comparison using ensemble method to reduce color bias.
        Compares both orders and takes the average to ensure symmetry.
        """
        try:
            # Get comparison in both directions
            forward = self.compare_positions(fen1, fen2)
            backward = self.compare_positions(fen2, fen1)
            
            # These should be symmetric: backward ≈ 1.0 - forward
            # Take the average to reduce any systematic bias
            expected_backward = 1.0 - forward
            symmetry_error = abs(backward - expected_backward)
            
            # If symmetry error is large, there might be bias - use more conservative approach
            if symmetry_error > 0.1:
                # Use the average of both interpretations for more stability
                ensemble_result = (forward + (1.0 - backward)) / 2.0
            else:
                # Use the forward comparison if symmetry is good
                ensemble_result = forward
            
            return ensemble_result
            
        except Exception as e:
            print(f"NN Ensemble comparison error: {e}")
            return 0.5

    def evaluate_move_comparison(self, current_fen: str, move_fens: list, player_is_white: bool) -> list:
        """
        Compare multiple move positions using the neural network.
        Returns a list of comparison scores for each move FEN relative to current position.
        Higher scores mean better for the current player.
        IMPROVED: Uses ensemble method to reduce color bias.
        """
        if not move_fens:
            return []
        
        scores = []
        for move_fen in move_fens:
            try:
                # Use ensemble comparison for better stability
                comparison = self.compare_positions_ensemble(current_fen, move_fen)
                
                # More symmetric interpretation for both colors
                if player_is_white:
                    # For White: if move_result > current, that's good (high comparison value)
                    # We want high scores for good moves, so don't invert
                    raw_score = comparison
                else:
                    # For Black: if move_result < current for White perspective, that's good for Black
                    # So we want low comparison values to translate to high scores
                    raw_score = 1.0 - comparison
                
                # Apply confidence scaling - moves near 0.5 are uncertain
                confidence = abs(comparison - 0.5) * 2.0  # 0 to 1 scale
                
                # Enhanced confidence adjustment for color balance
                if confidence < 0.2:
                    # Very uncertain comparison - use neutral score
                    score = 0.5
                elif confidence < 0.4:
                    # Somewhat uncertain - blend with neutral
                    score = 0.5 * 0.3 + raw_score * 0.7
                else:
                    # Confident comparison - use raw score
                    score = raw_score
                
                scores.append(score)
                
            except Exception as e:
                print(f"NN Move comparison error: {e}")
                scores.append(0.5)  # Neutral if NN fails
        
        return scores

    def evaluate_absolute_score_white_pov(self, fen_to_evaluate: str, fen_context_for_pov: str) -> float:
        """
        Blunder-safe evaluation method. Uses classical evaluation as primary with NN adjustments.
        IMPROVED: Uses ensemble NN comparison for better color balance.
        """
        try:
            # Always start with classical evaluation as the foundation
            classic_cp = classical_material_eval(fen_to_evaluate)
            
            # Check for obviously bad positions (large material disadvantage)
            board = chess.Board(fen_to_evaluate)
            if board.is_checkmate():
                return -10000.0 if board.turn == chess.WHITE else 10000.0
            
            if board.is_stalemate() or board.is_insufficient_material():
                return 0.0
            
            # Use NN comparison conservatively only if we have meaningful context
            if fen_context_for_pov and fen_context_for_pov != fen_to_evaluate:
                # Use ensemble comparison for better color balance
                comparison = self.compare_positions_ensemble(fen_to_evaluate, fen_context_for_pov)
                
                # Only apply NN adjustment if the comparison is confident (not close to 0.5)
                confidence = abs(comparison - 0.5) * 2.0
                if confidence > 0.2:  # Only use NN if reasonably confident
                    # Convert comparison to small adjustment factor (-100 to +100 cp)
                    nn_adjustment = (comparison - 0.5) * 200 * confidence
                    
                    # Apply conservative adjustment - never more than 20% of classical eval
                    max_adjustment = abs(classic_cp) * 0.2 if classic_cp != 0 else 50
                    nn_adjustment = max(-max_adjustment, min(max_adjustment, nn_adjustment))
                    
                    final_score = classic_cp + nn_adjustment
                else:
                    final_score = classic_cp
            else:
                final_score = classic_cp
            
            # Stricter bounds to prevent extreme evaluations
            return max(-1500, min(1500, final_score))
            
        except Exception as e:
            print(f"NN Evaluation error: {e}")
            # Fallback to classical evaluation
            return classical_material_eval(fen_to_evaluate)

# ------- Engine Implementation -------
class Engine:
    def __init__(self, fen: str, model_path: str = "model.tflite"):
        self.board = chess.Board(fen)
        self.zobrist_hasher = ZobristHash()
        self.tt = FixedTT()
        self.nn_evaluator = NNEvaluator() # Changed: No longer passes model_path

        self.start_time_for_move = None
        self.time_limit_for_move = Config.ITERATIVE_DEEPENING_TIME_LIMIT_PER_MOVE

        self.nodes_searched = 0; self.q_nodes_searched = 0; self.tt_hits = 0
        self.beta_cutoffs = 0; self.nmp_cutoffs = 0; self.lmr_activations = 0
        self.killer_moves = [[None, None] for _ in range(Config.MAX_PLY_FOR_KILLERS)]

    def get_version_info(self) -> dict:
        return {"name": ENGINE_NAME, "version": ENGINE_VERSION, "features": ENGINE_FEATURES}

    def time_is_up(self) -> bool:
        if self.start_time_for_move is None: return False
        elapsed = time.time() - self.start_time_for_move
        
        # Allow extra time for critical positions (checks, captures, tactics)
        time_limit = self.time_limit_for_move
        if self.board.is_check() or len(list(self.board.legal_moves)) <= 3:
            time_limit *= Config.SEARCH_TIME_EXTENSION_FACTOR
        
        return elapsed >= time_limit

    def get_move(self) -> str:
        best_move_uci, _ = self.iterative_deepening_search()
        if best_move_uci is None:
            if self.board.is_checkmate(): return "checkmate"
            if self.board.is_stalemate() or self.board.is_insufficient_material() or \
               self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition():
                return "draw"
            legal_moves = list(self.board.legal_moves)
            if legal_moves: return legal_moves[0].uci()
            return "no_legal_moves_or_error"
        return best_move_uci

    def order_moves(self, legal_moves, ply_count, tt_best_move_uci):
        move_scores = []
        tt_move_obj = None
        if tt_best_move_uci:
            try:
                tt_move_obj = chess.Move.from_uci(tt_best_move_uci)
                if tt_move_obj not in legal_moves: tt_move_obj = None
            except ValueError: tt_move_obj = None

        # Safer approach: Only use NN ordering for very simple cases
        use_nn_ordering = False  # Disabled for now to prevent illegal moves
        nn_scores = []
        
        # Commented out NN ordering until we can fix the illegal move issue
        # if use_nn_ordering:
        #     try:
        #         current_fen = self.board.fen()
        #         move_fens = []
        #         
        #         # Generate FENs for each move
        #         for move in legal_moves:
        #             self.board.push(move)
        #             move_fens.append(self.board.fen())
        #             self.board.pop()
        #         
        #         # Get NN comparison scores
        #         nn_scores = self.nn_evaluator.evaluate_move_comparison(
        #             current_fen, move_fens, self.board.turn == chess.WHITE
        #         )
        #     except Exception as e:
        #         print(f"NN ordering error: {e}")
        #         use_nn_ordering = False
        #         nn_scores = []

        for i, move in enumerate(legal_moves):
            score = 0
            
            # Highest priority: TT move
            if move == tt_move_obj: 
                score = 10000000
            # Second priority: Captures (MVV-LVA)
            elif self.board.is_capture(move):
                score = 2000000 
                victim_type = chess.PAWN 
                if self.board.is_en_passant(move): 
                    victim_type = chess.PAWN
                else:
                    victim_piece_obj = self.board.piece_at(move.to_square)
                    if victim_piece_obj: 
                        victim_type = victim_piece_obj.piece_type
                attacker_piece_obj = self.board.piece_at(move.from_square)
                attacker_type = attacker_piece_obj.piece_type if attacker_piece_obj else chess.PAWN
                score += (PIECE_VALUES.get(victim_type, 0) * 100) - PIECE_VALUES.get(attacker_type, 100)
            # Third priority: Promotions
            elif move.promotion == chess.QUEEN: 
                score = 1500000
            elif move.promotion: 
                score = PIECE_VALUES.get(move.promotion, 300) * 1000
            # Fourth priority: Killer moves
            elif ply_count < Config.MAX_PLY_FOR_KILLERS and move in self.killer_moves[ply_count]:
                score = 1000000 if move == self.killer_moves[ply_count][0] else 900000
            # Fifth priority: Checks and castling
            elif self.board.gives_check(move): 
                score = 500000
            elif self.board.is_castling(move): 
                score = 300000
            # Central moves and piece development (simple heuristics)
            else:
                from_square = move.from_square
                to_square = move.to_square
                
                # Bonus for central squares
                center_bonus = 0
                center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
                if to_square in center_squares:
                    center_bonus = 5000
                
                # Bonus for piece development 
                piece = self.board.piece_at(from_square)
                development_bonus = 0
                if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                    if piece.color == chess.WHITE and from_square in [chess.B1, chess.C1, chess.F1, chess.G1]:
                        development_bonus = 3000
                    elif piece.color == chess.BLACK and from_square in [chess.B8, chess.C8, chess.F8, chess.G8]:
                        development_bonus = 3000
                
                score = 100000 + center_bonus + development_bonus
                
            move_scores.append((score, move))
        
        move_scores.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in move_scores]

    def iterative_deepening_search(self):
        self.start_time_for_move = time.time()
        self.nodes_searched = 0; self.q_nodes_searched = 0; self.tt_hits = 0
        self.beta_cutoffs = 0; self.nmp_cutoffs = 0; self.lmr_activations = 0
        
        if len(self.tt.entries) > (1 << (Config.TT_SIZE_POWER - 2)):
             self.tt = FixedTT()
             self.killer_moves = [[None, None] for _ in range(Config.MAX_PLY_FOR_KILLERS)]

        best_move_overall_uci = None; best_score_overall_white_pov = -float('inf')
        initial_fen_at_root = self.board.fen()

        for depth in range(1, Config.MAX_SEARCH_DEPTH_ID + 1):
            if self.time_is_up() and depth > 1 :
                print(f"Time up before Depth {depth}. Using Depth {depth-1} result.")
                break
            
            search_start_time_this_depth = time.time()
            try:
                # Call alpha_beta with initial_fen_at_root as the context for NN evals, and is_pv_node=True for root
                score_from_root_player_pov = self.alpha_beta(depth, -float('inf'), float('inf'),
                                                                initial_fen_at_root, True, 0) # is_pv_node=True, ply_count = 0
            except TimeoutError:
                print(f"Search for Depth {depth} timed out.")
                tt_entry_root = self.tt.lookup(self.zobrist_hasher.hash(self.board))
                if tt_entry_root and tt_entry_root[2]:
                    best_move_overall_uci = tt_entry_root[2]
                break 

            search_duration_this_depth = time.time() - search_start_time_this_depth
            
            tt_entry_root = self.tt.lookup(self.zobrist_hasher.hash(self.board))
            current_best_move_uci_from_tt = None
            if tt_entry_root and tt_entry_root[2]:
                current_best_move_uci_from_tt = tt_entry_root[2]
            
            score_white_pov = score_from_root_player_pov if self.board.turn == chess.WHITE else -score_from_root_player_pov

            if current_best_move_uci_from_tt:
                best_move_overall_uci = current_best_move_uci_from_tt
                best_score_overall_white_pov = score_white_pov
                print(f"Depth {depth}: PV={best_move_overall_uci} Score(WPOV)={score_white_pov:.2f} Nodes={self.nodes_searched} (Q:{self.q_nodes_searched}) TTHits={self.tt_hits} BetaCuts={self.beta_cutoffs} NMPCuts={self.nmp_cutoffs} LMR_Acts={self.lmr_activations} Time={search_duration_this_depth:.2f}s")
            else: # Should be rare if search completes
                print(f"Depth {depth}: No PV from TT for root. Score(WPOV)={score_white_pov:.2f} Nodes={self.nodes_searched} ... Time={search_duration_this_depth:.2f}s")

            if abs(score_white_pov) > 90000: print("Mate score detected..."); break # Check WPOV score
        
        if best_move_overall_uci is None:
             legal_moves = list(self.board.legal_moves)
             if legal_moves:
                 best_move_overall_uci = legal_moves[0].uci()
                 print("Warning: Fallback to first legal move at end of ID.")
        
        return best_move_overall_uci, best_score_overall_white_pov

    def alpha_beta(self, depth_remaining: int, alpha: float, beta: float, 
                     root_fen_for_nn_context: str, is_pv_node: bool, ply_count: int) -> float: # Returns score from current player's POV
        
        if self.time_is_up() and ply_count > 0 : raise TimeoutError("Search time limit exceeded")

        self.nodes_searched += 1
        alpha_original = alpha
        current_node_zobrist_key = self.zobrist_hasher.hash(self.board)

        if ply_count > 0 and (self.board.can_claim_threefold_repetition() or self.board.is_fifty_moves()):
            return 0.0 

        # Re-enable transposition table
        tt_entry = self.tt.lookup(current_node_zobrist_key)
        tt_best_move_uci = None
        if tt_entry:
            tt_depth, tt_score_white_pov, tt_best_move_uci, tt_flag = tt_entry
            if tt_depth >= depth_remaining:
                self.tt_hits += 1
                tt_score_current_player_pov = tt_score_white_pov if self.board.turn == chess.WHITE else -tt_score_white_pov
                if tt_flag == "EXACT": return tt_score_current_player_pov
                if tt_flag == "LOWERBOUND": alpha = max(alpha, tt_score_current_player_pov)
                elif tt_flag == "UPPERBOUND": beta = min(beta, tt_score_current_player_pov)
                if alpha >= beta: return tt_score_current_player_pov

        if self.board.is_checkmate(): return -100000.0 + ply_count 
        if self.board.is_stalemate() or self.board.is_insufficient_material(): return 0.0

        if depth_remaining <= 0:
            return self.quiescence(alpha, beta, root_fen_for_nn_context, ply_count)

        can_do_nmp = depth_remaining >= (Config.NMP_REDUCTION + 1) and not self.board.is_check() and ply_count > 0
        if can_do_nmp:
            non_pawn_material = sum(len(self.board.pieces(pt, self.board.turn)) for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
            if non_pawn_material < 2 : can_do_nmp = False
        
        if can_do_nmp:
            self.board.push(chess.Move.null())
            score = -self.alpha_beta(depth_remaining - 1 - Config.NMP_REDUCTION, -beta, -alpha, 
                                     root_fen_for_nn_context, False, ply_count + 1) # NMP is not a PV node
            self.board.pop()
            if score >= beta: 
                self.nmp_cutoffs +=1
                score_to_store_white_pov = score if self.board.turn == chess.WHITE else -score
                self.tt.store(current_node_zobrist_key, (depth_remaining, score_to_store_white_pov, None, "LOWERBOUND"))
                return beta 
        
        legal_moves = list(self.board.legal_moves)
        if not legal_moves: return 0.0 
        ordered_moves = self.order_moves(legal_moves, ply_count, tt_best_move_uci)
        
        best_move_found_uci_for_tt = None
        value_from_current_player_pov = -float('inf') 

        for i, move in enumerate(ordered_moves):
            self.board.push(move)
            
            is_giving_check = self.board.is_check() 
            # Simplified extension logic - only extend checks
            extension = 1 if is_giving_check and depth_remaining >= 2 else 0
                
            child_search_depth = depth_remaining - 1 + extension
            
            current_move_score_from_child_pov = 0 

            if i == 0 : # First move, always full window search, PV status depends on parent
                current_move_score_from_child_pov = -self.alpha_beta(child_search_depth, -beta, -alpha,
                                                                root_fen_for_nn_context, 
                                                                is_pv_node, # Child is PV only if parent is PV and it's the first move
                                                                ply_count + 1)
            else: # Subsequent moves, PVS / LMR logic
                # Conservative LMR to avoid missing tactics
                can_reduce = (
                    child_search_depth >= Config.LMR_REDUCTION and 
                    i >= Config.LMR_MIN_MOVES_TRIED and 
                    not extension and 
                    not self.board.is_capture(move) and 
                    not move.promotion and
                    ply_count >= 3  # Don't reduce near root
                )
                
                if can_reduce:
                    self.lmr_activations += 1
                    current_move_score_from_child_pov = -self.alpha_beta(child_search_depth - Config.LMR_REDUCTION, 
                                                                    -alpha -1, -alpha, 
                                                                    root_fen_for_nn_context, False, ply_count + 1) # LMR is not PV
                else: # If LMR not applicable, do a null-window search first
                     current_move_score_from_child_pov = -self.alpha_beta(child_search_depth, 
                                                                    -alpha -1, -alpha, 
                                                                    root_fen_for_nn_context, False, ply_count + 1) # Null-window search is not PV

                # If null-window search failed high, re-search with full window (PVS)
                if current_move_score_from_child_pov > alpha and current_move_score_from_child_pov < beta : 
                    current_move_score_from_child_pov = -self.alpha_beta(child_search_depth, -beta, -alpha,
                                                                    root_fen_for_nn_context, is_pv_node, ply_count + 1)
            self.board.pop()

            if current_move_score_from_child_pov > value_from_current_player_pov:
                value_from_current_player_pov = current_move_score_from_child_pov
                best_move_found_uci_for_tt = move.uci()
            
            alpha = max(alpha, value_from_current_player_pov)
            if alpha >= beta:
                self.beta_cutoffs +=1
                if not self.board.is_capture(move) and ply_count < Config.MAX_PLY_FOR_KILLERS:
                    km = self.killer_moves[ply_count]
                    if move != km[0]: km[1] = km[0]; km[0] = move
                break 
        
        final_score_white_pov = value_from_current_player_pov if self.board.turn == chess.WHITE else -value_from_current_player_pov
        tt_flag = "EXACT"
        if value_from_current_player_pov <= alpha_original: tt_flag = "UPPERBOUND"
        elif value_from_current_player_pov >= beta: tt_flag = "LOWERBOUND"
        
        self.tt.store(current_node_zobrist_key, (depth_remaining, final_score_white_pov, best_move_found_uci_for_tt, tt_flag))
        return value_from_current_player_pov

    def quiescence(self, alpha: float, beta: float, 
                    root_fen_for_nn_context: str, ply_count: int) -> float: # Returns score from current player's POV
        self.q_nodes_searched += 1
        
        if self.time_is_up() and ply_count > 0: raise TimeoutError("Search time limit exceeded")

        if self.board.can_claim_threefold_repetition() or self.board.is_fifty_moves(): return 0.0
        
        stand_pat_score_white_pov = self.nn_evaluator.evaluate_absolute_score_white_pov(self.board.fen(), root_fen_for_nn_context)
        
        stand_pat_current_player_pov = stand_pat_score_white_pov if self.board.turn == chess.WHITE else -stand_pat_score_white_pov

        if stand_pat_current_player_pov >= beta: return beta
        alpha = max(alpha, stand_pat_current_player_pov)

        q_moves_with_scores = []
        for move in self.board.legal_moves:
            score = 0
            if self.board.is_capture(move):
                score = 2000000 
                victim_type = chess.PAWN
                if self.board.is_en_passant(move): victim_type = chess.PAWN
                else:
                    victim_piece_obj = self.board.piece_at(move.to_square)
                    if victim_piece_obj: victim_type = victim_piece_obj.piece_type
                attacker_piece_obj = self.board.piece_at(move.from_square)
                attacker_type = attacker_piece_obj.piece_type if attacker_piece_obj else chess.PAWN
                score += (PIECE_VALUES.get(victim_type, 0) * 100) - PIECE_VALUES.get(attacker_type, 100)
            elif move.promotion:
                score = 1500000 + (PIECE_VALUES.get(move.promotion, 0) * 1000)
            
            if score > 0 : q_moves_with_scores.append((score, move))
        
        q_moves_with_scores.sort(key=lambda item: item[0], reverse=True)
        ordered_q_moves = [item[1] for item in q_moves_with_scores]

        if not ordered_q_moves or ply_count >= Config.MAX_PLY_FOR_KILLERS + Config.QUIESCENCE_MAX_DEPTH_RELATIVE :
            return stand_pat_current_player_pov

        best_val_current_player_pov = stand_pat_current_player_pov
        fen_at_this_q_level = self.board.fen() # Not strictly needed anymore for NN context

        for move in ordered_q_moves:
            self.board.push(move)
            score = -self.quiescence(-beta, -alpha, 
                                     root_fen_for_nn_context, 
                                     ply_count + 1)
            self.board.pop()
            
            if score > best_val_current_player_pov:
                best_val_current_player_pov = score
            
            alpha = max(alpha, best_val_current_player_pov)
            if alpha >= beta: break 
        
        return best_val_current_player_pov

def classical_material_eval(fen: str) -> int:
    board = chess.Board(fen)
    MATERIAL_VALUES = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
                       chess.ROOK: 500, chess.QUEEN: 900}
    score = 0
    
    # Basic material count
    for square, piece in board.piece_map().items():
        value = MATERIAL_VALUES.get(piece.piece_type, 0)
        score += value if piece.color == chess.WHITE else -value
    
    # Basic safety checks to prevent obvious blunders
    # Penalize hanging pieces heavily
    for color in [chess.WHITE, chess.BLACK]:
        multiplier = 1 if color == chess.WHITE else -1
        
        # Check for undefended pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                if piece.piece_type != chess.PAWN:  # Don't check pawns
                    attackers = board.attackers(not color, square)
                    defenders = board.attackers(color, square)
                    
                    if attackers and len(attackers) > len(defenders):
                        # Hanging piece penalty
                        penalty = MATERIAL_VALUES.get(piece.piece_type, 0) // 2
                        score -= penalty * multiplier
    
    # King safety - heavily penalize exposed kings
    for color in [chess.WHITE, chess.BLACK]:
        king_square = board.king(color)
        if king_square is not None:
            multiplier = 1 if color == chess.WHITE else -1
            enemy_color = not color
            
            # Count enemy attacks around king
            king_zone_attacks = 0
            for offset in [-9, -8, -7, -1, 1, 7, 8, 9]:
                target_square = king_square + offset
                if 0 <= target_square < 64:
                    if board.is_attacked_by(enemy_color, target_square):
                        king_zone_attacks += 1
            
            # Penalty for exposed king
            if king_zone_attacks >= 3:
                score -= (king_zone_attacks * 15) * multiplier
    
    return score