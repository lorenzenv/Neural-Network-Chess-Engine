import chess
import numpy as np
import functools
import time
import tflite_runtime.interpreter as tflite
from util import * # Changed from specific imports to wildcard

# Global model loading
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
# These are global lists of dictionaries
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ------- Engine Metadata -------
ENGINE_NAME = "AdvSearch++ Pure NN Eval"
ENGINE_VERSION = "1.9.3-PVSFix" # Version increment
ENGINE_FEATURES = [
    "🧠 NN Relative Eval for Deltas, Accumulated Absolute Score (White's POV Tracking)",
    "♟️ Iterative Deepening Alpha-Beta with PVS",
    "💾 Fixed-Size Zobrist-Hashed Transposition Table",
    "🔪 MVV-LVA Move Ordering & Killer Moves",
    "🤫 Quiescence Search with Futility Pruning (Corrected Stand-Pat)",
    "📉 Null Move Pruning & Late Move Reductions",
    "🔍 Check Extensions",
]

# ------- Configuration -------
class Config:
    NN_SCALING_FACTOR = 1000.0
    MAX_PLY_FOR_KILLERS = 30
    QUIESCENCE_MAX_DEPTH_RELATIVE = 5
    LMR_MIN_MOVES_TRIED = 3 # Number of full-depth/PV moves to try before LMR can activate
    LMR_REDUCTION = 1       # Depth reduction for LMR moves
    NMP_REDUCTION = 3
    TT_SIZE_POWER = 22 # 2^22 entries (approx 4 million)
    FEN_CACHE_SIZE = 4096
    MAX_SEARCH_DEPTH_ID = 6
    ITERATIVE_DEEPENING_TIME_LIMIT_PER_MOVE = 20.0 # Max time for the entire get_move call

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
def make_x_cached(fen_cur: str, fen_par: str):
    b1 = make_bitboard(beautifyFEN(fen_cur))
    b2 = make_bitboard(beautifyFEN(fen_par))
    return (
        np.array(b1, dtype=np.float32).reshape(1, 769),
        np.array(b2, dtype=np.float32).reshape(1, 769)
    )

# ------- NN Evaluator -------
class NNEvaluator:
    def __init__(self): # model_path argument removed
        # Interpreter is now global, no need to initialize here
        pass

    def evaluate_single_delta_white_pov(self, fen_current: str, fen_parent: str) -> float:
        x1_np, x2_np = make_x_cached(fen_current, fen_parent)
        
        # Use global interpreter and details
        # input_details[0] for the first input tensor
        # input_details[1] for the second input tensor
        # output_details[0] for the output tensor
        interpreter.set_tensor(input_details[0]['index'], x1_np)
        interpreter.set_tensor(input_details[1]['index'], x2_np)
        interpreter.invoke()
        raw_evaluation = interpreter.get_tensor(output_details[0]['index'])[0][0]
        
        scaled_delta_for_white = (float(raw_evaluation) - 0.5) * Config.NN_SCALING_FACTOR
        
        # ---- START DEBUG (Uncomment selectively for debugging) ----
        # if "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq" in fen_parent and Config.NN_SCALING_FACTOR > 1.0:
        # if True:
        #    unscaled_delta = float(raw_evaluation) - 0.5
        #    print(f"DEBUG DELTA: fen_curr={fen_current.split()[0]} fen_par={fen_parent.split()[0]} raw={raw_evaluation:.4f} unscaled_delta={unscaled_delta:.4f} scaled_delta={scaled_delta_for_white:.2f}")
        # ---- END DEBUG ----
        return scaled_delta_for_white

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
        return (time.time() - self.start_time_for_move) >= self.time_limit_for_move

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

        for move in legal_moves:
            score = 0
            if move == tt_move_obj: score = 10000000 
            elif self.board.is_capture(move):
                score = 2000000 
                victim_type = chess.PAWN 
                if self.board.is_en_passant(move): victim_type = chess.PAWN
                else:
                    victim_piece_obj = self.board.piece_at(move.to_square)
                    if victim_piece_obj: victim_type = victim_piece_obj.piece_type
                attacker_piece_obj = self.board.piece_at(move.from_square)
                attacker_type = attacker_piece_obj.piece_type if attacker_piece_obj else chess.PAWN
                score += (PIECE_VALUES.get(victim_type, 0) * 100) - PIECE_VALUES.get(attacker_type, 100)
            elif ply_count < Config.MAX_PLY_FOR_KILLERS and move in self.killer_moves[ply_count]:
                score = 1000000 if move == self.killer_moves[ply_count][0] else 900000
            elif move.promotion == chess.QUEEN: score = 1500000
            elif move.promotion: score = PIECE_VALUES.get(move.promotion, 300) * 1000 
            elif self.board.gives_check(move): score = 500000
            move_scores.append((score, move))
        move_scores.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in move_scores]

    def iterative_deepening_search(self):
        self.start_time_for_move = time.time()
        self.nodes_searched = 0; self.q_nodes_searched = 0; self.tt_hits = 0
        self.beta_cutoffs = 0; self.nmp_cutoffs = 0; self.lmr_activations = 0
        
        # Basic TT clearing strategy
        if len(self.tt.entries) > (1 << (Config.TT_SIZE_POWER - 2)): # Clear if more than quarter full
             self.tt = FixedTT()
             self.killer_moves = [[None, None] for _ in range(Config.MAX_PLY_FOR_KILLERS)]

        best_move_overall_uci = None; best_score_overall_white_pov = -float('inf')
        initial_zobrist_key_root = self.zobrist_hasher.hash(self.board)
        initial_fen_at_root = self.board.fen()


        for depth in range(1, Config.MAX_SEARCH_DEPTH_ID + 1):
            if self.time_is_up() and depth > 1 :
                print(f"Time up before Depth {depth}. Using Depth {depth-1} result.")
                break
            
            search_start_time_this_depth = time.time()
            try:
                score_from_root_player_pov = self.alpha_beta(depth, -float('inf'), float('inf'), 
                                                                initial_fen_at_root, 0.0, 
                                                                True, 0)
            except TimeoutError:
                print(f"Search for Depth {depth} timed out.")
                # Try to use previous depth's best move if available
                tt_entry_root = self.tt.lookup(initial_zobrist_key_root)
                if tt_entry_root and tt_entry_root[2]:
                    best_move_overall_uci = tt_entry_root[2]
                    # Score might be from previous iteration, not current one
                break # Stop iterative deepening

            search_duration_this_depth = time.time() - search_start_time_this_depth
            
            tt_entry_root = self.tt.lookup(initial_zobrist_key_root)
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
                     parent_fen_for_delta: str, parent_abs_white_pov: float,
                     is_pv_node: bool, ply_count: int) -> float: # Returns score from current player's POV
        
        if self.time_is_up() and ply_count > 0 : raise TimeoutError("Search time limit exceeded") # Check deeper in search

        self.nodes_searched += 1
        alpha_original = alpha # Used for TT flag
        current_node_zobrist_key = self.zobrist_hasher.hash(self.board)

        if ply_count > 0 and (self.board.can_claim_threefold_repetition() or self.board.is_fifty_moves()):
            return 0.0 

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

        current_node_abs_score_white_pov = parent_abs_white_pov + \
            self.nn_evaluator.evaluate_single_delta_white_pov(self.board.fen(), parent_fen_for_delta)
        
        # ---- START DEBUG ----
        # if ply_count < 3 and Config.NN_SCALING_FACTOR > 1.0 :
        #    delta_for_print = self.nn_evaluator.evaluate_single_delta_white_pov(self.board.fen(), parent_fen_for_delta)
        #    print(f"DEBUG SCORE ACC: ply={ply_count} parent_abs_WPOV={parent_abs_white_pov:.2f} delta_WPOV={delta_for_print:.2f} current_abs_WPOV={current_node_abs_score_white_pov:.2f} fen={self.board.fen().split()[0]} ToMove={'W' if self.board.turn else 'B'} alpha={alpha:.2f} beta={beta:.2f} PV={is_pv_node}")
        # ---- END DEBUG ----

        if depth_remaining <= 0:
            return self.quiescence(alpha, beta, self.board.fen(), current_node_abs_score_white_pov, ply_count)

        can_do_nmp = depth_remaining >= (Config.NMP_REDUCTION + 1) and not self.board.is_check() and ply_count > 0
        if can_do_nmp:
            non_pawn_material = sum(len(self.board.pieces(pt, self.board.turn)) for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
            if non_pawn_material < 2 : can_do_nmp = False
        
        if can_do_nmp:
            self.board.push(chess.Move.null())
            # Parent for NMP child is current node (null-moved), its score is current_node_abs_score_white_pov
            score = -self.alpha_beta(depth_remaining - 1 - Config.NMP_REDUCTION, -beta, -alpha, 
                                     self.board.fen(), current_node_abs_score_white_pov, 
                                     False, ply_count + 1) 
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
            extension = 1 if is_giving_check else 0
            child_search_depth = depth_remaining - 1 + extension
            
            current_move_score_from_child_pov = 0 

            if i == 0 or not is_pv_node: 
                current_move_score_from_child_pov = -self.alpha_beta(child_search_depth, -beta, -alpha,
                                                                self.board.fen(), current_node_abs_score_white_pov,
                                                                is_pv_node, # Child is PV only if parent is PV and it's the first move
                                                                ply_count + 1)
            else: 
                # LMR for non-PV moves or non-first moves in PV node
                if child_search_depth >= Config.LMR_REDUCTION and \
                   i >= Config.LMR_MIN_MOVES_TRIED and \
                   not extension and not self.board.is_capture(move) and not move.promotion:
                    
                    self.lmr_activations += 1
                    current_move_score_from_child_pov = -self.alpha_beta(child_search_depth - Config.LMR_REDUCTION, 
                                                                    -alpha -1, -alpha, 
                                                                    self.board.fen(), current_node_abs_score_white_pov,
                                                                    False, ply_count + 1)
                else: # If LMR not applicable (e.g. capture, promo, check, or too shallow/few moves tried)
                      # Still do a null-window search if it's not the first move of a PV line
                     current_move_score_from_child_pov = -self.alpha_beta(child_search_depth, 
                                                                    -alpha -1, -alpha, 
                                                                    self.board.fen(), current_node_abs_score_white_pov,
                                                                    False, ply_count + 1)

                if current_move_score_from_child_pov > alpha and current_move_score_from_child_pov < beta : 
                    current_move_score_from_child_pov = -self.alpha_beta(child_search_depth, -beta, -alpha,
                                                                    self.board.fen(), current_node_abs_score_white_pov,
                                                                    True, ply_count + 1) 
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
                    parent_fen_for_delta: str, parent_abs_white_pov: float, 
                    ply_count: int) -> float: # Returns score from current player's POV
        self.q_nodes_searched += 1
        
        if self.time_is_up() and ply_count > 0: raise TimeoutError("Search time limit exceeded")

        if self.board.can_claim_threefold_repetition() or self.board.is_fifty_moves(): return 0.0
        
        stand_pat_score_white_pov = parent_abs_white_pov + \
            self.nn_evaluator.evaluate_single_delta_white_pov(self.board.fen(), parent_fen_for_delta)
        
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

        for move in ordered_q_moves:
            self.board.push(move)
            # Child's parent_fen is current_q_node's fen, child's parent_abs_white_pov is current_q_node's stand_pat_score_white_pov
            score = -self.quiescence(-beta, -alpha, 
                                     self.board.fen(), stand_pat_score_white_pov, 
                                     ply_count + 1)
            self.board.pop()
            
            if score > best_val_current_player_pov:
                best_val_current_player_pov = score
            
            alpha = max(alpha, best_val_current_player_pov)
            if alpha >= beta: break 
        
        return best_val_current_player_pov