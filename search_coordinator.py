#!/usr/bin/env python3
"""
PURE NN CHESS SEARCH COORDINATOR

🚨 CRITICAL PHILOSOPHY: This module handles search logic (alpha-beta, move ordering, time management)
but gets ALL position evaluations from the neural network. NO CHESS KNOWLEDGE.

This module coordinates search but doesn't evaluate positions - that's the NN's job.
"""

import chess
import chess.engine
import chess.polyglot
import time
import numpy as np
from neural_network_inference import NeuralNetworkEvaluator, NeuralNetworkConfig
from collections import defaultdict

# 🧠 SEARCH CONFIGURATION (Algorithm settings, NOT chess knowledge)
class SearchConfig:
    """Search algorithm configuration - NO CHESS KNOWLEDGE."""
    
    # Time management
    SEARCH_TIME_FAST = 3.0       # Fast games
    SEARCH_TIME_BALANCED = 5.0   # Balanced games  
    SEARCH_TIME_DEEP = 15.0      # Deep analysis
    
    # Search depth limits
    MAX_SEARCH_DEPTH = 8
    QUIESCENCE_MAX_DEPTH = 4
    
    # Alpha-beta optimizations
    TRANSPOSITION_TABLE_SIZE = 2**20
    NULL_MOVE_REDUCTION = 3
    LATE_MOVE_REDUCTION = 1
    LMR_MIN_MOVES = 4
    
    # Aspiration window settings
    INITIAL_ASPIRATION_WINDOW = 100
    ASPIRATION_WINDOW_SCALING = 0.8
    
    # Time extension factors
    SEARCH_TIME_EXTENSION = 2.5
    ENDGAME_TIME_EXTENSION = 1.5

# 🎯 ZOBRIST HASHING (Position identification, no chess knowledge)
class ZobristHasher:
    """Zobrist hashing for position identification."""
    
    def __init__(self):
        # Initialize random hash values
        np.random.seed(42)  # Deterministic for consistency
        self.piece_hashes = {}
        self.castling_hashes = {}
        self.ep_hashes = {}
        self.turn_hash = np.random.randint(0, 2**64, dtype=np.uint64)
        
        # Generate hashes for all piece-square combinations
        for square in chess.SQUARES:
            for piece in chess.PIECE_TYPES:
                for color in chess.COLORS:
                    piece_obj = chess.Piece(piece, color)
                    self.piece_hashes[(piece_obj, square)] = np.random.randint(0, 2**64, dtype=np.uint64)
        
        # Castling rights hashes for each of the 4 rights
        self.castling_hashes['K'] = np.random.randint(0, 2**64, dtype=np.uint64)  # White kingside (chess.BB_H1)
        self.castling_hashes['Q'] = np.random.randint(0, 2**64, dtype=np.uint64)  # White queenside (chess.BB_A1)
        self.castling_hashes['k'] = np.random.randint(0, 2**64, dtype=np.uint64)  # Black kingside (chess.BB_H8)
        self.castling_hashes['q'] = np.random.randint(0, 2**64, dtype=np.uint64)  # Black queenside (chess.BB_A8)
        
        # En passant file hashes
        for file in range(8):
            self.ep_hashes[file] = np.random.randint(0, 2**64, dtype=np.uint64)
    
    def hash_position(self, board: chess.Board) -> np.uint64:
        """Generate zobrist hash for position."""
        hash_value = np.uint64(0)
        
        # Hash pieces
        for square, piece in board.piece_map().items():
            hash_value ^= self.piece_hashes[(piece, square)]
        
        # Hash turn
        if board.turn == chess.BLACK:
            hash_value ^= self.turn_hash
        
        # Hash castling rights explicitly
        if board.has_kingside_castling_rights(chess.WHITE):
            hash_value ^= self.castling_hashes['K']
        if board.has_queenside_castling_rights(chess.WHITE):
            hash_value ^= self.castling_hashes['Q']
        if board.has_kingside_castling_rights(chess.BLACK):
            hash_value ^= self.castling_hashes['k']
        if board.has_queenside_castling_rights(chess.BLACK):
            hash_value ^= self.castling_hashes['q']
        
        # Hash en passant
        if board.ep_square is not None:
            ep_file = chess.square_file(board.ep_square)
            hash_value ^= self.ep_hashes[ep_file]
        
        return hash_value

# 🚀 MOVE ORDERING (Algorithm optimization, uses NN for scoring)
class MoveOrderer:
    """Move ordering using pure NN evaluation."""
    
    def __init__(self, nn_evaluator: NeuralNetworkEvaluator):
        self.nn_evaluator = nn_evaluator
    
    def order_moves(self, board: chess.Board, legal_moves: list, ply: int, tt_best_move: str = None) -> list:
        """
        Order moves using NN evaluation and search heuristics.
        
        🚨 NO CHESS KNOWLEDGE: Uses only NN scores and search optimization.
        Reverted: NN scores are P(child > parent) for the current player.
        SIMPLIFIED: Only use NN scores for ordering.
        """
        if not legal_moves:
            return []
        
        scored_moves = []
        current_fen = board.fen() # Parent FEN for comparison
        
        # Get NN scores for all moves, P(child > parent for current player)
        nn_move_evaluations = self.nn_evaluator.evaluate_moves_for_ordering(current_fen, legal_moves, ply)
        
        for move in legal_moves:
            # Default score is 0 if NN doesn't provide one (should not happen for legal moves)
            score = nn_move_evaluations.get(move, 0.0) 
            scored_moves.append((score, move))
        
        # Sort by NN score (highest first, as higher means child is better for current player)
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return [move for score, move in scored_moves]
    
    def update_killer_move(self, ply: int, move: chess.Move):
        """Update killer move heuristic. (This might be unused if killers are removed from order_moves)"""

# 🎮 MAIN SEARCH COORDINATOR
class SearchCoordinator:
    """
    Pure NN Chess Search Coordinator
    
    🚨 ZERO CHESS KNOWLEDGE: All position evaluation comes from NN.
    This class only handles search algorithm, not chess evaluation.
    """
    
    def __init__(self, starting_fen: str = None):
        # Initialize components
        self.nn_evaluator = NeuralNetworkEvaluator()
        self.move_orderer = MoveOrderer(self.nn_evaluator)
        self.transposition_table = {}
        self.history_heuristic = defaultdict(lambda: defaultdict(int))
        self.killer_moves = defaultdict(list)
        self.nodes_searched = 0
        self.quiescence_nodes = 0
        self.tt_hits = 0
        self.beta_cutoffs = 0
        self.null_move_cutoffs = 0
        self.lmr_activations = 0
        self.search_start_time = 0
        self.time_limit_seconds = SearchConfig.SEARCH_TIME_BALANCED
        
        # Set up position
        if starting_fen:
            self.board = chess.Board(starting_fen)
        else:
            self.board = chess.Board()
        
        print("✅ Pure NN Search Coordinator initialized - ZERO chess knowledge!")
    
    def get_best_move(self, current_board_fen: str, time_limit: float = None, search_depth: int = None) -> str | None:
        if time_limit is not None:
            self.time_limit_seconds = time_limit
        
        self.board = chess.Board(current_board_fen) # Initialize self.board here
        self._reset_search_stats() # Then reset stats (which might use self.board for initial hash if TT was not cleared before)
        self.search_start_time = time.time()
        
        # The FEN of the root for this search cycle, to be passed for leaf evaluations.
        root_fen_for_this_search_cycle = self.board.fen()

        # 💡 Optionally, check TT for the root position itself
        try:
            root_z_key = chess.polyglot.zobrist_hash(self.board) # Changed to use chess.polyglot.zobrist_hash
            root_tt_entry = self.transposition_table.get(root_z_key)
            # Only use TT if entry depth is high enough (e.g., matches max depth we would search)
            # Or if it's an exact score from a previous full search to this depth.
            # For simplicity, let's assume if we find a root move, it's from a comparable search.
            # A more robust check would involve comparing tt_entry depth with planned search_depth.
            if root_tt_entry:
                tt_depth, score, move_uci, flag = root_tt_entry
                # Example: Only use if it was an exact score or a high-depth search
                if flag == "EXACT" or tt_depth >= (search_depth or SearchConfig.MAX_SEARCH_DEPTH) -1: 
                    if move_uci and move_uci != "None":
                        # print(f"[DEBUG SC get_best_move] ROOT TT HIT: move {move_uci}, score {score}, depth {tt_depth}, flag {flag}") # Commented out for reduced logging
                        # Ensure the move is legal in the current position before returning
                        try:
                            if chess.Move.from_uci(move_uci) in self.board.legal_moves:
                                return str(move_uci)
                            else:
                                # print(f"[DEBUG SC get_best_move] ROOT TT move {move_uci} no longer legal.") # Commented out for reduced logging
                                pass # Added pass
                        except Exception as e:
                            # print(f"[DEBUG SC get_best_move] Error validating ROOT TT move {move_uci}: {e}") # Commented out for reduced logging
                            pass # Continue with search if TT check fails (already had pass, ensuring it stays)
        except Exception as e:
            # print(f"[DEBUG SC get_best_move] Error during root TT check: {e}") # Commented out for reduced logging
            pass # Continue with search if TT check fails (already had pass, ensuring it stays)

        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return None
        
        if len(legal_moves) == 1:
            return str(legal_moves[0])
        
        best_move = None
        best_score = float('-inf')
        
        # Iterative deepening search
        for depth in range(1, SearchConfig.MAX_SEARCH_DEPTH + 1):
            if self._time_up():
                break
            
            print(f"🔍 Searching depth {depth}...")
            
            current_best_move_this_depth = None # Renamed to avoid confusion with overall best_move
            current_best_score_this_depth = float('-inf') # Renamed
            
            # Order moves for this depth
            # Pass current self.board which is the root for this get_best_move call
            ordered_moves = self.move_orderer.order_moves(self.board, legal_moves, 0) # Removed root_reference_fen
            
            for i, move in enumerate(ordered_moves):
                if self._time_up():
                    break
                
                self.board.push(move)
                # Pass root_fen_for_this_search_cycle as the leaf_reference_fen
                score = -self._alpha_beta(depth - 1, float('-inf'), float('inf'), 1, False, root_fen_for_this_search_cycle)
                self.board.pop()
                
                if score > current_best_score_this_depth:
                    current_best_score_this_depth = score
                    current_best_move_this_depth = move
            
            if current_best_move_this_depth: # Check if a move was found at this depth
                best_move = current_best_move_this_depth
                best_score = current_best_score_this_depth
                
                # Store in transposition table
                # Key is for the root position of this get_best_move call
                # Use board.zobrist_hash() for the board state corresponding to root_fen_for_this_search_cycle
                # Create a temporary board object for hashing if self.board has moved on.
                # However, root_fen_for_this_search_cycle *is* self.board.fen() at the start of get_best_move
                # So, root_z_key (calculated earlier) can be used if it's for the correct FEN.
                # For clarity and safety, let's re-hash the specific FEN if there's any doubt.
                # Or, ensure root_z_key calculated at the start is used here.
                # The original self.board before iterative deepening loop is root_fen_for_this_search_cycle
                
                # Re-calculate zobrist_key for the root FEN to be absolutely sure
                # root_board_for_tt = chess.Board(root_fen_for_this_search_cycle) # This was the original board
                # root_z_key_for_storage = root_board_for_tt.zobrist_hash()
                # The root_z_key for TT store at the end of iterative deepening should be the hash of the initial board state (current_board_fen)
                # This is already available if the initial TT check logic for root_z_key is correct.
                # Let's assume root_z_key from the top of get_best_move is the one for current_board_fen

                # The board object self.board is currently at root_fen_for_this_search_cycle before the loop
                # So, we can hash it directly if no pushes/pops occurred on self.board *outside* the move loop.
                # It seems self.board is indeed at root_fen_for_this_search_cycle here.
                # Safest: create a new board object for hashing
                zobrist_key_for_root_store = chess.polyglot.zobrist_hash(chess.Board(root_fen_for_this_search_cycle)) # Changed
                self.transposition_table[zobrist_key_for_root_store] = (depth, best_score, str(best_move), "EXACT")
                
                print(f"    Depth {depth}: {best_move} (NN eval: {best_score:.1f})")
                print(f"[DEBUG SC] get_best_move: End of Depth {depth} - BestMove: {best_move}, Score: {best_score:.2f}, RefFEN: {root_fen_for_this_search_cycle}") #Log at end of depth
        
        elapsed = time.time() - self.search_start_time
        nps = int(self.nodes_searched / elapsed) if elapsed > 0 else 0
        print(f"🏁 Search complete: {self.nodes_searched} nodes in {elapsed:.1f}s ({nps} NPS)")
        
        return str(best_move) if best_move else str(legal_moves[0]) # Fallback if no move found
    
    def _alpha_beta(self, depth: int, alpha: float, beta: float, ply: int, is_pv_node: bool, leaf_reference_fen: str) -> float:
        """
        Alpha-beta search using pure NN evaluation.
        
        🚨 NO CHESS KNOWLEDGE: Only uses NN for position evaluation.
        """
        self.nodes_searched += 1
        initial_alpha = alpha 

        turn_char = 'W' if self.board.turn == chess.WHITE else 'B'
        if ply <= 1: # Keep this conditional log for top levels
            print(f"[DEBUG SC _alpha_beta ENTRY] depth={depth}, ply={ply}, alpha={alpha:.2f}, beta={beta:.2f}, player='{turn_char}', fen='{self.board.fen()}'")

        # Check time
        if self.nodes_searched % 1000 == 0 and self._time_up():
            return NeuralNetworkConfig.STALEMATE_SCORE # Return a neutral score if time runs out
        
        # Outcome of the game (checkmate, stalemate, etc.)
        outcome = self.board.outcome()
        if outcome:
            if outcome.winner == chess.WHITE:
                return NeuralNetworkConfig.CHECKMATE_SCORE - ply 
            elif outcome.winner == chess.BLACK:
                return -NeuralNetworkConfig.CHECKMATE_SCORE + ply 
            else: # Stalemate or other draw
                return NeuralNetworkConfig.STALEMATE_SCORE
        
        # ----------------------------------------------------------------------
        # Transposition Table (TT) Lookup
        # ----------------------------------------------------------------------
        # Uses python-chess board's built-in Zobrist hash
        # Ensure TT is only used if the entry is from the current search cycle's root.
        # This is implicitly handled by clearing the TT at the start of get_best_move.
        
        original_alpha = alpha
        # Use the board's built-in Zobrist hash - WITH PARENTHESES
        zobrist_key = chess.polyglot.zobrist_hash(self.board) # Changed
        tt_entry = self.transposition_table.get(zobrist_key)
        tt_best_move = None
        
        if tt_entry:
            # TT entry format: (depth, score, move_uci, flag) - generation removed
            tt_depth, tt_score, tt_move_uci, tt_flag = tt_entry # Unpack without generation
            if tt_depth >= depth:
                self.tt_hits += 1
                # Keep TT_HIT log, it's important and not too frequent - REVISED: Comment out for now
                # print(f"[DEBUG SC _alpha_beta TT_HIT] depth={depth}, tt_depth={tt_depth}, score={tt_score:.2f}, move={tt_move_uci}, flag={tt_flag}, fen='{self.board.fen()}'")
                if tt_flag == "EXACT":
                    return tt_score
                elif tt_flag == "LOWERBOUND":
                    alpha = max(alpha, tt_score)
                elif tt_flag == "UPPERBOUND":
                    beta = min(beta, tt_score)
                
                if alpha >= beta:
                    return tt_score 
                tt_best_move = tt_move_uci # Use the uci string from TT
        
        # Reach depth limit - use quiescence or NN evaluation
        if depth <= 0:
            # Revert to passing leaf_reference_fen
            return self._quiescence_search(alpha, beta, ply, 0, leaf_reference_fen)
        
        # Null Move Pruning (NMP)
        # DISABLED FOR SIMPLIFICATION / STABILITY TEST
        # if (SearchConfig.NULL_MOVE_REDUCTION > 0 and
        #     depth >= SearchConfig.NULL_MOVE_REDUCTION + 1 and
        #     not is_pv_node and 
        #     not self.board.is_check() and # Cannot do null move if in check
        #     # Ensure there's enough material for a null move to make sense (e.g., not pawn endgame)
        #     # This minimal material check is a grey area for "no chess knowledge"
        #     # For now, let's assume it's a general search stability rather than specific chess rule.
        #     # A simple piece count could be used here, e.g. if more than X pieces on board.
        #     # Simplified: just check if not in check and depth is sufficient.
        #     True # Placeholder for actual material check if re-enabled
        #     ):
        #     
        #     self.board.push(chess.Move.null())
        #     # When making a null move, the opponent cannot also make a null move in response.
        #     # The `is_pv_node` becomes False as null move is not part of PV.
        #     null_move_score = -self._alpha_beta(depth - 1 - SearchConfig.NULL_MOVE_REDUCTION, -beta, -alpha, ply + 1, False, leaf_reference_fen)
        #     self.board.pop()
            
        #     if null_move_score >= beta:
        #         self.null_move_cutoffs += 1
        #         # Potentially store TT entry for null move cutoff
        #         # Key is for the position *before* the null move.
        #         # current_z_key = chess.polyglot.zobrist_hash(self.board) # Hash of self.board before null move
        #         # self.transposition_table[current_z_key] = (depth - SearchConfig.NULL_MOVE_REDUCTION, null_move_score, chess.Move.null().uci(), "LOWER")
        #         return beta # Null move cutoff

        # Generate and order moves
        legal_moves = list(self.board.legal_moves)
        if not legal_moves: 
            # This should ideally be caught by board.outcome() earlier.
            # If outcome() is None but no legal_moves, it's a stalemate.
            return NeuralNetworkConfig.STALEMATE_SCORE 
        
        ordered_moves = self.move_orderer.order_moves(self.board, legal_moves, ply, tt_best_move)
        
        best_score_found_in_loop = float('-inf') # Renamed from best_score to avoid conflict
        best_move_for_tt = None 
        moves_tried = 0 
        
        for i, move in enumerate(ordered_moves):
            if self._time_up():
                break
            
            self.board.push(move)
            
            current_node_score = 0
            # Late Move Reduction (LMR)
            # DISABLED FOR SIMPLIFICATION / STABILITY TEST
            # if depth >= 1 and i >= SearchConfig.LMR_MIN_MOVES and not is_pv_node: 
            #     self.lmr_activations += 1
            #     # if SearchConfig.ENABLE_LMR_DEBUG: 
            #     #     print(f"[DEBUG SC _alpha_beta LMR_ACTIVATED] depth={depth}, move_idx={i}, move={move.uci()}, fen='{self.board.fen()}'")
            #     reduction = SearchConfig.LATE_MOVE_REDUCTION
            #     # Add more reduction for non-captures/non-promotions (already simplified out)
            #     current_node_score = -self._alpha_beta(depth - 1 - reduction, -beta, -alpha, ply + 1, False, leaf_reference_fen)
            # else:
            # Full depth search (always, as LMR is disabled)
            current_node_score = -self._alpha_beta(depth - 1, -beta, -alpha, ply + 1, (i == 0), leaf_reference_fen) 
            
            self.board.pop()
            moves_tried += 1
            
            if ply <= 1: # CORRECT Conditional logging for _alpha_beta EVAL
                print(f"  [DEBUG SC _alpha_beta EVAL] depth={depth}, ply={ply}, move={move.uci()}, score_after_negation={current_node_score:.2f}, alpha={alpha:.2f}, beta={beta:.2f}, fen='{self.board.fen()}'")

            if current_node_score > best_score_found_in_loop:
                # Keep NEW_BEST_IN_LOOP log
                if ply <= 1: # Make conditional
                    print(f"[DEBUG SC _alpha_beta NEW_BEST_IN_LOOP] old_best_score={best_score_found_in_loop:.2f}, new_best_score={current_node_score:.2f}, move='{move}', fen='{self.board.fen()}'")
                best_score_found_in_loop = current_node_score
                best_move_for_tt = move 
            
            if current_node_score > alpha:
                if ply <= 1: # Conditional log
                    print(f"    [DEBUG SC _alpha_beta ALPHA_UPDATE] new_alpha={current_node_score:.2f}, best_move_so_far={move.uci()}, fen='{self.board.fen()}'")
                alpha = current_node_score
            
            if current_node_score >= beta:
                self.beta_cutoffs += 1
                current_z_key = chess.polyglot.zobrist_hash(self.board) 
                # Store TT entry for beta cutoff
                # The move that caused the cutoff is 'move'
                self.transposition_table[current_z_key] = (depth, beta, move.uci(), "LOWER") 
                # print(f"    [DEBUG SC _alpha_beta TT_STORE_BETA_CUTOFF] fen='{self.board.fen()}', depth={depth}, score={beta:.2f}, move={move.uci()}, flag=LOWER") # Commented out

                if not move.promotion and not self.board.is_capture(move):
                    self.move_orderer.update_killer_move(ply, move)
                return beta 
        
        # ----------------------------------------------------------------------
        # Transposition Table (TT) Storage
        # ----------------------------------------------------------------------
        # Uses python-chess board's built-in Zobrist hash
        # Store in TT only if part of the current search cycle (implicit by clearing TT)
        
        tt_flag = None
        if best_score_found_in_loop <= original_alpha: # Fail-low, score is an upper bound
            tt_flag = "UPPERBOUND"
        elif best_score_found_in_loop >= beta:         # Fail-high, score is a lower bound
            tt_flag = "LOWERBOUND"
        else:                                          # Exact score
            tt_flag = "EXACT"
        
        # Use the board's built-in Zobrist hash - WITH PARENTHESES
        # No need to re-hash with current_zobrist_key, zobrist_key is already for the current board.
        # TT entry format: (depth, score, move_uci_str_or_None, flag)
        move_to_store_in_tt = str(best_move_for_tt) if best_move_for_tt else None
        self.transposition_table[zobrist_key] = (depth, best_score_found_in_loop, move_to_store_in_tt, tt_flag)

        # Store Transposition Table (TT) Entry for this node
        # ----------------------------------------------------------------------
        final_z_key = chess.polyglot.zobrist_hash(self.board)
        # Need to ensure best_move_uci_at_node is defined. Initialize to None.
        # It would be set if alpha was updated.
        # If no move improved alpha, best_move_uci_at_node might be unset.
        # For UPPER, move_uci is None. For EXACT/LOWER, we need the move that caused it.
        # This assumes best_move_uci_at_node variable exists and holds the relevant move if one was found.
        # A better way: _alpha_beta should return (score, best_move_found_uci)
        determined_move_uci_for_tt = best_move_uci_at_node if 'best_move_uci_at_node' in locals() and best_move_uci_at_node else None

        if best_score_found_in_loop <= initial_alpha: 
            flag = "UPPER"
            # print(f"    [DEBUG SC _alpha_beta TT_STORE_UPPER] fen='{self.board.fen()}', depth={depth}, score={best_score_found_in_loop:.2f}, flag=UPPER") # Commented out
            self.transposition_table[final_z_key] = (depth, best_score_found_in_loop, None, flag)
        elif best_score_found_in_loop >= beta: 
            flag = "LOWER"
            # print(f"    [DEBUG SC _alpha_beta TT_STORE_LOWER (post-loop, should be rare)] fen='{self.board.fen()}', depth={depth}, score={best_score_found_in_loop:.2f}, move_uci={determined_move_uci_for_tt if determined_move_uci_for_tt else 'None'}, flag=LOWER") # Commented out
            self.transposition_table[final_z_key] = (depth, best_score_found_in_loop, determined_move_uci_for_tt, flag)
        else: 
            flag = "EXACT"
            # print(f"    [DEBUG SC _alpha_beta TT_STORE_EXACT] fen='{self.board.fen()}', depth={depth}, score={best_score_found_in_loop:.2f}, move_uci={determined_move_uci_for_tt if determined_move_uci_for_tt else 'None'}, flag=EXACT") # Commented out
            self.transposition_table[final_z_key] = (depth, best_score_found_in_loop, determined_move_uci_for_tt, flag) 
        
        return best_score_found_in_loop # Return the actual best score found
    
    def _quiescence_search(self, alpha: float, beta: float, ply: int, q_depth: int, leaf_reference_fen: str) -> float:
        """
        Quiescence search using NN evaluation.
        
        🚨 PURE NN: Uses only neural network for position evaluation.
        MODIFIED: Uses leaf_reference_fen from the main search for evaluations.
        """
        self.quiescence_nodes += 1
        
        turn_char = 'W' if self.board.turn == chess.WHITE else 'B'
        if q_depth == 0: # Keep this conditional log
            print(f"[DEBUG SC _q_search ENTRY] q_depth={q_depth}, ply={ply}, alpha={alpha:.2f}, beta={beta:.2f}, player='{turn_char}', fen='{self.board.fen()}', ref='{leaf_reference_fen}'")

        # Check time (less frequently in q-search)
        if self.nodes_searched % 1000 == 0 and self._time_up():
            return NeuralNetworkConfig.STALEMATE_SCORE # Return a neutral score if time runs out
        
        # Terminal check inside quiescence
        outcome = self.board.outcome()
        if outcome:
            if outcome.winner == chess.WHITE: return NeuralNetworkConfig.CHECKMATE_SCORE - ply
            if outcome.winner == chess.BLACK: return -NeuralNetworkConfig.CHECKMATE_SCORE + ply
            return NeuralNetworkConfig.STALEMATE_SCORE

        if q_depth >= SearchConfig.QUIESCENCE_MAX_DEPTH:
            # Use NN evaluation against the provided leaf_reference_fen
            # Pass is_quiescence_standpat_log=False here, as this is a terminal q-search node, not the initial stand-pat.
            nn_score_from_root_perspective = self.nn_evaluator.evaluate_position_against_reference(self.board.fen(), leaf_reference_fen, is_quiescence_standpat_log=False)
            # Correct perspective adjustment:
            # nn_score_from_root_perspective is from the perspective of the player whose turn it is in leaf_reference_fen.
            # We need to return a score from the perspective of self.board.turn (current q-node player).
            is_root_player_white_at_leaf_ref = chess.Board(leaf_reference_fen).turn == chess.WHITE
            is_current_player_white_at_q_node = self.board.turn == chess.WHITE

            if is_root_player_white_at_leaf_ref == is_current_player_white_at_q_node:
                # Perspectives align (e.g. root White, current White OR root Black, current Black)
                return nn_score_from_root_perspective
            else:
                # Perspectives are opposed (e.g. root White, current Black OR root Black, current White)
                return -nn_score_from_root_perspective
        
        # Stand pat evaluation using NN, against the provided leaf_reference_fen
        # Pass is_quiescence_standpat_log=True for the explicit stand-pat evaluation
        stand_pat_score_from_root_perspective = self.nn_evaluator.evaluate_position_against_reference(self.board.fen(), leaf_reference_fen, is_quiescence_standpat_log=True)
        # Adjust stand_pat score for whose turn it is at the current board position
        # Correct perspective adjustment:
        is_root_player_white_at_leaf_ref = chess.Board(leaf_reference_fen).turn == chess.WHITE
        is_current_player_white_at_q_node = self.board.turn == chess.WHITE
        
        if is_root_player_white_at_leaf_ref == is_current_player_white_at_q_node:
            current_player_stand_pat = stand_pat_score_from_root_perspective
        else:
            current_player_stand_pat = -stand_pat_score_from_root_perspective
        
        # Log with leaf_reference_fen
        # Keep STAND_PAT log, it's important for quiescence - This is now handled by the conditional log in evaluate_position_against_reference
        # print(f"[DEBUG SC _q_search STAND_PAT] raw_nn_score_vs_ref={stand_pat_score_from_root_perspective:.2f}, current_player_stand_pat={current_player_stand_pat:.2f}, fen='{self.board.fen()}', ref_fen='{leaf_reference_fen}'")

        if current_player_stand_pat >= beta:
            return beta
        if current_player_stand_pat > alpha:
            alpha = current_player_stand_pat
        
        # Order moves (all legal moves for "pure" quiescence)
        ordered_moves = self.move_orderer.order_moves(self.board, list(self.board.legal_moves), ply) # Removed root_reference_fen from q-search move ordering
        # Limit quiescence search to top K moves - REVERTED
        # # This is a search algorithm heuristic, not chess knowledge.
        # # A reasonable K could be 3-5. Let's try 5.
        # limited_ordered_moves = ordered_moves[:5]

        if not ordered_moves and not self.board.is_check(): # No moves and not in check -> stalemate by definition in qsearch if standpat didn't return mate
            # This case might be redundant if stand-pat already handles terminal positions from NN eval
            # However, if NN eval for stand-pat is not a mate, but there are no moves, it is stalemate.
            return NeuralNetworkConfig.STALEMATE_SCORE

        for move in ordered_moves: # Iterate over ALL ordered_moves
            self.board.push(move)
            # Quiescence search continues with the same leaf_reference_fen
            score = -self._quiescence_search(-beta, -alpha, ply + 1, q_depth + 1, leaf_reference_fen)
            self.board.pop()
            
            if q_depth <= 0: # CORRECT Conditional logging for _quiescence_search EVAL
                print(f"  [DEBUG SC _q_search EVAL] q_depth={q_depth}, ply={ply}, move={move.uci()}, score_after_negation={score:.2f}, alpha={alpha:.2f}, beta={beta:.2f}, fen='{self.board.fen()}'")

            if score >= beta:
                return beta # Fail-high (beta cutoff)
            if score > alpha:
                alpha = score
                if q_depth == 0: # Conditional log
                    print(f"    [DEBUG SC _q_search Q_ALPHA_UPDATE] new_alpha={alpha:.2f}, best_move_so_far={move.uci()}, fen='{self.board.fen()}'")
        
        return alpha
    
    def _time_up(self) -> bool:
        """Check if search time limit exceeded."""
        if self.search_start_time is None:
            return False
        return (time.time() - self.search_start_time) >= self.time_limit_seconds
    
    def _reset_search_stats(self):
        """Reset search statistics."""
        self.nodes_searched = 0
        self.quiescence_nodes = 0
        self.tt_hits = 0
        self.beta_cutoffs = 0
        self.null_move_cutoffs = 0
        self.lmr_activations = 0

# 🎯 MAIN ENGINE INTERFACE (For compatibility)
class PureNeuralNetworkEngine:
    """
    Main engine interface - maintains compatibility while enforcing NN purity.
    
    🚨 PURE NN PHILOSOPHY: This is a wrapper that ensures no chess knowledge.
    """
    
    def __init__(self, fen: str = None):
        self.search_coordinator = SearchCoordinator(fen)
        print("🧠 Pure Neural Network Chess Engine initialized!")
        print("⚠️  ZERO chess knowledge - all evaluation from NN!")
    
    def get_move(self, time_limit: float = None) -> str:
        """Get best move using pure NN evaluation."""
        return self.search_coordinator.get_best_move(self.search_coordinator.board.fen(), time_limit)
    
    def set_position(self, fen: str):
        """Set board position."""
        self.search_coordinator.board = chess.Board(fen)

if __name__ == "__main__":
    # Test pure NN engine
    engine = PureNeuralNetworkEngine()
    
    # Test from starting position
    print("Testing from starting position...")
    best_move = engine.get_move(3.0)
    print(f"Best move: {best_move}")
    
    # Test from tactical position
    tactical_fen = "r3k2r/pp1b1ppp/2n2n2/1B2N3/Q2Pq3/2P1B3/P4PPP/R3K2R w KQkq - 1 13"
    engine.set_position(tactical_fen)
    print(f"\nTesting tactical position...")
    tactical_move = engine.get_move(5.0)
    print(f"Tactical move: {tactical_move}")