#!/usr/bin/env python3
"""
PURE NN CHESS SEARCH COORDINATOR

🚨 CRITICAL PHILOSOPHY: This module handles search logic (alpha-beta, move ordering, time management)
but gets ALL position evaluations from the neural network. NO CHESS KNOWLEDGE.

This module coordinates search but doesn't evaluate positions - that's the NN's job.
"""

import chess
import chess.engine
import time
import numpy as np
from neural_network_inference import NeuralNetworkEvaluator, NeuralNetworkConfig

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

# 🔍 TRANSPOSITION TABLE (Search optimization, no chess knowledge)
class TranspositionTable:
    """Enhanced transposition table for search optimization."""
    
    def __init__(self, size_power: int = 20):
        self.size = 2 ** size_power
        self.table = {}
        self.generation = 0
        self.hits = 0
        self.misses = 0
    
    def new_search(self):
        """Start new search generation."""
        self.generation += 1
    
    def store(self, zobrist_key: np.uint64, depth: int, score: float, best_move: str, flag: str):
        """Store search result."""
        entry = (depth, score, best_move, flag, self.generation)
        self.table[zobrist_key % self.size] = entry
    
    def probe(self, zobrist_key: np.uint64):
        """Probe transposition table."""
        key = zobrist_key % self.size
        if key in self.table:
            self.hits += 1
            return self.table[key]
        else:
            self.misses += 1
            return None

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
        self.killer_moves = {}  # Killer move heuristic
    
    def order_moves(self, board: chess.Board, legal_moves: list, ply: int, tt_best_move: str = None) -> list:
        """
        Order moves using NN evaluation and search heuristics.
        
        🚨 NO CHESS KNOWLEDGE: Uses only NN scores and search optimization.
        """
        if not legal_moves:
            return []
        
        scored_moves = []
        current_fen = board.fen()
        
        # Get NN scores for all moves
        nn_scores = self.nn_evaluator.evaluate_moves_for_ordering(current_fen, legal_moves)
        
        for move in legal_moves:
            score = 0
            
            # 1. Transposition table best move gets highest priority
            if tt_best_move and str(move) == tt_best_move:
                score = 10000000
            
            # 2. NN-guided move scoring (primary method)
            elif move in nn_scores:
                nn_score = nn_scores[move]
                confidence = abs(nn_score - 0.5) * 2.0
                
                # Conservative NN confidence scaling
                if confidence > 0.7:
                    score = 5500000 + int(nn_score * 1000000)
                elif confidence > 0.5:
                    score = 5000000 + int(nn_score * 800000)
                elif confidence > 0.3:
                    score = 4700000 + int(nn_score * 600000)
                else:
                    score = 4500000 + int(nn_score * 400000)
            
            # 3. Killer move heuristic (search optimization)
            elif ply in self.killer_moves and move in self.killer_moves[ply]:
                score = 2000000
            
            # 4. Default score
            else:
                score = 1000000
            
            scored_moves.append((score, move))
        
        # Sort by score (highest first)
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return [move for score, move in scored_moves]
    
    def update_killer_move(self, ply: int, move: chess.Move):
        """Update killer move heuristic."""
        if ply not in self.killer_moves:
            self.killer_moves[ply] = []
        
        if move not in self.killer_moves[ply]:
            self.killer_moves[ply].insert(0, move)
            if len(self.killer_moves[ply]) > 2:
                self.killer_moves[ply].pop()

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
        self.transposition_table = TranspositionTable()
        self.zobrist_hasher = ZobristHasher()
        
        # Set up position
        if starting_fen:
            self.board = chess.Board(starting_fen)
        else:
            self.board = chess.Board()
        
        # Search statistics
        self.nodes_searched = 0
        self.quiescence_nodes = 0
        self.tt_hits = 0
        self.beta_cutoffs = 0
        self.null_move_cutoffs = 0
        self.lmr_activations = 0
        
        # Time management
        self.start_time = None
        self.time_limit = SearchConfig.SEARCH_TIME_BALANCED
        
        print("✅ Pure NN Search Coordinator initialized - ZERO chess knowledge!")
    
    def get_best_move(self, time_limit: float = None) -> str:
        """
        Find best move using pure NN evaluation with alpha-beta search.
        
        🚨 PURE NN: All position evaluation comes from neural network.
        """
        if time_limit:
            self.time_limit = time_limit
        
        self.start_time = time.time()
        self.transposition_table.new_search()
        self._reset_search_stats()
        
        root_fen_for_this_search_cycle = self.board.fen() # Capture current FEN as reference

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
            ordered_moves = self.move_orderer.order_moves(self.board, legal_moves, 0) 
            
            for i, move in enumerate(ordered_moves):
                if self._time_up():
                    break
                
                self.board.push(move)
                # Pass root_fen_for_this_search_cycle as the leaf_reference_fen
                score = -self._alpha_beta(depth - 1, float('-inf'), float('inf'), 1, False, leaf_reference_fen=root_fen_for_this_search_cycle)
                self.board.pop()
                
                if score > current_best_score_this_depth:
                    current_best_score_this_depth = score
                    current_best_move_this_depth = move
            
            if current_best_move_this_depth: # Check if a move was found at this depth
                best_move = current_best_move_this_depth
                best_score = current_best_score_this_depth
                
                # Store in transposition table
                # Key is for the root position of this get_best_move call
                zobrist_key = self.zobrist_hasher.hash_position(chess.Board(root_fen_for_this_search_cycle))
                self.transposition_table.store(zobrist_key, depth, best_score, str(best_move), "EXACT") # TT stores info about root_fen_for_this_search_cycle
                
                print(f"    Depth {depth}: {best_move} (NN eval: {best_score:.1f})")
        
        elapsed = time.time() - self.start_time
        nps = int(self.nodes_searched / elapsed) if elapsed > 0 else 0
        print(f"🏁 Search complete: {self.nodes_searched} nodes in {elapsed:.1f}s ({nps} NPS)")
        
        return str(best_move) if best_move else str(legal_moves[0]) # Fallback if no move found
    
    def _alpha_beta(self, depth: int, alpha: float, beta: float, ply: int, is_pv_node: bool, leaf_reference_fen: str) -> float:
        """
        Alpha-beta search using pure NN evaluation.
        
        🚨 NO CHESS KNOWLEDGE: Only uses NN for position evaluation.
        """
        self.nodes_searched += 1
        initial_alpha = alpha # Preserve the original alpha for TT flag
        
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
        
        # Transposition table probe
        zobrist_key = self.zobrist_hasher.hash_position(self.board)
        tt_entry = self.transposition_table.probe(zobrist_key)
        tt_best_move = None
        
        if tt_entry:
            tt_depth, tt_score, tt_move, tt_flag, tt_generation = tt_entry
            # Ensure entry is from current search generation if TT generations are implemented fully
            # For now, assume tt_generation check is implicitly handled or not strictly needed for this change
            if tt_depth >= depth:
                self.tt_hits += 1
                if tt_flag == "EXACT":
                    return tt_score
                elif tt_flag == "LOWERBOUND":
                    alpha = max(alpha, tt_score)
                elif tt_flag == "UPPERBOUND":
                    beta = min(beta, tt_score)
                
                if alpha >= beta:
                    return tt_score 
                tt_best_move = tt_move
        
        # Reach depth limit - use quiescence or NN evaluation
        if depth <= 0:
            return self._quiescence_search(alpha, beta, ply, 0, leaf_reference_fen)
        
        # Null move pruning (search optimization)
        if (depth >= 3 and not self.board.is_check() and ply > 0): # ply > 0 to avoid null move at root
            # Make sure self.board allows null moves (e.g. not in zugzwang, though NN engine doesn't know zugzwang)
            # python-chess board.push(chess.Move.null()) handles legality.
            try: # Protect against board states where null move is illegal (should be rare unless in check, already handled)
                self.board.push(chess.Move.null())
                null_score = -self._alpha_beta(depth - 1 - SearchConfig.NULL_MOVE_REDUCTION, -beta, -alpha, ply + 1, False, leaf_reference_fen)
                self.board.pop()
                
                if null_score >= beta:
                    self.null_move_cutoffs += 1
                    # Storing to TT for null move cutoffs (key is for position *before* null move)
                    # self.transposition_table.store(zobrist_key, depth, beta, None, "LOWERBOUND") # Example
                    return beta 
            except (AssertionError, ValueError): # Catch errors if null move is illegal on current board state
                pass # Skip null move if illegal

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
            
            reduction = 0
            if (moves_tried >= SearchConfig.LMR_MIN_MOVES and 
                depth >= 3 and 
                not self.board.is_check() and # is_check on the board *after* the move
                not self.board.is_capture(move) and # LMR usually not for captures
                not move.promotion): # LMR usually not for promotions
                reduction = SearchConfig.LATE_MOVE_REDUCTION
                self.lmr_activations += 1
            
            current_node_score = 0
            if moves_tried == 0 or not is_pv_node: 
                current_node_score = -self._alpha_beta(depth - 1 - reduction, -beta, -alpha, ply + 1, False, leaf_reference_fen)
            else:
                current_node_score = -self._alpha_beta(depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, False, leaf_reference_fen)
                if current_node_score > alpha and current_node_score < beta: 
                    current_node_score = -self._alpha_beta(depth - 1 - reduction, -beta, -alpha, ply + 1, True, leaf_reference_fen)
            
            self.board.pop()
            moves_tried += 1
            
            if current_node_score > best_score_found_in_loop:
                best_score_found_in_loop = current_node_score
                best_move_for_tt = move 
            
            if current_node_score > alpha:
                alpha = current_node_score
            
            if alpha >= beta:
                self.beta_cutoffs += 1
                if not move.drop: # chess.Move.drop is for Chess960, null move is chess.Move.null()
                    self.move_orderer.update_killer_move(ply, move)
                # TT store on beta cutoff: value is beta, flag is LOWERBOUND
                # Key is for the position *before* this set of moves was tried
                self.transposition_table.store(zobrist_key, depth, beta, str(move) if move else None, "LOWERBOUND")
                return beta 
        
        # Store in transposition table
        if best_move_for_tt: 
            flag = ""
            if best_score_found_in_loop <= initial_alpha: 
                flag = "UPPERBOUND"
            elif best_score_found_in_loop >= beta: 
                flag = "LOWERBOUND" 
            else: 
                flag = "EXACT"
            self.transposition_table.store(zobrist_key, depth, best_score_found_in_loop, str(best_move_for_tt), flag)

        return best_score_found_in_loop # Return the actual best score found
    
    def _quiescence_search(self, alpha: float, beta: float, ply: int, q_depth: int, leaf_reference_fen: str) -> float:
        """
        Quiescence search using NN evaluation.
        
        🚨 PURE NN: Uses only neural network for position evaluation.
        """
        self.quiescence_nodes += 1
        
        # Terminal check inside quiescence
        outcome = self.board.outcome()
        if outcome:
            if outcome.winner == chess.WHITE: return NeuralNetworkConfig.CHECKMATE_SCORE - ply
            if outcome.winner == chess.BLACK: return -NeuralNetworkConfig.CHECKMATE_SCORE + ply
            return NeuralNetworkConfig.STALEMATE_SCORE

        if q_depth >= SearchConfig.QUIESCENCE_MAX_DEPTH:
            # Use NN evaluation against the provided leaf_reference_fen
            nn_score = self.nn_evaluator.evaluate_position_against_reference(self.board.fen(), reference_fen=leaf_reference_fen)
            return nn_score if self.board.turn == chess.WHITE else -nn_score # Adjust for current player
        
        # Stand pat evaluation using NN, against the provided leaf_reference_fen
        stand_pat = self.nn_evaluator.evaluate_position_against_reference(self.board.fen(), reference_fen=leaf_reference_fen)
        # Adjust stand_pat score for whose turn it is at the current board position
        current_player_stand_pat = stand_pat if self.board.turn == chess.WHITE else -stand_pat
        
        if current_player_stand_pat >= beta:
            return beta
        if current_player_stand_pat > alpha:
            alpha = current_player_stand_pat
        
        # Evaluate all legal moves in quiescence, relying on NN evaluation
        legal_moves = list(self.board.legal_moves) 
        if not legal_moves and not outcome: # Double check for stalemate if outcome missed it
             return NeuralNetworkConfig.STALEMATE_SCORE

        # Move ordering for quiescence could be beneficial, e.g. check-evasions, checks, then others.
        # For now, iterate through them as python-chess provides them.
        # A simple heuristic might be to try checks first if any:
        # ordered_q_moves = sorted(legal_moves, key=lambda m: self.board.gives_check(m), reverse=True)
        # For now, using legal_moves directly.
        
        for move in legal_moves: 
            if self._time_up(): # Check time limit inside the loop
                break 
            
            self.board.push(move)
            score = -self._quiescence_search(-beta, -alpha, ply + 1, q_depth + 1, leaf_reference_fen)
            self.board.pop()
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        
        return alpha
    
    def _time_up(self) -> bool:
        """Check if search time limit exceeded."""
        if self.start_time is None:
            return False
        return (time.time() - self.start_time) >= self.time_limit
    
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
        return self.search_coordinator.get_best_move(time_limit)
    
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