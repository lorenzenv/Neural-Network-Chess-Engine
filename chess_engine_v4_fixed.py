import chess
import numpy as np
import random
from util import *

# TensorFlow Lite compatibility layer
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite

# Engine Version Information
ENGINE_VERSION = "4.2"
ENGINE_NAME = "Neural Chess Engine V4.2 - Fixed Evaluation"
ENGINE_FEATURES = [
    "🧠 Pure Neural Network Strength (no opening book)",
    "⚡ Enhanced Alpha-Beta (depths 2-5 for stronger play)",
    "🎯 Fast Move Ordering (MVV-LVA + Killers)", 
    "💾 Enhanced Transposition Tables",
    "🔍 Quick Null Move & Late Move Reductions",
    "🎲 Smart Move Variety", 
    "🔄 Three-fold Repetition Avoidance",
    "📊 FIXED Position Evaluation Logic",
    "🚀 Fixed side-to-play evaluation bugs"
]

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

# FIXED: Use the original working evaluation logic
def evaluate_pos(first, second):
    x_1, x_2 = make_x(first,second)
    interpreter.set_tensor(input_details[0]['index'], x_1)
    interpreter.set_tensor(input_details[1]['index'], x_2)
    interpreter.invoke()
    raw_evaluation = interpreter.get_tensor(output_details[0]['index'])[0][0]
    
    # Original working logic: just flip based on whose turn it is in first position
    board = chess.Board(first)
    if board.turn == chess.BLACK:
        return -raw_evaluation
    return raw_evaluation

def order_moves_v4(board, moves, killer_moves=None):
    """V4 Fast move ordering optimized for speed"""
    move_scores = []
    
    if killer_moves is None:
        killer_moves = []
    
    for move in moves:
        score = 0
        
        # Killer moves get highest priority
        if move in killer_moves:
            score += 50000
            move_scores.append((score, move))
            continue
        
        # Advanced capture evaluation (MVV-LVA) - simplified for speed
        if board.piece_at(move.to_square) is not None:
            captured_piece = board.piece_at(move.to_square)
            moving_piece = board.piece_at(move.from_square)
            
            # Simplified piece values for speed
            piece_values = {
                chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100
            }
            
            if captured_piece and moving_piece:
                victim_value = piece_values.get(captured_piece.piece_type, 0)
                attacker_value = piece_values.get(moving_piece.piece_type, 1)
                # MVV-LVA simplified
                score += (victim_value * 1000) - (attacker_value * 10)
        
        # Promotion bonuses
        if move.promotion:
            if move.promotion == chess.QUEEN:
                score += 8000
            else:
                score += 2000
        
        # Quick check bonus
        board.push(move)
        if board.is_check():
            score += 2000
            if board.is_checkmate():
                score += 50000
        board.pop()
        
        # Castling safety bonus
        if board.is_castling(move):
            score += 300
        
        # Central control - simplified
        if move.to_square in [chess.E4, chess.E5, chess.D4, chess.D5]:
            score += 100
        elif move.to_square in [chess.C3, chess.C4, chess.C5, chess.C6, 
                              chess.F3, chess.F4, chess.F5, chess.F6]:
            score += 50
        
        # Basic development bonus - simplified
        if board.piece_at(move.from_square):
            piece = board.piece_at(move.from_square)
            if piece.color == chess.BLACK:
                from_rank = chess.square_rank(move.from_square)
                to_rank = chess.square_rank(move.to_square)
                
                # Simple development bonus
                if from_rank == 7 and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                    score += 150
                
                # Forward progress
                if to_rank < from_rank:
                    score += 30
        
        move_scores.append((score, move))
    
    # Sort by score descending
    move_scores.sort(key=lambda x: x[0], reverse=True)
    return [move for score, move in move_scores]

class EngineV4:
    def __init__(self, fen, bot_color="black"):
        self.board = chess.Board()
        self.board.set_fen(fen)
        
        # Tournament compatibility
        self.name = "V4.2 Fixed Evaluation"
        self.version = "4.2"
        
        # FIXED: Determine what color the ENGINE is playing by looking at the position
        # If it's White's turn in the FEN, the engine is playing White
        # If it's Black's turn in the FEN, the engine is playing Black
        self.engine_plays_white = (self.board.turn == chess.WHITE)
        self.engine_plays_black = (self.board.turn == chess.BLACK)
        
        # Enhanced caching system
        self.transposition_table = {}
        self.nodes_searched = 0
        self.cache_hits = 0
        
        # Advanced move ordering
        self.killer_moves = [[] for _ in range(15)]
        
        # Anti-repetition
        self.position_history = []
        
        # Performance tracking
        self.null_move_cutoffs = 0
        self.late_move_reductions = 0

    def get_version_info(self):
        """Return version information about this engine"""
        return {
            "version": ENGINE_VERSION,
            "name": ENGINE_NAME,
            "features": ENGINE_FEATURES
        }

    def get_move(self):
        best_move = self.make_move()
        return str(best_move)

    def make_move(self):
        self.nodes_searched = 0
        self.cache_hits = 0
        self.null_move_cutoffs = 0
        self.late_move_reductions = 0
        
        current_fen = self.board.fen()
        print("calculating with FIXED evaluation logic\n")
        
        # Add current position to history
        current_position = current_fen.split(' ')[0]
        if current_position not in self.position_history:
            self.position_history.append(current_position)
        
        # Clean up tables only when really needed
        if len(self.transposition_table) > 100000:
            self.transposition_table.clear()
        
        # Iterative deepening
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        
        for depth in range(2, 5):  # Depths 2-5 for stronger play
            try:
                # Aspiration window only for depth 3+
                if depth > 2 and best_move:
                    window = 30
                    alpha = max(float('-inf'), alpha - window)
                    beta = min(float('inf'), beta + window)
                
                move, score = self.alpha_beta_root(depth, alpha, beta)
                if move:
                    best_move = move
                    alpha = score
                    beta = score
                
                print(f"Depth {depth}: {move} (score: {score:.2f}, nodes: {self.nodes_searched})")
                
                # Early exit for very strong positions
                if abs(score) > 1500:
                    break
                    
            except Exception as e:
                print(f"Search error at depth {depth}: {e}")
                break
        
        if best_move is None:
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                best_move = legal_moves[0]
            else:
                return "checkmate"
        
        print(f"🎯 best move: {best_move}")
        print(f"📊 stats: {self.nodes_searched:,} nodes, {self.cache_hits} cache hits")
        if self.null_move_cutoffs > 0 or self.late_move_reductions > 0:
            print(f"⚡ optimizations: {self.null_move_cutoffs} null cuts, {self.late_move_reductions} LMR")
        
        return str(best_move)
    
    def alpha_beta_root(self, depth, alpha, beta):
        """Enhanced root search with advanced techniques"""
        killer_moves = self.killer_moves[0] if depth < len(self.killer_moves) else []
        legal_moves = order_moves_v4(self.board, list(self.board.legal_moves), killer_moves)
        
        if not legal_moves:
            return None, float('-inf')
        
        best_move = None
        best_score = float('-inf')
        moves_searched = 0
        
        for move in legal_moves:
            moves_searched += 1
            if moves_searched % 10 == 0:
                print(moves_searched, end="")
            else:
                print(".", end="")
            
            # Repetition avoidance
            repetition_penalty = 0
            if self.would_cause_repetition(move):
                current_eval = self.evaluate_position()
                if current_eval > 50:  # We have advantage
                    repetition_penalty = -300
            
            self.board.push(move)
            
            if self.board.is_checkmate():
                self.board.pop()
                return move, 10000
            
            # Search with different techniques based on move number
            if moves_searched == 1:
                # Search first move with full window
                score = self.alpha_beta(depth - 1, alpha, beta, False, 1)
            else:
                # Try to prove other moves are worse (PVS)
                score = self.alpha_beta(depth - 1, alpha, alpha + 1, False, 1)
                if alpha < score < beta:
                    # Re-search with full window
                    score = self.alpha_beta(depth - 1, alpha, beta, False, 1)
            
            score += repetition_penalty
            self.board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
                
                # Update killer moves
                if depth < len(self.killer_moves) and move not in self.killer_moves[0]:
                    if len(self.killer_moves[0]) >= 3:
                        self.killer_moves[0].pop()
                    self.killer_moves[0].insert(0, move)
            
            alpha = max(alpha, score)
            if beta <= alpha:
                break
        
        # Store in transposition table only for depth 3+
        if depth >= 3 and best_move:
            position_key = self.board.fen()
            self.transposition_table[position_key] = (depth, best_score, best_move, "exact")
        
        return best_move, best_score
    
    def alpha_beta(self, depth, alpha, beta, maximizing_player, ply):
        """Alpha-beta search with correct evaluation"""
        self.nodes_searched += 1
        
        # Terminal node checks
        if self.board.is_checkmate():
            return 10000 - ply if not maximizing_player else -10000 + ply
        
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0
        
        # Transposition table lookup
        if depth >= 3:
            position_key = self.board.fen()
            if position_key in self.transposition_table:
                stored_depth, stored_score, stored_move, node_type = self.transposition_table[position_key]
                if stored_depth >= depth:
                    self.cache_hits += 1
                    return stored_score
        
        # Quiescence search at leaf nodes
        if depth == 0:
            return self.quiescence(alpha, beta, maximizing_player, 2)
        
        # Null move pruning
        if (depth >= 3 and not self.board.is_check()):
            if len(self.board.pieces(chess.KNIGHT, chess.BLACK)) + len(self.board.pieces(chess.BISHOP, chess.BLACK)) + len(self.board.pieces(chess.ROOK, chess.BLACK)) + len(self.board.pieces(chess.QUEEN, chess.BLACK)) > 0:
                self.board.push(chess.Move.null())
                null_score = self.alpha_beta(depth - 2, alpha, beta, not maximizing_player, ply + 1)
                self.board.pop()
                
                if not maximizing_player and null_score <= alpha:
                    self.null_move_cutoffs += 1
                    return alpha
                elif maximizing_player and null_score >= beta:
                    self.null_move_cutoffs += 1
                    return beta
        
        # Generate and order moves
        killer_moves = self.killer_moves[ply] if ply < len(self.killer_moves) else []
        legal_moves = order_moves_v4(self.board, list(self.board.legal_moves), killer_moves)
        
        best_score = float('-inf') if maximizing_player else float('inf')
        best_move = None
        moves_searched = 0
        
        for move in legal_moves:
            moves_searched += 1
            self.board.push(move)
            
            # Late Move Reductions
            reduction = 0
            if (moves_searched > 3 and depth >= 2 and 
                not self.board.is_check() and 
                self.board.piece_at(move.to_square) is None):
                reduction = 1
                self.late_move_reductions += 1
            
            # Search with possible reduction
            if maximizing_player:
                score = self.alpha_beta(depth - 1 - reduction, alpha, beta, False, ply + 1)
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, score)
            else:
                score = self.alpha_beta(depth - 1 - reduction, alpha, beta, True, ply + 1)
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, score)
            
            self.board.pop()
            
            if beta <= alpha:
                # Update killer moves only for cutoffs
                if ply < len(self.killer_moves) and move not in self.killer_moves[ply]:
                    if len(self.killer_moves[ply]) >= 2:
                        self.killer_moves[ply].pop()
                    self.killer_moves[ply].insert(0, move)
                break
        
        # Store in transposition table
        if depth >= 3 and best_move:
            position_key = self.board.fen()
            self.transposition_table[position_key] = (depth, best_score, best_move, "exact")
        
        return best_score
    
    def quiescence(self, alpha, beta, maximizing_player, depth):
        """Quiescence search for tactical stability"""
        if depth == 0:
            return self.evaluate_position()
        
        stand_pat = self.evaluate_position()
        
        if maximizing_player:
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
            
            # Only consider good captures
            captures = []
            for move in self.board.legal_moves:
                if self.board.piece_at(move.to_square) is not None:
                    captured = self.board.piece_at(move.to_square)
                    attacker = self.board.piece_at(move.from_square)
                    if captured and attacker:
                        values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                                chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
                        if values.get(captured.piece_type, 0) >= values.get(attacker.piece_type, 1):
                            captures.append(move)
            
            if len(captures) > 3:
                captures = captures[:3]
            
            for move in captures:
                self.board.push(move)
                score = self.quiescence(alpha, beta, False, depth - 1)
                self.board.pop()
                
                if score >= beta:
                    return beta
                alpha = max(alpha, score)
            
            return alpha
        else:
            if stand_pat <= alpha:
                return alpha
            beta = min(beta, stand_pat)
            
            captures = []
            for move in self.board.legal_moves:
                if self.board.piece_at(move.to_square) is not None:
                    captured = self.board.piece_at(move.to_square)
                    attacker = self.board.piece_at(move.from_square)
                    if captured and attacker:
                        values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                                chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
                        if values.get(captured.piece_type, 0) >= values.get(attacker.piece_type, 1):
                            captures.append(move)
            
            if len(captures) > 3:
                captures = captures[:3]
            
            for move in captures:
                self.board.push(move)
                score = self.quiescence(alpha, beta, True, depth - 1)
                self.board.pop()
                
                if score <= alpha:
                    return alpha
                beta = min(beta, score)
            
            return beta
    
    def evaluate_position(self):
        """FIXED: Simple position evaluation using original working logic"""
        current_fen = self.board.fen()
        
        # Use the original working evaluation: just evaluate the position as-is
        # The evaluate_pos function already handles the side-to-move correctly
        raw_score = evaluate_pos(current_fen, current_fen)
        
        # Small randomization for move variety
        raw_score += random.uniform(-0.5, 0.5)
        
        return raw_score
    
    # Anti-repetition methods
    def would_cause_repetition(self, move):
        """Check if a move would cause position repetition"""
        self.board.push(move)
        current_position = self.board.fen().split(' ')[0]
        self.board.pop()
        
        position_count = self.position_history.count(current_position)
        return position_count >= 1
    
    def update_position_history(self, move):
        """Update position history after making a move"""
        self.board.push(move)
        current_position = self.board.fen().split(' ')[0]
        self.position_history.append(current_position)
        
        # Keep manageable history
        if len(self.position_history) > 30:
            self.position_history = self.position_history[-30:]
        
        self.board.pop() 