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
ENGINE_VERSION = "4.0"
ENGINE_NAME = "Neural Chess Engine V4.0 - Pure Neural Power"
ENGINE_FEATURES = [
    "🧠 Pure Neural Network Strength (no opening book)",
    "⚡ Speed-Optimized Alpha-Beta (depths 2-3 like V3)",
    "🎯 Fast Move Ordering (MVV-LVA + Killers)", 
    "💾 Enhanced Transposition Tables",
    "🔍 Quick Null Move & Late Move Reductions",
    "🎲 Smart Move Variety", 
    "🔄 Three-fold Repetition Avoidance",
    "📊 Streamlined Position Evaluation",
    "🚀 Faster than V3 with Advanced Features"
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

# evaluate two input positions, adjusting score based on whose turn it was in the first FEN
def evaluate_pos(first, second):
    x_1, x_2 = make_x(first,second)
    interpreter.set_tensor(input_details[0]['index'], x_1)
    interpreter.set_tensor(input_details[1]['index'], x_2)
    interpreter.invoke()
    raw_evaluation = interpreter.get_tensor(output_details[0]['index'])[0][0]
    
    # Flip score based on whose turn it is in the first position (like original)
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

def is_passed_pawn(board, square, color):
    """Check if a pawn would be passed"""
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    
    opponent_color = chess.WHITE if color == chess.BLACK else chess.BLACK
    
    # Check if any opponent pawns can stop this pawn
    for opponent_square in board.pieces(chess.PAWN, opponent_color):
        opp_file = chess.square_file(opponent_square)
        opp_rank = chess.square_rank(opponent_square)
        
        if abs(opp_file - file) <= 1:
            if color == chess.BLACK and opp_rank <= rank:
                return False
            elif color == chess.WHITE and opp_rank >= rank:
                return False
    
    return True

def has_pawn_support(board, square, color):
    """Check if a pawn has support from other pawns"""
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    
    # Check diagonal support
    support_squares = []
    if color == chess.BLACK:
        if file > 0:
            support_squares.append(chess.square(file - 1, rank + 1))
        if file < 7:
            support_squares.append(chess.square(file + 1, rank + 1))
    else:
        if file > 0:
            support_squares.append(chess.square(file - 1, rank - 1))
        if file < 7:
            support_squares.append(chess.square(file + 1, rank - 1))
    
    for support_square in support_squares:
        piece = board.piece_at(support_square)
        if piece and piece.piece_type == chess.PAWN and piece.color == color:
            return True
    
    return False

class EngineV4:
    def __init__(self, fen, bot_color="black"):
        self.board = chess.Board()
        self.board.set_fen(fen)
        
        # Tournament compatibility
        self.name = "V4.0 Pure Neural Power"
        self.version = "4.0"
        
        # Store what color this bot is playing ("white" or "black")
        self.bot_color = bot_color
        self.playing_as_black = (bot_color == "black")
        
        # Enhanced caching system
        self.transposition_table = {}  # Stores {position: (depth, score, move, node_type)}
        self.nodes_searched = 0
        self.cache_hits = 0
        
        # Advanced move ordering
        self.killer_moves = [[] for _ in range(15)]  # More depth levels
        
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
        print("calculating pure neural strength\n")
        
        # Add current position to history
        current_position = current_fen.split(' ')[0]
        if current_position not in self.position_history:
            self.position_history.append(current_position)
        
        # Clean up tables only when really needed
        if len(self.transposition_table) > 100000:
            self.transposition_table.clear()  # Simple clear for speed
        
        # Iterative deepening optimized for speed vs strength balance
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        
        for depth in range(2, 4):  # Depths 2-3 like V3, but with advanced techniques
            try:
                # Aspiration window only for depth 3+ to save time
                if depth > 2 and best_move:
                    window = 30  # Smaller window for speed
                    alpha = max(float('-inf'), alpha - window)
                    beta = min(float('inf'), beta + window)
                
                move, score = self.alpha_beta_root(depth, alpha, beta)
                if move:
                    best_move = move
                    alpha = score
                    beta = score
                
                print(f"Depth {depth}: {move} (score: {score:.2f}, nodes: {self.nodes_searched})")
                
                # Early exit for very strong positions to save time
                if abs(score) > 800:  # Lower threshold for faster exit
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
        
        # Update history tables
        move_key = str(best_move)
        if move_key in self.killer_moves[0]:
            self.killer_moves[0].remove(move)
            self.killer_moves[0].insert(0, move)
        
        # Update position history
        if best_move != "checkmate":
            try:
                move_obj = chess.Move.from_uci(str(best_move))
                self.update_position_history(move_obj)
            except:
                pass
        
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
        """Fast alpha-beta search optimized for speed"""
        self.nodes_searched += 1
        
        # Terminal node checks
        if self.board.is_checkmate():
            return 10000 - ply if not maximizing_player else -10000 + ply
        
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0
        
        # Simplified transposition table lookup - only for depth 3+
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
        
        # Simplified null move pruning - only for depth 3+
        if (depth >= 3 and not self.board.is_check()):
            # Quick material check instead of complex function
            if len(self.board.pieces(chess.KNIGHT, chess.BLACK)) + len(self.board.pieces(chess.BISHOP, chess.BLACK)) + len(self.board.pieces(chess.ROOK, chess.BLACK)) + len(self.board.pieces(chess.QUEEN, chess.BLACK)) > 0:
                # Try null move with reduced depth
                self.board.push(chess.Move.null())
                null_score = self.alpha_beta(depth - 2, alpha, beta, not maximizing_player, ply + 1)
                self.board.pop()
                
                if not maximizing_player and null_score <= alpha:
                    self.null_move_cutoffs += 1
                    return alpha
                elif maximizing_player and null_score >= beta:
                    self.null_move_cutoffs += 1
                    return beta
        
        # Generate and order moves with simpler ordering
        killer_moves = self.killer_moves[ply] if ply < len(self.killer_moves) else []
        legal_moves = order_moves_v4(self.board, list(self.board.legal_moves), killer_moves)
        
        best_score = float('-inf') if maximizing_player else float('inf')
        best_move = None
        moves_searched = 0
        
        for move in legal_moves:
            moves_searched += 1
            self.board.push(move)
            
            # Simplified Late Move Reductions
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
        
        # Store in transposition table only for depth 3+
        if depth >= 3 and best_move:
            position_key = self.board.fen()
            self.transposition_table[position_key] = (depth, best_score, best_move, "exact")
        
        return best_score
    
    def has_non_pawn_material(self, is_maximizing):
        """Check if side has non-pawn material for null move pruning"""
        color = chess.BLACK if is_maximizing else chess.WHITE
        
        pieces = (self.board.pieces(chess.KNIGHT, color) | 
                 self.board.pieces(chess.BISHOP, color) |
                 self.board.pieces(chess.ROOK, color) |
                 self.board.pieces(chess.QUEEN, color))
        
        return len(pieces) > 0
    
    def quiescence(self, alpha, beta, maximizing_player, depth):
        """Fast quiescence search for tactical stability"""
        # Reduced depth for speed - same as V3
        if depth == 0:
            return self.evaluate_position()
        
        stand_pat = self.evaluate_position()
        
        if maximizing_player:
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
            
            # Only consider good captures (simplified filtering)
            captures = []
            for move in self.board.legal_moves:
                if self.board.piece_at(move.to_square) is not None:
                    # Simplified good capture check
                    captured = self.board.piece_at(move.to_square)
                    attacker = self.board.piece_at(move.from_square)
                    if captured and attacker:
                        # Simple piece values for quick comparison
                        values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                                chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
                        if values.get(captured.piece_type, 0) >= values.get(attacker.piece_type, 1):
                            captures.append(move)
            
            # Limit to best 3 captures for speed (like V3)
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
            
            # Same optimization for minimizing player
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
    
    def see_capture(self, move):
        """Simplified Static Exchange Evaluation"""
        if self.board.piece_at(move.to_square) is None:
            return 0
        
        # Simplified piece values for speed
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100
        }
        
        captured = self.board.piece_at(move.to_square)
        attacker = self.board.piece_at(move.from_square)
        
        if not captured or not attacker:
            return 0
        
        captured_value = piece_values.get(captured.piece_type, 0)
        attacker_value = piece_values.get(attacker.piece_type, 1)
        
        # Simple: gain - potential loss
        return captured_value - attacker_value
    
    def evaluate_position(self):
        """Fast position evaluation optimized for speed"""
        current_fen = self.board.fen()
        
        # Check cache first
        if current_fen in self.transposition_table:
            _, cached_score, _, _ = self.transposition_table[current_fen]
            return cached_score
        
        # Neural network evaluation - evaluate the current position strength
        # Use the same position twice for absolute position evaluation (like V3)
        raw_nn_score = evaluate_pos(current_fen, current_fen)
        
        # Convert to bot's perspective
        # The neural network/evaluate_pos gives score from perspective of side-to-move in current position
        if self.playing_as_black:
            # If we're playing black, use the score as-is (assuming evaluate_pos returns from Black's perspective)
            nn_score = raw_nn_score
        else:
            # If we're playing white, flip the score 
            nn_score = -raw_nn_score
        
        # Simplified and faster positional factors (all from bot's perspective)
        positional_bonus = 0
        
        # Quick check bonus/penalty
        if self.board.is_check():
            if self.playing_as_black:
                positional_bonus += 30 if self.board.turn == chess.BLACK else -30
            else:
                positional_bonus += 30 if self.board.turn == chess.WHITE else -30
        
        # Fast material balance (from bot's perspective)
        material_score = self.quick_material_balance()
        if not self.playing_as_black:
            material_score = -material_score  # Flip for White
        
        # Simplified piece activity (from bot's perspective)
        mobility_score = len(list(self.board.legal_moves))
        if self.playing_as_black:
            # When playing black, more mobility when it's our turn is good
            if self.board.turn == chess.WHITE:
                mobility_score = -mobility_score
        else:
            # When playing white, more mobility when it's our turn is good  
            if self.board.turn == chess.BLACK:
                mobility_score = -mobility_score
        
        # Quick king safety (from bot's perspective)
        king_safety = self.quick_king_safety()
        if not self.playing_as_black:
            king_safety = -king_safety  # Flip for White
        
        # Combine factors with reduced weights for speed
        final_score = (nn_score + 
                      (positional_bonus * 0.08) + 
                      (material_score * 0.04) + 
                      (mobility_score * 0.02) +
                      (king_safety * 0.03))
        
        # Small randomization for move variety
        final_score += random.uniform(-1, 1)
        
        return final_score
    
    def quick_material_balance(self):
        """Fast material calculation from Black's perspective"""
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        
        balance = 0
        for piece_type in piece_values:
            black_count = len(self.board.pieces(piece_type, chess.BLACK))
            white_count = len(self.board.pieces(piece_type, chess.WHITE))
            balance += (black_count - white_count) * piece_values[piece_type]
        
        return balance * 20  # Scale up slightly
    
    def quick_king_safety(self):
        """Fast king safety evaluation"""
        score = 0
        black_king = self.board.king(chess.BLACK)
        if black_king:
            # Castled king bonus
            if black_king in [chess.G8, chess.C8]:
                score += 25
            # King in center penalty
            if chess.square_rank(black_king) < 6:
                score -= 30
        return score
    
    # Anti-repetition methods (carried over from V3.2)
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