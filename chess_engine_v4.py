import chess
import numpy as np
import tensorflow as tf
import random
from util import *

# Engine Version Information
ENGINE_VERSION = "4.0"
ENGINE_NAME = "Neural Chess Engine V4.0 - Pure Neural Power"
ENGINE_FEATURES = [
    "🧠 Pure Neural Network Strength (no opening book)",
    "⚡ Advanced Alpha-Beta Pruning with Null Move",
    "🎯 Sophisticated Move Ordering (MVV-LVA + History)",
    "💾 Enhanced Transposition Tables with Depth",
    "🔍 Late Move Reductions for Speed",
    "🎲 Smart Randomization for Variety", 
    "🔄 Three-fold Repetition Avoidance",
    "📊 Multi-layer Position Evaluation",
    "🚀 Optimized for Speed + Strength"
]

# load model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
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

def order_moves_v4(board, moves, killer_moves=None, history_table=None):
    """V4 Advanced move ordering with sophisticated scoring"""
    move_scores = []
    
    if killer_moves is None:
        killer_moves = []
    if history_table is None:
        history_table = {}
    
    for move in moves:
        score = 0
        
        # Killer moves get highest priority
        if move in killer_moves:
            score += 50000
            move_scores.append((score, move))
            continue
        
        # Advanced capture evaluation (MVV-LVA)
        if board.piece_at(move.to_square) is not None:
            captured_piece = board.piece_at(move.to_square)
            moving_piece = board.piece_at(move.from_square)
            
            piece_values = {
                chess.PAWN: 100,
                chess.KNIGHT: 320,
                chess.BISHOP: 330,
                chess.ROOK: 500,
                chess.QUEEN: 900,
                chess.KING: 20000
            }
            
            if captured_piece and moving_piece:
                victim_value = piece_values.get(captured_piece.piece_type, 0)
                attacker_value = piece_values.get(moving_piece.piece_type, 100)
                # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
                score += (victim_value * 10) - (attacker_value // 10)
        
        # Promotion bonuses
        if move.promotion:
            if move.promotion == chess.QUEEN:
                score += 9000
            elif move.promotion == chess.ROOK:
                score += 5000
            else:
                score += 3000
        
        # Check bonus (more sophisticated)
        board.push(move)
        if board.is_check():
            score += 3000
            if board.is_checkmate():
                score += 100000  # Checkmate is ultimate goal
        board.pop()
        
        # Castling safety bonus
        if board.is_castling(move):
            score += 500
        
        # Central control with piece-specific bonuses
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        extended_center = [chess.C3, chess.C4, chess.C5, chess.C6, 
                          chess.F3, chess.F4, chess.F5, chess.F6]
        
        if move.to_square in center_squares:
            score += 200
        elif move.to_square in extended_center:
            score += 100
        
        # Advanced piece development
        if board.piece_at(move.from_square):
            piece = board.piece_at(move.from_square)
            if piece.color == chess.BLACK:
                from_rank = chess.square_rank(move.from_square)
                to_rank = chess.square_rank(move.to_square)
                
                # Development bonuses
                if from_rank == 7 and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                    score += 300  # Strong development bonus
                
                # Forward progress
                if to_rank < from_rank:
                    score += 50
                
                # Piece-specific positioning
                if piece.piece_type == chess.KNIGHT:
                    # Knights love outposts
                    knight_outposts = [chess.C6, chess.D6, chess.E6, chess.F6, chess.C5, chess.D5, chess.E5, chess.F5]
                    if move.to_square in knight_outposts:
                        score += 150
                elif piece.piece_type == chess.BISHOP:
                    # Bishops love long diagonals
                    if move.to_square in [chess.B7, chess.G7, chess.B2, chess.G2]:
                        score += 100
                elif piece.piece_type == chess.QUEEN:
                    # Queen activity but not too early
                    if to_rank < 6:  # Don't bring queen out too early
                        score -= 50
        
        # History table bonus (moves that were good before)
        move_key = str(move)
        if move_key in history_table:
            score += min(history_table[move_key], 1000)  # Cap history bonus
        
        # Pawn structure considerations
        if board.piece_at(move.from_square) and board.piece_at(move.from_square).piece_type == chess.PAWN:
            # Passed pawn advance
            if is_passed_pawn(board, move.to_square, chess.BLACK):
                score += 200
            
            # Pawn chains
            if has_pawn_support(board, move.to_square, chess.BLACK):
                score += 100
        
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
    def __init__(self, fen):
        self.board = chess.Board()
        self.board.set_fen(fen)
        
        # Tournament compatibility
        self.name = "V4.0 Pure Neural Power"
        self.version = "4.0"
        
        # Enhanced caching system
        self.transposition_table = {}  # Stores {position: (depth, score, move, node_type)}
        self.nodes_searched = 0
        self.cache_hits = 0
        
        # Advanced move ordering
        self.killer_moves = [[] for _ in range(15)]  # More depth levels
        self.history_table = {}  # Track good moves
        
        # Anti-repetition
        self.position_history = []
        self.initial_fen = fen
        
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
        
        # Clean up tables periodically
        if len(self.transposition_table) > 50000:
            # Keep only recent entries
            items = list(self.transposition_table.items())
            self.transposition_table = dict(items[-25000:])
        
        # Iterative deepening with aspiration windows
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        
        for depth in range(1, 5):  # Depths 1-4 for thorough search
            try:
                # Aspiration window for deeper searches
                if depth > 2 and best_move:
                    window = 50
                    alpha = max(float('-inf'), alpha - window)
                    beta = min(float('inf'), beta + window)
                
                move, score = self.alpha_beta_root(depth, alpha, beta)
                if move:
                    best_move = move
                    alpha = score
                    beta = score
                
                print(f"Depth {depth}: {move} (score: {score:.2f}, nodes: {self.nodes_searched})")
                
                # Early exit for forced wins/losses
                if abs(score) > 900:  # Likely checkmate
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
        print(f"⚡ optimizations: {self.null_move_cutoffs} null cuts, {self.late_move_reductions} LMR")
        
        # Update history tables
        move_key = str(best_move)
        if move_key in self.history_table:
            self.history_table[move_key] += 10
        else:
            self.history_table[move_key] = 10
        
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
        legal_moves = order_moves_v4(self.board, list(self.board.legal_moves), 
                                   killer_moves, self.history_table)
        
        if not legal_moves:
            return None, float('-inf')
        
        best_move = None
        best_score = float('-inf')
        moves_searched = 0
        
        for move in legal_moves:
            moves_searched += 1
            print("." if moves_searched % 10 != 0 else moves_searched)
            
            # Repetition avoidance
            repetition_penalty = 0
            if self.would_cause_repetition(move):
                current_eval = self.evaluate_position()
                if current_eval > 50:  # We have advantage
                    repetition_penalty = -300
                    print(f"⚠️  Repetition penalty for {move}")
            
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
        
        return best_move, best_score
    
    def alpha_beta(self, depth, alpha, beta, maximizing_player, ply):
        """Enhanced alpha-beta with advanced pruning techniques"""
        self.nodes_searched += 1
        
        # Terminal node checks
        if self.board.is_checkmate():
            return 10000 - ply if not maximizing_player else -10000 + ply
        
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0
        
        # Transposition table lookup
        position_key = self.board.fen()
        if position_key in self.transposition_table:
            stored_depth, stored_score, stored_move, node_type = self.transposition_table[position_key]
            if stored_depth >= depth:
                self.cache_hits += 1
                return stored_score
        
        # Quiescence search at leaf nodes
        if depth == 0:
            return self.quiescence(alpha, beta, maximizing_player, 3)
        
        # Null move pruning for speed
        if (depth >= 3 and not self.board.is_check() and 
            self.has_non_pawn_material(maximizing_player)):
            
            # Try null move
            self.board.push(chess.Move.null())
            null_score = self.alpha_beta(depth - 3, alpha, beta, not maximizing_player, ply + 1)
            self.board.pop()
            
            if not maximizing_player and null_score <= alpha:
                self.null_move_cutoffs += 1
                return alpha
            elif maximizing_player and null_score >= beta:
                self.null_move_cutoffs += 1
                return beta
        
        # Generate and order moves
        killer_moves = self.killer_moves[ply] if ply < len(self.killer_moves) else []
        legal_moves = order_moves_v4(self.board, list(self.board.legal_moves), 
                                   killer_moves, self.history_table)
        
        best_score = float('-inf') if maximizing_player else float('inf')
        best_move = None
        moves_searched = 0
        
        for move in legal_moves:
            moves_searched += 1
            self.board.push(move)
            
            # Late Move Reductions (LMR)
            reduction = 0
            if (moves_searched > 4 and depth >= 3 and 
                not self.board.is_check() and 
                self.board.piece_at(move.to_square) is None):  # Non-capture
                reduction = 1
                self.late_move_reductions += 1
            
            # Search with possible reduction
            if maximizing_player:
                score = self.alpha_beta(depth - 1 - reduction, alpha, beta, False, ply + 1)
                
                # Re-search if reduced search failed high
                if reduction > 0 and score > alpha:
                    score = self.alpha_beta(depth - 1, alpha, beta, False, ply + 1)
                
                if score > best_score:
                    best_score = score
                    best_move = move
                
                alpha = max(alpha, score)
            else:
                score = self.alpha_beta(depth - 1 - reduction, alpha, beta, True, ply + 1)
                
                if reduction > 0 and score < beta:
                    score = self.alpha_beta(depth - 1, alpha, beta, True, ply + 1)
                
                if score < best_score:
                    best_score = score
                    best_move = move
                
                beta = min(beta, score)
            
            self.board.pop()
            
            if beta <= alpha:
                # Update killer moves
                if (ply < len(self.killer_moves) and 
                    move not in self.killer_moves[ply] and
                    self.board.piece_at(move.to_square) is None):  # Quiet move
                    
                    if len(self.killer_moves[ply]) >= 3:
                        self.killer_moves[ply].pop()
                    self.killer_moves[ply].insert(0, move)
                
                break
        
        # Store in transposition table
        if depth >= 2:  # Only store deeper searches
            node_type = "exact"
            self.transposition_table[position_key] = (depth, best_score, best_move, node_type)
        
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
        """Enhanced quiescence search"""
        if depth == 0:
            return self.evaluate_position()
        
        stand_pat = self.evaluate_position()
        
        if maximizing_player:
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
            
            # Only consider good captures and checks
            moves = []
            for move in self.board.legal_moves:
                if self.board.piece_at(move.to_square) is not None:
                    # Use SEE (Static Exchange Evaluation) approximation
                    if self.see_capture(move) >= 0:  # Only good/equal captures
                        moves.append(move)
                elif self.board.is_check():
                    moves.append(move)  # Include checks
            
            # Order captures by value
            moves = order_moves_v4(self.board, moves)[:5]  # Top 5 moves only
            
            for move in moves:
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
            
            moves = []
            for move in self.board.legal_moves:
                if self.board.piece_at(move.to_square) is not None:
                    if self.see_capture(move) >= 0:
                        moves.append(move)
                elif self.board.is_check():
                    moves.append(move)
            
            moves = order_moves_v4(self.board, moves)[:5]
            
            for move in moves:
                self.board.push(move)
                score = self.quiescence(alpha, beta, True, depth - 1)
                self.board.pop()
                
                if score <= alpha:
                    return alpha
                beta = min(beta, score)
            
            return beta
    
    def see_capture(self, move):
        """Simple Static Exchange Evaluation for captures"""
        if self.board.piece_at(move.to_square) is None:
            return 0
        
        piece_values = {
            chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000
        }
        
        captured = self.board.piece_at(move.to_square)
        attacker = self.board.piece_at(move.from_square)
        
        if not captured or not attacker:
            return 0
        
        captured_value = piece_values.get(captured.piece_type, 0)
        attacker_value = piece_values.get(attacker.piece_type, 100)
        
        # Simple approximation: gain - potential loss
        return captured_value - attacker_value
    
    def evaluate_position(self):
        """Enhanced position evaluation"""
        current_fen = self.board.fen()
        
        # Check cache first
        if current_fen in self.transposition_table:
            _, cached_score, _, _ = self.transposition_table[current_fen]
            return cached_score
        
        # Neural network evaluation
        original_fen = self.board.fen()
        nn_score = evaluate_pos(original_fen, current_fen)
        
        # Enhanced positional factors
        positional_bonus = 0
        
        # Check bonus/penalty
        if self.board.is_check():
            positional_bonus += 25 if self.board.turn == chess.BLACK else -25
        
        # Material balance
        material_score = self.calculate_material_balance()
        
        # Advanced positional evaluation
        piece_activity = self.evaluate_piece_activity()
        king_safety = self.evaluate_king_safety()
        pawn_structure = self.evaluate_pawn_structure()
        
        # Combine all factors
        final_score = (nn_score + 
                      (positional_bonus * 0.1) + 
                      (material_score * 0.05) + 
                      (piece_activity * 0.08) +
                      (king_safety * 0.06) +
                      (pawn_structure * 0.04))
        
        # Add small randomization for variety
        final_score += random.uniform(-1, 1)
        
        return final_score
    
    def calculate_material_balance(self):
        """Calculate material advantage"""
        piece_values = {
            chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900
        }
        
        balance = 0
        for piece_type in piece_values:
            black_count = len(self.board.pieces(piece_type, chess.BLACK))
            white_count = len(self.board.pieces(piece_type, chess.WHITE))
            balance += (black_count - white_count) * piece_values[piece_type]
        
        return balance
    
    def evaluate_piece_activity(self):
        """Evaluate piece activity and mobility"""
        score = 0
        
        # Count legal moves (mobility)
        current_turn = self.board.turn
        
        # Black mobility
        if current_turn == chess.BLACK:
            black_mobility = len(list(self.board.legal_moves))
        else:
            self.board.turn = chess.BLACK
            black_mobility = len(list(self.board.legal_moves))
            self.board.turn = chess.WHITE
        
        # White mobility  
        if current_turn == chess.WHITE:
            white_mobility = len(list(self.board.legal_moves))
        else:
            self.board.turn = chess.WHITE
            white_mobility = len(list(self.board.legal_moves))
            self.board.turn = chess.BLACK
        
        # Restore turn
        self.board.turn = current_turn
        
        score += (black_mobility - white_mobility) * 2
        
        # Center control
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        for square in center_squares:
            piece = self.board.piece_at(square)
            if piece:
                score += 20 if piece.color == chess.BLACK else -20
        
        return score
    
    def evaluate_king_safety(self):
        """Evaluate king safety"""
        score = 0
        
        black_king = self.board.king(chess.BLACK)
        if black_king:
            # Castled king is safer
            if black_king in [chess.G8, chess.C8]:
                score += 30
            
            # King in center is dangerous
            if chess.square_rank(black_king) < 6:
                score -= 40
        
        return score
    
    def evaluate_pawn_structure(self):
        """Evaluate pawn structure"""
        score = 0
        
        black_pawns = self.board.pieces(chess.PAWN, chess.BLACK)
        
        for pawn_square in black_pawns:
            # Passed pawn bonus
            if is_passed_pawn(self.board, pawn_square, chess.BLACK):
                rank = chess.square_rank(pawn_square)
                score += 20 + (7 - rank) * 10  # More valuable closer to promotion
            
            # Doubled pawn penalty
            file = chess.square_file(pawn_square)
            same_file_pawns = [p for p in black_pawns if chess.square_file(p) == file]
            if len(same_file_pawns) > 1:
                score -= 15
        
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