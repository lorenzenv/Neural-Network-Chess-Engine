#!/usr/bin/env python3
"""
🏆 NEURAL CHESS ENGINE TOURNAMENT
Let different engine versions battle to find the strongest one!
"""

import chess
import time
import json
from datetime import datetime

# Mock evaluation for testing (replace with actual when available)
def mock_evaluate_pos(first, second):
    """Mock evaluation for testing without neural network"""
    board = chess.Board(second)
    
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 300,
        chess.BISHOP: 300,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0
    }
    
    material_score = 0
    for piece_type in piece_values:
        white_pieces = len(board.pieces(piece_type, chess.WHITE))
        black_pieces = len(board.pieces(piece_type, chess.BLACK))
        material_score += (black_pieces - white_pieces) * piece_values[piece_type]
    
    # Add positional factors
    positional_score = 0
    
    # Control of center
    center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
    for square in center_squares:
        piece = board.piece_at(square)
        if piece:
            if piece.color == chess.BLACK:
                positional_score += 30
            else:
                positional_score -= 30
    
    # Development bonus
    for color in [chess.WHITE, chess.BLACK]:
        developed = 0
        back_rank_squares = [chess.B1, chess.C1, chess.F1, chess.G1] if color == chess.WHITE else [chess.B8, chess.C8, chess.F8, chess.G8]
        for square in back_rank_squares:
            piece = board.piece_at(square)
            if piece is None or piece.piece_type == chess.KING:
                developed += 1
        
        if color == chess.BLACK:
            positional_score += developed * 20
        else:
            positional_score -= developed * 20
    
    return material_score + positional_score

# ORIGINAL ENGINE (V1.0)
class EngineOriginal:
    def __init__(self, fen):
        self.board = chess.Board()
        self.board.set_fen(fen)
        self.version = "1.0"
        self.name = "Original Engine"

    def get_move(self):
        best_move = self.make_move()
        return str(best_move)

    def make_move(self):
        all_pos = {}
        black_legal_moves = self.board.legal_moves
        current_fen_x = self.board.fen()
        black_response = {}
        
        for black_move in black_legal_moves:
            white_response = {}
            self.board.push(black_move)
            if self.board.is_checkmate():
                self.board.pop()
                return str(black_move)
            
            white_legal_moves = self.board.legal_moves
            for white_move in white_legal_moves:
                self.board.push(white_move)
                if self.board.is_checkmate():
                    white_response[white_move] = 0
                    self.board.pop()
                    break
                
                black_legal_moves_depth_2 = self.board.legal_moves
                black_response_depth_2 = {}
                
                for black_move_depth_2 in black_legal_moves_depth_2:
                    self.board.push(black_move_depth_2)
                    if self.board.is_checkmate():
                        black_response_depth_2[black_move_depth_2] = 1
                        self.board.pop()
                        break
                    
                    next_fen_x = self.board.fen()
                    if next_fen_x in all_pos:
                        prediction_number = all_pos[next_fen_x]
                    else:
                        prediction_number = mock_evaluate_pos(current_fen_x, next_fen_x)
                        all_pos[next_fen_x] = prediction_number
                    
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
            return str(best_move)
        else:
            return "checkmate"

# IMPROVED ENGINE (V1.5) 
def order_moves_v15(board, moves):
    """Basic move ordering"""
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
        
        # Prioritize development
        if board.piece_at(move.from_square):
            piece = board.piece_at(move.from_square)
            if piece.color == chess.BLACK:
                if chess.square_rank(move.from_square) == 7:
                    score += 15
        
        move_scores.append((score, move))
    
    move_scores.sort(key=lambda x: x[0], reverse=True)
    return [move for score, move in move_scores]

class EngineImproved:
    def __init__(self, fen):
        self.board = chess.Board()
        self.board.set_fen(fen)
        self.position_cache = {}
        self.version = "1.5"
        self.name = "Improved Engine (Move Ordering)"

    def get_move(self):
        best_move = self.make_move()
        return str(best_move)

    def make_move(self):
        current_fen_x = self.board.fen()
        black_response = {}
        
        # Order moves for better search
        black_legal_moves = order_moves_v15(self.board, list(self.board.legal_moves))
        
        for black_move in black_legal_moves:
            white_response = {}
            self.board.push(black_move)
            if self.board.is_checkmate():
                self.board.pop()
                return str(black_move)
            
            # Order white moves too
            white_legal_moves = order_moves_v15(self.board, list(self.board.legal_moves))
            
            for white_move in white_legal_moves:
                self.board.push(white_move)
                if self.board.is_checkmate():
                    white_response[white_move] = 0
                    self.board.pop()
                    break
                
                black_legal_moves_depth_2 = order_moves_v15(self.board, list(self.board.legal_moves))
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
                        prediction_number = mock_evaluate_pos(current_fen_x, next_fen_x)
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
            return str(best_move)
        else:
            return "checkmate"

# V2.0 ENGINE (Alpha-Beta)
def order_moves_v2(board, moves):
    """Advanced move ordering for V2.0"""
    move_scores = []
    
    for move in moves:
        score = 0
        
        # High priority for captures
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
                score += victim_value - (attacker_value // 10)
        
        # Very high priority for checks
        board.push(move)
        if board.is_check():
            score += 1000
            if board.is_checkmate():
                score += 10000
        board.pop()
        
        # Promotion bonus
        if move.promotion:
            score += 800
        
        # Central control
        if move.to_square in [chess.E4, chess.E5, chess.D4, chess.D5]:
            score += 50
        elif move.to_square in [chess.C3, chess.C6, chess.F3, chess.F6]:
            score += 30
        
        # Development bonus
        if board.piece_at(move.from_square):
            piece = board.piece_at(move.from_square)
            if piece.color == chess.BLACK:
                from_rank = chess.square_rank(move.from_square)
                to_rank = chess.square_rank(move.to_square)
                
                if from_rank == 7 and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                    score += 40
                
                if to_rank < from_rank:
                    score += 10
        
        # Castle bonus
        if board.is_castling(move):
            score += 60
        
        move_scores.append((score, move))
    
    move_scores.sort(key=lambda x: x[0], reverse=True)
    return [move for score, move in move_scores]

class EngineV2:
    def __init__(self, fen):
        self.board = chess.Board()
        self.board.set_fen(fen)
        self.position_cache = {}
        self.nodes_searched = 0
        self.version = "2.0"
        self.name = "Alpha-Beta Engine"

    def get_move(self):
        best_move = self.make_move()
        return str(best_move)

    def make_move(self):
        self.nodes_searched = 0
        
        # Use alpha-beta search
        best_move, best_score = self.alpha_beta_root(3, float('-inf'), float('inf'))
        
        if best_move is None:
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                best_move = legal_moves[0]
            else:
                return "checkmate"
        
        return str(best_move)
    
    def alpha_beta_root(self, depth, alpha, beta):
        legal_moves = order_moves_v2(self.board, list(self.board.legal_moves))
        
        if not legal_moves:
            return None, float('-inf')
        
        best_move = None
        best_score = float('-inf')
        
        for move in legal_moves:
            self.board.push(move)
            
            if self.board.is_checkmate():
                self.board.pop()
                return move, 10000
            
            score = self.alpha_beta(depth - 1, alpha, beta, False)
            self.board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, score)
            if beta <= alpha:
                break
        
        return best_move, best_score
    
    def alpha_beta(self, depth, alpha, beta, maximizing_player):
        self.nodes_searched += 1
        
        if self.board.is_checkmate():
            if maximizing_player:
                return float('-inf')
            else:
                return float('inf')
        
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0
        
        if depth == 0:
            return self.evaluate_position()
        
        legal_moves = order_moves_v2(self.board, list(self.board.legal_moves))
        
        if maximizing_player:
            max_eval = float('-inf')
            for move in legal_moves:
                self.board.push(move)
                eval_score = self.alpha_beta(depth - 1, alpha, beta, False)
                self.board.pop()
                
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                if beta <= alpha:
                    break
            
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                self.board.push(move)
                eval_score = self.alpha_beta(depth - 1, alpha, beta, True)
                self.board.pop()
                
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                if beta <= alpha:
                    break
            
            return min_eval
    
    def evaluate_position(self):
        current_fen = self.board.fen()
        
        if current_fen in self.position_cache:
            return self.position_cache[current_fen]
        
        # Get neural network evaluation (using mock for now)
        nn_score = mock_evaluate_pos(current_fen, current_fen)
        
        # Add tactical bonuses
        tactical_bonus = 0
        
        if self.board.is_check():
            if self.board.turn == chess.WHITE:
                tactical_bonus -= 50
            else:
                tactical_bonus += 50
        
        # Material balance
        material_score = self.calculate_material_advantage()
        
        final_score = nn_score + (tactical_bonus * 0.1) + (material_score * 0.05)
        
        self.position_cache[current_fen] = final_score
        return final_score
    
    def calculate_material_advantage(self):
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0
        }
        
        black_material = 0
        white_material = 0
        
        for piece_type in piece_values:
            black_pieces = len(self.board.pieces(piece_type, chess.BLACK))
            white_pieces = len(self.board.pieces(piece_type, chess.WHITE))
            
            piece_value = piece_values[piece_type]
            black_material += black_pieces * piece_value
            white_material += white_pieces * piece_value
        
        return black_material - white_material

# V3.0 ENGINE (Quiescence + Advanced Features)
def order_moves_v3(board, moves, killer_moves=None):
    """V3 Enhanced move ordering with killer moves"""
    move_scores = []
    
    if killer_moves is None:
        killer_moves = []
    
    for move in moves:
        score = 0
        
        # Killer moves get high priority
        if move in killer_moves:
            score += 5000
        
        # Captures with MVV-LVA
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
                score += victim_value - (attacker_value // 20)
        
        # Checks and checkmates
        board.push(move)
        if board.is_check():
            score += 2000
            if board.is_checkmate():
                score += 15000
        board.pop()
        
        # Promotions
        if move.promotion:
            if move.promotion == chess.QUEEN:
                score += 1500
            elif move.promotion == chess.ROOK:
                score += 1000
            else:
                score += 500
        
        # Castling for safety
        if board.is_castling(move):
            score += 200
        
        # Central control with higher values
        if move.to_square in [chess.E4, chess.E5, chess.D4, chess.D5]:
            score += 80
        elif move.to_square in [chess.C3, chess.C6, chess.F3, chess.F6, chess.C4, chess.C5, chess.F4, chess.F5]:
            score += 50
        
        # Advanced development bonuses
        if board.piece_at(move.from_square):
            piece = board.piece_at(move.from_square)
            if piece.color == chess.BLACK:
                from_rank = chess.square_rank(move.from_square)
                to_rank = chess.square_rank(move.to_square)
                
                if from_rank == 7 and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                    score += 70
                
                if to_rank < from_rank:
                    score += 20
                
                if piece.piece_type == chess.KNIGHT:
                    if move.to_square in [chess.C6, chess.E6, chess.F6, chess.D6]:
                        score += 40
                elif piece.piece_type == chess.BISHOP:
                    if move.to_square in [chess.B7, chess.C8, chess.F8, chess.G7]:
                        score += 30
        
        move_scores.append((score, move))
    
    move_scores.sort(key=lambda x: x[0], reverse=True)
    return [move for score, move in move_scores]

class EngineV3:
    def __init__(self, fen):
        self.board = chess.Board()
        self.board.set_fen(fen)
        self.position_cache = {}
        self.nodes_searched = 0
        self.killer_moves = [[] for _ in range(10)]
        self.cache_hits = 0
        self.version = "3.0"
        self.name = "Quiescence Engine (V3.0)"

    def get_move(self):
        best_move = self.make_move()
        return str(best_move)

    def make_move(self):
        self.nodes_searched = 0
        self.cache_hits = 0
        
        # Use iterative deepening alpha-beta with quiescence
        best_move = None
        for depth in range(1, 4):  # Depths 1-3 for tournament speed
            try:
                move, score = self.alpha_beta_root(depth, float('-inf'), float('inf'))
                if move:
                    best_move = move
            except:
                break
        
        if best_move is None:
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                best_move = legal_moves[0]
            else:
                return "checkmate"
        
        return str(best_move)
    
    def alpha_beta_root(self, depth, alpha, beta):
        killer_moves = self.killer_moves[0] if depth < len(self.killer_moves) else []
        legal_moves = order_moves_v3(self.board, list(self.board.legal_moves), killer_moves)
        
        if not legal_moves:
            return None, float('-inf')
        
        best_move = None
        best_score = float('-inf')
        
        for move in legal_moves:
            self.board.push(move)
            
            if self.board.is_checkmate():
                self.board.pop()
                return move, 20000
            
            score = self.alpha_beta(depth - 1, alpha, beta, False, 1)
            self.board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
                
                if depth < len(self.killer_moves) and move not in self.killer_moves[0]:
                    if len(self.killer_moves[0]) >= 2:
                        self.killer_moves[0].pop()
                    self.killer_moves[0].insert(0, move)
            
            alpha = max(alpha, score)
            if beta <= alpha:
                break
        
        return best_move, best_score
    
    def alpha_beta(self, depth, alpha, beta, maximizing_player, ply):
        self.nodes_searched += 1
        
        if self.board.is_checkmate():
            if maximizing_player:
                return float('-inf') + ply
            else:
                return float('inf') - ply
        
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0
        
        # Enhanced caching
        current_fen = self.board.fen()
        cache_key = f"{current_fen}_{depth}_{alpha}_{beta}_{maximizing_player}"
        if cache_key in self.position_cache:
            self.cache_hits += 1
            return self.position_cache[cache_key]
        
        if depth == 0:
            # Enter quiescence search for tactical stability
            score = self.quiescence(alpha, beta, maximizing_player, 3)
            self.position_cache[cache_key] = score
            return score
        
        killer_moves = self.killer_moves[ply] if ply < len(self.killer_moves) else []
        legal_moves = order_moves_v3(self.board, list(self.board.legal_moves), killer_moves)
        
        if maximizing_player:
            max_eval = float('-inf')
            for move in legal_moves:
                self.board.push(move)
                eval_score = self.alpha_beta(depth - 1, alpha, beta, False, ply + 1)
                self.board.pop()
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    
                    if ply < len(self.killer_moves) and move not in self.killer_moves[ply]:
                        if len(self.killer_moves[ply]) >= 2:
                            self.killer_moves[ply].pop()
                        self.killer_moves[ply].insert(0, move)
                
                alpha = max(alpha, eval_score)
                
                if beta <= alpha:
                    break
            
            self.position_cache[cache_key] = max_eval
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                self.board.push(move)
                eval_score = self.alpha_beta(depth - 1, alpha, beta, True, ply + 1)
                self.board.pop()
                
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                if beta <= alpha:
                    break
            
            self.position_cache[cache_key] = min_eval
            return min_eval
    
    def quiescence(self, alpha, beta, maximizing_player, depth):
        """Quiescence search to avoid horizon effect"""
        if depth == 0:
            return self.evaluate_position()
        
        stand_pat = self.evaluate_position()
        
        if maximizing_player:
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
            
            # Only consider captures and checks
            captures = [move for move in self.board.legal_moves 
                       if self.board.piece_at(move.to_square) is not None]
            
            if not captures:
                return stand_pat
            
            captures = order_moves_v3(self.board, captures)
            
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
            
            captures = [move for move in self.board.legal_moves 
                       if self.board.piece_at(move.to_square) is not None]
            
            if not captures:
                return stand_pat
            
            captures = order_moves_v3(self.board, captures)
            
            for move in captures:
                self.board.push(move)
                score = self.quiescence(alpha, beta, True, depth - 1)
                self.board.pop()
                
                if score <= alpha:
                    return alpha
                beta = min(beta, score)
            
            return beta
    
    def evaluate_position(self):
        current_fen = self.board.fen()
        
        if current_fen in self.position_cache:
            return self.position_cache[current_fen]
        
        # Enhanced evaluation with mock
        nn_score = mock_evaluate_pos(current_fen, current_fen)
        
        # Enhanced tactical bonuses
        tactical_bonus = 0
        
        if self.board.is_check():
            if self.board.turn == chess.WHITE:
                tactical_bonus -= 75
            else:
                tactical_bonus += 75
        
        # Material calculation
        material_score = self.calculate_material_advantage()
        
        # Pattern recognition
        positional_bonus = self.evaluate_patterns()
        
        final_score = (nn_score + 
                      (tactical_bonus * 0.15) + 
                      (material_score * 0.08) + 
                      (positional_bonus * 0.1))
        
        self.position_cache[current_fen] = final_score
        return final_score
    
    def calculate_material_advantage(self):
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0
        }
        
        black_material = 0
        white_material = 0
        
        for piece_type in piece_values:
            black_pieces = len(self.board.pieces(piece_type, chess.BLACK))
            white_pieces = len(self.board.pieces(piece_type, chess.WHITE))
            
            piece_value = piece_values[piece_type]
            black_material += black_pieces * piece_value
            white_material += white_pieces * piece_value
        
        return black_material - white_material
    
    def evaluate_patterns(self):
        """Enhanced pattern evaluation"""
        score = 0
        
        # Center control
        for square in [chess.E4, chess.E5, chess.D4, chess.D5]:
            piece = self.board.piece_at(square)
            if piece:
                if piece.color == chess.BLACK:
                    score += 40
                else:
                    score -= 40
        
        # King safety
        black_king = self.board.king(chess.BLACK)
        if black_king:
            if black_king in [chess.G8, chess.C8]:
                score += 50
            if chess.square_rank(black_king) < 6:
                score -= 40
        
        return score

def play_game(white_engine, black_engine, max_moves=100):
    """Play a game between two engines"""
    board = chess.Board()
    moves_played = 0
    
    print(f"🏁 Game: {white_engine.name} (White) vs {black_engine.name} (Black)")
    
    while not board.is_game_over() and moves_played < max_moves:
        if board.turn == chess.WHITE:
            # White to move - flip board for white engine to play as black
            flipped_fen = flip_fen(board.fen())
            white_engine.board.set_fen(flipped_fen)
            move_str = white_engine.get_move()
            
            if move_str == "checkmate":
                break
            
            # Convert move back
            try:
                move = chess.Move.from_uci(move_str)
                move = flip_move(move)
                if move in board.legal_moves:
                    board.push(move)
                    moves_played += 1
                else:
                    print(f"❌ Illegal move by {white_engine.name}: {move_str}")
                    break
            except:
                print(f"❌ Invalid move by {white_engine.name}: {move_str}")
                break
        else:
            # Black to move
            black_engine.board.set_fen(board.fen())
            move_str = black_engine.get_move()
            
            if move_str == "checkmate":
                break
            
            try:
                move = chess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    board.push(move)
                    moves_played += 1
                else:
                    print(f"❌ Illegal move by {black_engine.name}: {move_str}")
                    break
            except:
                print(f"❌ Invalid move by {black_engine.name}: {move_str}")
                break
    
    # Determine result
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            result = "0-1"  # Black wins
            winner = black_engine.name
        else:
            result = "1-0"  # White wins
            winner = white_engine.name
    elif board.is_stalemate() or board.is_insufficient_material() or moves_played >= max_moves:
        result = "1/2-1/2"  # Draw
        winner = "Draw"
    else:
        result = "1/2-1/2"  # Draw (incomplete)
        winner = "Draw"
    
    print(f"🎯 Result: {result} - {winner}")
    return result, winner, moves_played

def flip_fen(fen):
    """Flip a FEN string to reverse colors"""
    parts = fen.split()
    
    # Flip the board
    rows = parts[0].split('/')
    flipped_rows = []
    for row in reversed(rows):
        flipped_row = ""
        for char in row:
            if char.isalpha():
                if char.isupper():
                    flipped_row += char.lower()
                else:
                    flipped_row += char.upper()
            else:
                flipped_row += char
        flipped_rows.append(flipped_row)
    
    # Flip active color
    parts[0] = '/'.join(flipped_rows)
    parts[1] = 'w' if parts[1] == 'b' else 'b'
    
    return ' '.join(parts)

def flip_move(move):
    """Flip a move to work with flipped board"""
    from_square = chess.square_mirror(move.from_square)
    to_square = chess.square_mirror(move.to_square)
    return chess.Move(from_square, to_square, move.promotion)

def run_tournament():
    """Run a tournament between all engine versions"""
    print("🏆" + "="*60 + "🏆")
    print("         NEURAL CHESS ENGINE TOURNAMENT")
    print("🏆" + "="*60 + "🏆")
    
    # Create engines
    engines = [
        EngineOriginal("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        EngineImproved("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        EngineV2("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        EngineV3("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    ]
    
    # Tournament results
    results = {}
    for engine in engines:
        results[engine.name] = {"wins": 0, "losses": 0, "draws": 0, "points": 0}
    
    # Play round-robin tournament
    games_per_pairing = 2  # Each engine plays as both white and black
    
    for i, engine1 in enumerate(engines):
        for j, engine2 in enumerate(engines):
            if i != j:
                print(f"\n🎮 MATCHUP: {engine1.name} vs {engine2.name}")
                print("-" * 50)
                
                for game_num in range(games_per_pairing):
                    if game_num == 0:
                        white_engine = engine1
                        black_engine = engine2
                    else:
                        white_engine = engine2
                        black_engine = engine1
                    
                    result, winner, moves = play_game(white_engine, black_engine)
                    
                    # Update results
                    if winner == "Draw":
                        results[white_engine.name]["draws"] += 1
                        results[black_engine.name]["draws"] += 1
                        results[white_engine.name]["points"] += 0.5
                        results[black_engine.name]["points"] += 0.5
                    elif winner == white_engine.name:
                        results[white_engine.name]["wins"] += 1
                        results[black_engine.name]["losses"] += 1
                        results[white_engine.name]["points"] += 1
                    else:
                        results[black_engine.name]["wins"] += 1
                        results[white_engine.name]["losses"] += 1
                        results[black_engine.name]["points"] += 1
    
    # Display final results
    print("\n" + "🏆" + "="*60 + "🏆")
    print("                 TOURNAMENT RESULTS")
    print("🏆" + "="*60 + "🏆")
    
    # Sort by points
    sorted_results = sorted(results.items(), key=lambda x: x[1]["points"], reverse=True)
    
    print(f"{'Rank':<6}{'Engine':<30}{'W':<4}{'L':<4}{'D':<4}{'Points':<8}")
    print("-" * 60)
    
    for rank, (engine_name, stats) in enumerate(sorted_results, 1):
        if rank == 1:
            trophy = "🥇"
        elif rank == 2:
            trophy = "🥈"
        elif rank == 3:
            trophy = "🥉"
        else:
            trophy = "  "
        
        print(f"{trophy} {rank:<3}{engine_name:<30}{stats['wins']:<4}{stats['losses']:<4}{stats['draws']:<4}{stats['points']:<8}")
    
    # Determine champion
    champion_name, champion_stats = sorted_results[0]
    print(f"\n🎉 CHAMPION: {champion_name}")
    print(f"🏆 Final Score: {champion_stats['points']} points")
    print(f"📊 Record: {champion_stats['wins']}W-{champion_stats['losses']}L-{champion_stats['draws']}D")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"tournament_results_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "champion": champion_name,
            "results": results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    return champion_name, results

if __name__ == "__main__":
    champion, results = run_tournament()
    print(f"\n🎯 The strongest engine is: {champion}!") 