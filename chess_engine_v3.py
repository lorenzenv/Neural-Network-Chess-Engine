import chess
import numpy as np
import tflite_runtime.interpreter as tflite
from util import *

# Engine Version Information
ENGINE_VERSION = "3.0"
ENGINE_NAME = "Neural Chess Engine V3 - Quiescence"
ENGINE_FEATURES = [
    "Quiescence Search (tactical stability)",
    "Killer Move Heuristic (speed boost)", 
    "Enhanced Alpha-Beta Pruning",
    "Advanced Move Ordering",
    "Intelligent Position Caching",
    "Tactical Pattern Recognition",
    "Opening Book Integration"
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

# evaluate two input positions
def evaluate_pos(first, second):
    x_1, x_2 = make_x(first,second)
    interpreter.set_tensor(input_details[0]['index'], x_1)
    interpreter.set_tensor(input_details[1]['index'], x_2)
    interpreter.invoke()
    evaluation = interpreter.get_tensor(output_details[0]['index'])[0][0]
    return evaluation

# Opening book for instant responses
OPENING_BOOK = {
    # Italian Game variations
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1": "e7e5",  # 1.e4 e5
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2": "g1f3",  # 2.Nf3
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2": "b8c6",  # 2...Nc6
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3": "f1c4",  # 3.Bc4
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3": "g8f6",  # 3...Nf6
    
    # Sicilian Defense
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1": "c7c5",  # 1.e4 c5
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2": "g1f3",  # 2.Nf3
    
    # French Defense  
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1": "e7e6",  # 1.e4 e6
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": "d2d4",  # 2.d4
    
    # King's Indian Defense
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1": "g8f6",  # 1.d4 Nf6
    "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2": "c2c4",  # 2.c4
}

def get_opening_move(fen):
    """Check if we have a book move for this position"""
    return OPENING_BOOK.get(fen)

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
                # Higher priority for favorable trades
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
                
                # Strong development bonus
                if from_rank == 7 and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                    score += 70
                
                # Forward progress
                if to_rank < from_rank:
                    score += 20
                
                # Piece activity bonus
                if piece.piece_type == chess.KNIGHT:
                    # Knights prefer central outposts
                    if move.to_square in [chess.C6, chess.E6, chess.F6, chess.D6]:
                        score += 40
                elif piece.piece_type == chess.BISHOP:
                    # Bishops prefer long diagonals
                    if move.to_square in [chess.B7, chess.C8, chess.F8, chess.G7]:
                        score += 30
        
        # Prevent opponent threats (basic)
        piece = board.piece_at(move.to_square)
        if piece and piece.color != board.turn:
            # Attacking opponent pieces
            score += 25
        
        move_scores.append((score, move))
    
    # Sort by score descending
    move_scores.sort(key=lambda x: x[0], reverse=True)
    return [move for score, move in move_scores]

class EngineV3:
    def __init__(self, fen):
        self.board = chess.Board()
        self.board.set_fen(fen)
        self.position_cache = {}
        self.nodes_searched = 0
        self.killer_moves = [[] for _ in range(10)]  # Killer moves per depth
        self.cache_hits = 0

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
        current_fen = self.board.fen()
        print("calculating\n")
        
        # Check opening book first
        book_move = get_opening_move(current_fen)
        if book_move:
            print(f"📚 Opening book move: {book_move}")
            return book_move
        
        # Use iterative deepening for better time management
        best_move = None
        for depth in range(1, 5):  # Search depths 1-4
            try:
                move, score = self.alpha_beta_root(depth, float('-inf'), float('inf'))
                if move:
                    best_move = move
                print(f"Depth {depth}: {move} (score: {score:.2f})")
            except:
                break
        
        if best_move is None:
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                best_move = legal_moves[0]
            else:
                return "checkmate"
        
        print(f"🎯 best move: {best_move} (nodes: {self.nodes_searched}, cache hits: {self.cache_hits})")
        return str(best_move)
    
    def alpha_beta_root(self, depth, alpha, beta):
        """Root level alpha-beta search with killer moves"""
        killer_moves = self.killer_moves[0] if depth < len(self.killer_moves) else []
        legal_moves = order_moves_v3(self.board, list(self.board.legal_moves), killer_moves)
        
        if not legal_moves:
            return None, float('-inf')
        
        best_move = None
        best_score = float('-inf')
        
        for move in legal_moves:
            print(".")
            self.board.push(move)
            
            # Immediate checkmate
            if self.board.is_checkmate():
                self.board.pop()
                return move, 20000
            
            # Search this move
            score = self.alpha_beta(depth - 1, alpha, beta, False, 1)
            self.board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
                
                # Update killer move
                if depth < len(self.killer_moves) and move not in self.killer_moves[0]:
                    if len(self.killer_moves[0]) >= 2:
                        self.killer_moves[0].pop()
                    self.killer_moves[0].insert(0, move)
            
            alpha = max(alpha, score)
            if beta <= alpha:
                break
        
        return best_move, best_score
    
    def alpha_beta(self, depth, alpha, beta, maximizing_player, ply):
        """Enhanced alpha-beta with quiescence search"""
        self.nodes_searched += 1
        
        # Terminal conditions
        if self.board.is_checkmate():
            if maximizing_player:
                return float('-inf') + ply  # Prefer faster checkmates
            else:
                return float('inf') - ply
        
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0
        
        # Check cache
        current_fen = self.board.fen()
        cache_key = f"{current_fen}_{depth}_{alpha}_{beta}_{maximizing_player}"
        if cache_key in self.position_cache:
            self.cache_hits += 1
            return self.position_cache[cache_key]
        
        if depth == 0:
            # Enter quiescence search for tactical stability
            score = self.quiescence(alpha, beta, maximizing_player, 5)
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
                    
                    # Update killer move for this ply
                    if ply < len(self.killer_moves) and move not in self.killer_moves[ply]:
                        if len(self.killer_moves[ply]) >= 2:
                            self.killer_moves[ply].pop()
                        self.killer_moves[ply].insert(0, move)
                
                alpha = max(alpha, eval_score)
                
                if beta <= alpha:
                    break  # Beta cutoff
            
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
                    break  # Alpha cutoff
            
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
                       if self.board.piece_at(move.to_square) is not None or self.board.is_check()]
            
            if not captures:
                return stand_pat
            
            # Order captures by MVV-LVA
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
                       if self.board.piece_at(move.to_square) is not None or self.board.is_check()]
            
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
        """Enhanced position evaluation with pattern recognition"""
        current_fen = self.board.fen()
        
        # Use cache if available
        if current_fen in self.position_cache:
            return self.position_cache[current_fen]
        
        # Get neural network evaluation
        original_fen = self.board.fen()
        nn_score = evaluate_pos(original_fen, current_fen)
        
        # Enhanced tactical bonuses
        tactical_bonus = 0
        
        # Check penalty/bonus
        if self.board.is_check():
            if self.board.turn == chess.WHITE:
                tactical_bonus -= 75  # Being in check is worse
            else:
                tactical_bonus += 75
        
        # Material balance with endgame scaling
        material_score = self.calculate_material_advantage()
        
        # Positional bonuses
        positional_bonus = self.evaluate_position_patterns()
        
        # King safety
        king_safety = self.evaluate_king_safety()
        
        # Combine all factors
        final_score = (nn_score + 
                      (tactical_bonus * 0.15) + 
                      (material_score * 0.08) + 
                      (positional_bonus * 0.1) + 
                      (king_safety * 0.12))
        
        # Cache the result
        self.position_cache[current_fen] = final_score
        
        return final_score
    
    def calculate_material_advantage(self):
        """Enhanced material calculation"""
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
    
    def evaluate_position_patterns(self):
        """Evaluate positional patterns"""
        score = 0
        
        # Control of center squares
        center_control = 0
        for square in [chess.E4, chess.E5, chess.D4, chess.D5]:
            piece = self.board.piece_at(square)
            if piece:
                if piece.color == chess.BLACK:
                    center_control += 30
                else:
                    center_control -= 30
        
        # Pawn structure bonuses
        pawn_structure = 0
        black_pawns = self.board.pieces(chess.PAWN, chess.BLACK)
        for square in black_pawns:
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            
            # Passed pawn bonus
            if self.is_passed_pawn(square, chess.BLACK):
                pawn_structure += 50 + (2 - rank) * 10  # More valuable closer to promotion
            
            # Connected pawns
            for adjacent_file in [file - 1, file + 1]:
                if 0 <= adjacent_file <= 7:
                    adjacent_square = chess.square(adjacent_file, rank)
                    if adjacent_square in black_pawns:
                        pawn_structure += 15
        
        score = center_control + pawn_structure
        return score
    
    def is_passed_pawn(self, square, color):
        """Check if a pawn is passed"""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        
        opponent_color = chess.WHITE if color == chess.BLACK else chess.BLACK
        opponent_pawns = self.board.pieces(chess.PAWN, opponent_color)
        
        # Check if any opponent pawns can stop this pawn
        for opponent_square in opponent_pawns:
            opp_file = chess.square_file(opponent_square)
            opp_rank = chess.square_rank(opponent_square)
            
            # Check files that can interfere
            if abs(opp_file - file) <= 1:
                if color == chess.BLACK and opp_rank <= rank:
                    return False
                elif color == chess.WHITE and opp_rank >= rank:
                    return False
        
        return True
    
    def evaluate_king_safety(self):
        """Evaluate king safety"""
        safety_score = 0
        
        black_king = self.board.king(chess.BLACK)
        if black_king:
            # King in corner is safer in opening/middlegame
            if black_king in [chess.G8, chess.C8]:  # Castled positions
                safety_score += 40
            
            # Penalty for exposed king
            if chess.square_rank(black_king) < 6:  # King too far forward
                safety_score -= 30
        
        return safety_score 