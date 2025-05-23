import chess
import numpy as np
import tflite_runtime.interpreter as tflite
from util import *

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

def order_moves(board, moves):
    """Advanced move ordering for better alpha-beta efficiency"""
    move_scores = []
    
    for move in moves:
        score = 0
        
        # High priority for captures (MVV-LVA)
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
                # Most Valuable Victim - Least Valuable Attacker
                victim_value = piece_values.get(captured_piece.piece_type, 0)
                attacker_value = piece_values.get(moving_piece.piece_type, 100)
                score += victim_value - (attacker_value // 10)
        
        # Very high priority for checks
        board.push(move)
        if board.is_check():
            score += 1000
            # Even higher for checkmate
            if board.is_checkmate():
                score += 10000
        board.pop()
        
        # Promotion bonus
        if move.promotion:
            score += 800
        
        # Central control bonus
        if move.to_square in [chess.E4, chess.E5, chess.D4, chess.D5]:
            score += 50
        elif move.to_square in [chess.C3, chess.C6, chess.F3, chess.F6]:  # Extended center
            score += 30
        
        # Development bonus for pieces moving off back rank
        if board.piece_at(move.from_square):
            piece = board.piece_at(move.from_square)
            if piece.color == chess.BLACK:
                from_rank = chess.square_rank(move.from_square)
                to_rank = chess.square_rank(move.to_square)
                
                # Bonus for developing from back rank
                if from_rank == 7 and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                    score += 40
                
                # Bonus for advancing pieces
                if to_rank < from_rank:  # Moving forward
                    score += 10
        
        # Castle bonus
        if board.is_castling(move):
            score += 60
        
        move_scores.append((score, move))
    
    # Sort by score descending
    move_scores.sort(key=lambda x: x[0], reverse=True)
    return [move for score, move in move_scores]

# create Engine class
class EngineV2:
    def __init__(self, fen):
        self.board = chess.Board()
        self.board.set_fen(fen)
        self.position_cache = {}
        self.nodes_searched = 0

    # is called in flask app
    def get_move(self):
        best_move = self.make_move()
        return str(best_move)

    # get best move for black with proper alpha-beta pruning
    def make_move(self):
        self.nodes_searched = 0
        current_fen = self.board.fen()
        print("calculating\n")
        
        # Use alpha-beta search with proper bounds
        best_move, best_score = self.alpha_beta_root(3, float('-inf'), float('inf'))
        
        if best_move is None:
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                best_move = legal_moves[0]
            else:
                return "checkmate"
        
        print(f"best move found: {best_move} (score: {best_score:.2f}, nodes: {self.nodes_searched})")
        return str(best_move)
    
    def alpha_beta_root(self, depth, alpha, beta):
        """Root level alpha-beta search"""
        legal_moves = order_moves(self.board, list(self.board.legal_moves))
        
        if not legal_moves:
            return None, float('-inf')
        
        best_move = None
        best_score = float('-inf')
        
        for move in legal_moves:
            print(".")
            self.board.push(move)
            
            # Check for immediate checkmate
            if self.board.is_checkmate():
                self.board.pop()
                return move, 10000
            
            # Search this move
            score = self.alpha_beta(depth - 1, alpha, beta, False)
            self.board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, score)
            if beta <= alpha:
                break  # Beta cutoff
        
        return best_move, best_score
    
    def alpha_beta(self, depth, alpha, beta, maximizing_player):
        """Alpha-beta search with proper pruning"""
        self.nodes_searched += 1
        
        # Terminal conditions
        if self.board.is_checkmate():
            if maximizing_player:
                return float('-inf')  # Bad for maximizing player
            else:
                return float('inf')   # Good for minimizing player
        
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0  # Draw
        
        if depth == 0:
            return self.evaluate_position()
        
        legal_moves = order_moves(self.board, list(self.board.legal_moves))
        
        if maximizing_player:
            max_eval = float('-inf')
            for move in legal_moves:
                self.board.push(move)
                eval_score = self.alpha_beta(depth - 1, alpha, beta, False)
                self.board.pop()
                
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                if beta <= alpha:
                    break  # Beta cutoff
            
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
                    break  # Alpha cutoff
            
            return min_eval
    
    def evaluate_position(self):
        """Enhanced position evaluation"""
        current_fen = self.board.fen()
        
        # Use cache if available
        if current_fen in self.position_cache:
            return self.position_cache[current_fen]
        
        # Get neural network evaluation
        original_fen = self.board.fen()  # For now, use current position
        nn_score = evaluate_pos(original_fen, current_fen)
        
        # Add tactical bonuses
        tactical_bonus = 0
        
        # Check bonus/penalty
        if self.board.is_check():
            if self.board.turn == chess.WHITE:
                tactical_bonus -= 50  # White is in check (bad for Black if we're Black)
            else:
                tactical_bonus += 50  # Black is in check (good for Black if we're White)
        
        # Material imbalance detection
        material_score = self.calculate_material_advantage()
        
        # Combine scores
        final_score = nn_score + (tactical_bonus * 0.1) + (material_score * 0.05)
        
        # Cache the result
        self.position_cache[current_fen] = final_score
        
        return final_score
    
    def calculate_material_advantage(self):
        """Calculate material advantage for Black"""
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