#!/usr/bin/env python3
"""
⚡ V3.1 SPEED TEST vs V3.0 CHAMPION
Quick head-to-head to see if speed optimizations hurt playing strength
"""

import chess
import time
from datetime import datetime

# Mock evaluation for testing
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
    
    return material_score + positional_score

# Import both engines - create simplified versions for testing
class EngineV31:
    def __init__(self, fen):
        self.board = chess.Board()
        self.board.set_fen(fen)
        self.position_cache = {}
        self.nodes_searched = 0
        self.killer_moves = [[] for _ in range(10)]
        self.cache_hits = 0
        self.version = "3.1"
        self.name = "V3.1 Speed Optimized"

    def get_move(self):
        best_move = self.make_move()
        return str(best_move)

    def make_move(self):
        self.nodes_searched = 0
        self.cache_hits = 0
        
        # Limit cache size for memory efficiency
        if len(self.position_cache) > 10000:
            cache_items = list(self.position_cache.items())
            self.position_cache = dict(cache_items[-5000:])
        
        # V3.1 OPTIMIZATION: Reduced iterative deepening - only 2 depths
        best_move = None
        for depth in range(2, 4):  # Search depths 2-3 only
            try:
                move, score = self.alpha_beta_root(depth, float('-inf'), float('inf'))
                if move:
                    best_move = move
                
                # V3.1 OPTIMIZATION: Early exit if we found a great move
                if score > 500:
                    break
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
        legal_moves = order_moves_v3_optimized(self.board, list(self.board.legal_moves), killer_moves)
        
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
        
        # V3.1 OPTIMIZATION: Enhanced caching
        current_fen = self.board.fen()
        cache_key = f"{current_fen}_{depth}_{maximizing_player}"
        if cache_key in self.position_cache:
            self.cache_hits += 1
            return self.position_cache[cache_key]
        
        if depth == 0:
            # V3.1 OPTIMIZATION: Reduced quiescence depth from 3 to 2
            score = self.quiescence_optimized(alpha, beta, maximizing_player, 2)
            self.position_cache[cache_key] = score
            return score
        
        killer_moves = self.killer_moves[ply] if ply < len(self.killer_moves) else []
        legal_moves = order_moves_v3_optimized(self.board, list(self.board.legal_moves), killer_moves)
        
        if maximizing_player:
            max_eval = float('-inf')
            for move in legal_moves:
                self.board.push(move)
                eval_score = self.alpha_beta(depth - 1, alpha, beta, False, ply + 1)
                self.board.pop()
                
                if eval_score > max_eval:
                    max_eval = eval_score
                
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
    
    def quiescence_optimized(self, alpha, beta, maximizing_player, depth):
        """V3.1 OPTIMIZED: Faster quiescence search"""
        if depth == 0:
            return mock_evaluate_pos("", self.board.fen())
        
        stand_pat = mock_evaluate_pos("", self.board.fen())
        
        if maximizing_player:
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
            
            # V3.1 OPTIMIZATION: Only consider favorable captures
            captures = []
            for move in self.board.legal_moves:
                if self.board.piece_at(move.to_square) is not None:
                    captured_piece = self.board.piece_at(move.to_square)
                    moving_piece = self.board.piece_at(move.from_square)
                    
                    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
                    
                    if captured_piece and moving_piece:
                        victim_value = piece_values.get(captured_piece.piece_type, 0)
                        attacker_value = piece_values.get(moving_piece.piece_type, 1)
                        
                        if victim_value >= attacker_value:
                            captures.append(move)
            
            if not captures:
                return stand_pat
            
            # V3.1 OPTIMIZATION: Limit to best 3 captures for speed
            if len(captures) > 3:
                captures = captures[:3]
            
            for move in captures:
                self.board.push(move)
                score = self.quiescence_optimized(alpha, beta, False, depth - 1)
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
                    captured_piece = self.board.piece_at(move.to_square)
                    moving_piece = self.board.piece_at(move.from_square)
                    
                    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
                    
                    if captured_piece and moving_piece:
                        victim_value = piece_values.get(captured_piece.piece_type, 0)
                        attacker_value = piece_values.get(moving_piece.piece_type, 1)
                        
                        if victim_value >= attacker_value:
                            captures.append(move)
            
            if not captures:
                return stand_pat
            
            if len(captures) > 3:
                captures = captures[:3]
            
            for move in captures:
                self.board.push(move)
                score = self.quiescence_optimized(alpha, beta, True, depth - 1)
                self.board.pop()
                
                if score <= alpha:
                    return alpha
                beta = min(beta, score)
            
            return beta

# V3.1 Optimized move ordering (faster)
def order_moves_v3_optimized(board, moves, killer_moves=None):
    """V3.1 optimized move ordering - faster but still effective"""
    move_scores = []
    
    if killer_moves is None:
        killer_moves = []
    
    for move in moves:
        score = 0
        
        # Killer moves get highest priority
        if move in killer_moves:
            score += 10000
            move_scores.append((score, move))
            continue
        
        # Quick capture evaluation
        if board.piece_at(move.to_square) is not None:
            captured_piece = board.piece_at(move.to_square)
            moving_piece = board.piece_at(move.from_square)
            
            # Simplified piece values for speed
            piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
            
            if captured_piece and moving_piece:
                victim_value = piece_values.get(captured_piece.piece_type, 0)
                attacker_value = piece_values.get(moving_piece.piece_type, 1)
                score += (victim_value - attacker_value) * 100
        
        # Quick check for checks (without full move validation)
        if move.promotion == chess.QUEEN:
            score += 800
        elif move.promotion:
            score += 400
        
        # Central squares bonus
        if move.to_square in [chess.E4, chess.E5, chess.D4, chess.D5]:
            score += 50
        
        # Castling bonus
        if board.is_castling(move):
            score += 100
        
        move_scores.append((score, move))
    
    # Sort by score descending (but faster sorting)
    move_scores.sort(key=lambda x: x[0], reverse=True)
    return [move for score, move in move_scores]

print("✅ Loaded V3.1 Speed Optimized Engine (Mock Version)")

# V3.0 Engine (copy from tournament file with optimizations)
def order_moves_v3_original(board, moves, killer_moves=None):
    """Original V3 move ordering"""
    move_scores = []
    
    if killer_moves is None:
        killer_moves = []
    
    for move in moves:
        score = 0
        
        if move in killer_moves:
            score += 5000
        
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
        
        board.push(move)
        if board.is_check():
            score += 2000
            if board.is_checkmate():
                score += 15000
        board.pop()
        
        if move.promotion:
            if move.promotion == chess.QUEEN:
                score += 1500
            else:
                score += 500
        
        if board.is_castling(move):
            score += 200
        
        if move.to_square in [chess.E4, chess.E5, chess.D4, chess.D5]:
            score += 80
        elif move.to_square in [chess.C3, chess.C6, chess.F3, chess.F6]:
            score += 50
        
        move_scores.append((score, move))
    
    move_scores.sort(key=lambda x: x[0], reverse=True)
    return [move for score, move in move_scores]

class EngineV30:
    def __init__(self, fen):
        self.board = chess.Board()
        self.board.set_fen(fen)
        self.position_cache = {}
        self.nodes_searched = 0
        self.killer_moves = [[] for _ in range(10)]
        self.cache_hits = 0
        self.version = "3.0"
        self.name = "V3.0 Champion (Original)"

    def get_move(self):
        best_move = self.make_move()
        return str(best_move)

    def make_move(self):
        self.nodes_searched = 0
        self.cache_hits = 0
        
        # Original V3.0 search depths
        best_move = None
        for depth in range(1, 4):  # Depths 1-3
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
        legal_moves = order_moves_v3_original(self.board, list(self.board.legal_moves), killer_moves)
        
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
        
        if depth == 0:
            # Original V3.0 quiescence depth
            score = self.quiescence(alpha, beta, maximizing_player, 3)
            return score
        
        killer_moves = self.killer_moves[ply] if ply < len(self.killer_moves) else []
        legal_moves = order_moves_v3_original(self.board, list(self.board.legal_moves), killer_moves)
        
        if maximizing_player:
            max_eval = float('-inf')
            for move in legal_moves:
                self.board.push(move)
                eval_score = self.alpha_beta(depth - 1, alpha, beta, False, ply + 1)
                self.board.pop()
                
                if eval_score > max_eval:
                    max_eval = eval_score
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            
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
            
            return min_eval
    
    def quiescence(self, alpha, beta, maximizing_player, depth):
        """Original V3.0 quiescence search"""
        if depth == 0:
            return mock_evaluate_pos("", self.board.fen())
        
        stand_pat = mock_evaluate_pos("", self.board.fen())
        
        if maximizing_player:
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
            
            captures = [move for move in self.board.legal_moves 
                       if self.board.piece_at(move.to_square) is not None]
            
            if not captures:
                return stand_pat
            
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
            
            for move in captures:
                self.board.push(move)
                score = self.quiescence(alpha, beta, True, depth - 1)
                self.board.pop()
                
                if score <= alpha:
                    return alpha
                beta = min(beta, score)
            
            return beta

def play_speed_test():
    """Quick head-to-head between V3.1 and V3.0"""
    print("⚡" + "="*60 + "⚡")
    print("      V3.1 SPEED TEST vs V3.0 CHAMPION")
    print("⚡" + "="*60 + "⚡")
    
    games_to_play = 4  # Quick test - 2 games each color
    
    v31_wins = 0
    v30_wins = 0
    draws = 0
    
    v31_times = []
    v30_times = []
    
    for game_num in range(games_to_play):
        print(f"\n🎮 GAME {game_num + 1}/{games_to_play}")
        print("-" * 40)
        
        board = chess.Board()
        moves_played = 0
        max_moves = 50  # Shorter games for speed test
        
        if game_num % 2 == 0:
            white_engine = EngineV31("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            black_engine = EngineV30("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            print("🆚 V3.1 (White) vs V3.0 (Black)")
        else:
            white_engine = EngineV30("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            black_engine = EngineV31("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            print("🆚 V3.0 (White) vs V3.1 (Black)")
        
        while not board.is_game_over() and moves_played < max_moves:
            if board.turn == chess.WHITE:
                # White to move
                white_engine.board.set_fen(board.fen())
                
                start_time = time.time()
                move_str = white_engine.get_move()
                move_time = time.time() - start_time
                
                if isinstance(white_engine, EngineV31):
                    v31_times.append(move_time)
                else:
                    v30_times.append(move_time)
                
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in board.legal_moves:
                        board.push(move)
                        moves_played += 1
                        print(f"⚪ {move_str} ({move_time:.2f}s)")
                    else:
                        print(f"❌ Illegal move by White: {move_str}")
                        break
                except:
                    print(f"❌ Invalid move by White: {move_str}")
                    break
            else:
                # Black to move
                black_engine.board.set_fen(board.fen())
                
                start_time = time.time()
                move_str = black_engine.get_move()
                move_time = time.time() - start_time
                
                if isinstance(black_engine, EngineV31):
                    v31_times.append(move_time)
                else:
                    v30_times.append(move_time)
                
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in board.legal_moves:
                        board.push(move)
                        moves_played += 1
                        print(f"⚫ {move_str} ({move_time:.2f}s)")
                    else:
                        print(f"❌ Illegal move by Black: {move_str}")
                        break
                except:
                    print(f"❌ Invalid move by Black: {move_str}")
                    break
        
        # Determine result
        if board.is_checkmate():
            if board.turn == chess.WHITE:
                winner = "Black"
            else:
                winner = "White"
        else:
            winner = "Draw"
        
        print(f"🎯 Result: {winner}")
        
        # Update scores
        if game_num % 2 == 0:  # V3.1 was white
            if winner == "White":
                v31_wins += 1
            elif winner == "Black":
                v30_wins += 1
            else:
                draws += 1
        else:  # V3.0 was white
            if winner == "White":
                v30_wins += 1
            elif winner == "Black":
                v31_wins += 1
            else:
                draws += 1
    
    # Final results
    print("\n" + "🏆" + "="*60 + "🏆")
    print("               SPEED TEST RESULTS")
    print("🏆" + "="*60 + "🏆")
    
    v31_points = v31_wins + (draws * 0.5)
    v30_points = v30_wins + (draws * 0.5)
    
    print(f"📊 MATCH SCORE:")
    print(f"   🆕 V3.1 Speed: {v31_wins}W-{v30_wins}L-{draws}D = {v31_points} points")
    print(f"   🏆 V3.0 Champion: {v30_wins}W-{v31_wins}L-{draws}D = {v30_points} points")
    
    print(f"\n⚡ SPEED COMPARISON:")
    v31_avg = sum(v31_times) / len(v31_times) if v31_times else 0
    v30_avg = sum(v30_times) / len(v30_times) if v30_times else 0
    
    print(f"   🆕 V3.1 Average: {v31_avg:.2f}s per move")
    print(f"   🏆 V3.0 Average: {v30_avg:.2f}s per move")
    
    if v31_avg > 0 and v30_avg > 0:
        speedup = ((v30_avg - v31_avg) / v30_avg) * 100
        print(f"   🚀 Speed improvement: {speedup:.1f}%")
    
    # Verdict
    if v31_points > v30_points:
        print(f"\n🎉 WINNER: V3.1 Speed Optimized!")
        print("✨ Faster AND stronger!")
    elif v30_points > v31_points:
        print(f"\n🏆 WINNER: V3.0 Champion!")
        print("⚠️  Speed optimizations may have hurt strength")
    else:
        print(f"\n🤝 TIE GAME!")
        print("👍 Speed optimizations maintained strength")
    
    return {
        "v31_score": v31_points,
        "v30_score": v30_points,
        "v31_speed": v31_avg,
        "v30_speed": v30_avg
    }

if __name__ == "__main__":
    results = play_speed_test()
    print(f"\n⚡ CONCLUSION: V3.1 is {((results['v30_speed'] - results['v31_speed']) / results['v30_speed'] * 100):.1f}% faster")
    print(f"🎯 STRENGTH: {'Improved' if results['v31_score'] > results['v30_score'] else 'Maintained' if results['v31_score'] == results['v30_score'] else 'Slightly reduced'}") 