#!/usr/bin/env python3
"""
Quick Color Bias Test
Tests the same positions with colors swapped to understand the bias better.
"""

import chess
from stockfish import Stockfish
from chess_engine import Engine
import sys

# Initialize Stockfish
def initialize_stockfish():
    possible_stockfish_paths = [
        "/opt/homebrew/bin/stockfish",
        "/usr/local/bin/stockfish", 
        "/usr/bin/stockfish",
        "stockfish"
    ]
    
    for path in possible_stockfish_paths:
        try:
            stockfish = Stockfish(path=path, depth=15, parameters={"Threads": 2})
            if stockfish.is_fen_valid("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
                print(f"✅ Stockfish initialized at: {path}")
                return stockfish
        except Exception as e:
            continue
    
    print("❌ Could not initialize Stockfish")
    sys.exit(1)

stockfish = initialize_stockfish()

def flip_fen(fen: str) -> str:
    """Flip a FEN string to swap colors"""
    parts = fen.split()
    board_part = parts[0]
    
    # Flip the board vertically
    ranks = board_part.split('/')
    flipped_ranks = []
    
    for rank in reversed(ranks):
        # Swap case (white<->black)
        flipped_rank = rank.swapcase()
        flipped_ranks.append(flipped_rank)
    
    flipped_board = '/'.join(flipped_ranks)
    
    # Flip the active color
    active_color = 'b' if parts[1] == 'w' else 'w'
    
    # Flip castling rights
    castling = parts[2]
    if castling != '-':
        # Swap white/black castling rights
        new_castling = ""
        for char in castling:
            if char.isupper():
                new_castling += char.lower()
            else:
                new_castling += char.upper()
        castling = new_castling
    
    # En passant square needs to be flipped vertically
    ep_square = parts[3]
    if ep_square != '-':
        file_char = ep_square[0]
        rank_num = int(ep_square[1])
        flipped_rank = 9 - rank_num
        ep_square = f"{file_char}{flipped_rank}"
    
    return f"{flipped_board} {active_color} {castling} {ep_square} {parts[4]} {parts[5]}"

def analyze_position_pair(original_fen: str, position_name: str):
    """Analyze a position and its color-flipped version"""
    print(f"\n{'='*70}")
    print(f"🧠 ANALYZING POSITION PAIR: {position_name}")
    print(f"{'='*70}")
    
    flipped_fen = flip_fen(original_fen)
    
    positions = [
        ("Original", original_fen),
        ("Flipped", flipped_fen)
    ]
    
    results = []
    
    for variant_name, fen in positions:
        print(f"\n--- {variant_name} Position ---")
        print(f"FEN: {fen}")
        
        board = chess.Board(fen)
        player_color = "White" if board.turn == chess.WHITE else "Black"
        print(f"To move: {player_color}")
        
        # Get NN move
        try:
            engine = Engine(fen)
            nn_move = engine.get_move()
            print(f"NN move: {nn_move}")
        except Exception as e:
            print(f"❌ NN failed: {e}")
            continue
        
        # Get Stockfish analysis
        try:
            stockfish.set_fen_position(fen)
            sf_moves = stockfish.get_top_moves(3)
            if sf_moves:
                sf_best = sf_moves[0]
                print(f"SF best: {sf_best['Move']} ({sf_best['Centipawn']} cp)")
                
                # Evaluate NN move
                board_copy = chess.Board(fen)
                if nn_move and nn_move not in ["checkmate", "draw", "no_legal_moves_or_error"]:
                    try:
                        move_obj = chess.Move.from_uci(nn_move)
                        if move_obj in board_copy.legal_moves:
                            board_copy.push(move_obj)
                            stockfish.set_fen_position(board_copy.fen())
                            nn_eval_info = stockfish.get_top_moves(1)
                            if nn_eval_info:
                                nn_eval = nn_eval_info[0]['Centipawn']
                                eval_diff = sf_best['Centipawn'] - nn_eval
                                print(f"NN eval: {nn_eval} cp")
                                print(f"Difference: {eval_diff} cp")
                                
                                results.append({
                                    'variant': variant_name,
                                    'color': player_color,
                                    'nn_move': nn_move,
                                    'sf_best': sf_best['Move'],
                                    'sf_eval': sf_best['Centipawn'],
                                    'nn_eval': nn_eval,
                                    'eval_diff': eval_diff
                                })
                    except Exception as e:
                        print(f"Error evaluating NN move: {e}")
        except Exception as e:
            print(f"❌ SF analysis failed: {e}")
    
    # Compare results
    if len(results) == 2:
        print(f"\n📊 COMPARISON:")
        orig = results[0]
        flip = results[1]
        
        print(f"Original ({orig['color']}): {orig['eval_diff']} cp difference")
        print(f"Flipped  ({flip['color']}): {flip['eval_diff']} cp difference")
        
        if orig['color'] != flip['color']:  # Different colors
            if abs(orig['eval_diff']) < abs(flip['eval_diff']):
                better_color = orig['color']
                worse_color = flip['color']
            else:
                better_color = flip['color']
                worse_color = orig['color']
            print(f"Engine performs better as: {better_color}")
    
    return results

def run_quick_color_test():
    """Run the quick color bias test"""
    print("🔍 QUICK COLOR BIAS INVESTIGATION")
    print("Testing the same positions with colors swapped")
    
    # Test positions from our previous analysis
    test_positions = [
        ("middlegame_tactical", "r3k2r/pp1b1ppp/2n2n2/1B2N3/Q2Pq3/2P1B3/P4PPP/R3K2R w KQkq - 1 13"),
        ("king_pawn_endgame", "8/8/1k6/8/8/8/4P3/3K4 w - - 0 1"),
        ("opening_position", "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"),
    ]
    
    all_results = []
    
    for pos_name, fen in test_positions:
        results = analyze_position_pair(fen, pos_name)
        all_results.extend(results)
    
    # Overall analysis
    print(f"\n{'='*70}")
    print("📊 OVERALL RESULTS")
    print(f"{'='*70}")
    
    white_diffs = [r['eval_diff'] for r in all_results if r['color'] == 'White']
    black_diffs = [r['eval_diff'] for r in all_results if r['color'] == 'Black']
    
    if white_diffs and black_diffs:
        white_avg = sum(white_diffs) / len(white_diffs)
        black_avg = sum(black_diffs) / len(black_diffs)
        
        print(f"White positions - Average eval difference: {white_avg:.1f} cp")
        print(f"Black positions - Average eval difference: {black_avg:.1f} cp")
        print(f"Bias (White - Black): {white_avg - black_avg:.1f} cp")
        
        if abs(white_avg - black_avg) > 20:
            if black_avg < white_avg:
                print("🎯 CONCLUSION: Engine performs better as BLACK")
            else:
                print("🎯 CONCLUSION: Engine performs better as WHITE")
        else:
            print("🎯 CONCLUSION: No significant color bias detected")

if __name__ == "__main__":
    run_quick_color_test() 