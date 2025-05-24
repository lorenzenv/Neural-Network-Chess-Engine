#!/usr/bin/env python3
"""
Color Bias Investigation Script
Analyzes whether the chess engine performs differently when playing as White vs Black.
"""

import chess
import chess.engine
from stockfish import Stockfish
from chess_engine import Engine
import random
import sys
import os
import logging
import json
import time
from typing import List, Dict, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

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
            stockfish = Stockfish(path=path, depth=18, parameters={"Threads": 2})
            if stockfish.is_fen_valid("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
                logger.info(f"✅ Stockfish initialized at: {path} with depth 18")
                return stockfish
        except Exception as e:
            continue
    
    logger.error("❌ Could not initialize Stockfish. Please ensure it's installed.")
    sys.exit(1)

stockfish = initialize_stockfish()

def get_stockfish_best_move_eval(fen: str) -> dict | None:
    """Gets Stockfish's best move and evaluation"""
    try:
        stockfish.set_fen_position(fen)
        top_move_info = stockfish.get_top_moves(1)
        if not top_move_info:
            return None
        
        best_move_uci = top_move_info[0]['Move']
        eval_data = top_move_info[0]
        
        value_cp = None
        if eval_data['Mate'] is not None:
            value_cp = eval_data['Mate'] * 10000
        elif eval_data['Centipawn'] is not None:
            value_cp = eval_data['Centipawn']
        
        return {"move_uci": best_move_uci, "value_cp": value_cp}
    except Exception as e:
        logger.error(f"Error getting Stockfish analysis for {fen}: {e}")
        return None

def get_stockfish_eval_for_move(fen: str, move_uci: str) -> dict | None:
    """Gets Stockfish's evaluation after a specific move"""
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return {"type": "illegal", "value_cp": None}
        
        board.push(move)
        stockfish.set_fen_position(board.fen())
        top_move_info = stockfish.get_top_moves(1)
        
        if not top_move_info:
            return {"type": "terminal", "value_cp": 0}
        
        eval_data = top_move_info[0]
        value_cp = None
        
        if eval_data['Mate'] is not None:
            value_cp = eval_data['Mate'] * 10000
        elif eval_data['Centipawn'] is not None:
            value_cp = eval_data['Centipawn']
        
        return {"type": "normal", "value_cp": value_cp}
    except Exception as e:
        logger.error(f"Error evaluating move {move_uci} on {fen}: {e}")
        return None

def test_position_with_color_analysis(fen: str, position_name: str) -> dict:
    """Test a position and return detailed analysis"""
    board = chess.Board(fen)
    player_color = "White" if board.turn == chess.WHITE else "Black"
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🧠 TESTING: {position_name}")
    logger.info(f"FEN: {fen}")
    logger.info(f"To move: {player_color}")
    
    # Get Neural Network's move
    try:
        engine = Engine(fen)
        neural_move_uci = engine.get_move()
        if not neural_move_uci or neural_move_uci in ["checkmate", "draw", "no_legal_moves_or_error"]:
            logger.warning(f"Engine returned special status: {neural_move_uci}")
            return {"status": "NN_ERROR", "color": player_color}
    except Exception as e:
        logger.error(f"❌ Neural network failed: {e}")
        return {"status": "NN_ERROR", "color": player_color}

    logger.info(f"NN Selected move: {neural_move_uci}")

    # Get Stockfish analysis
    sf_best_data = get_stockfish_best_move_eval(fen)
    if not sf_best_data or sf_best_data.get("value_cp") is None:
        logger.error("❌ Failed to get Stockfish best move")
        return {"status": "SF_ERROR", "color": player_color}
    
    nn_move_data = get_stockfish_eval_for_move(fen, neural_move_uci)
    if not nn_move_data or nn_move_data.get("value_cp") is None:
        logger.error(f"❌ Failed to evaluate NN move: {neural_move_uci}")
        return {"status": "SF_EVAL_ERROR", "color": player_color}

    logger.info(f"SF best move: {sf_best_data['move_uci']} (CP: {sf_best_data['value_cp']})")
    logger.info(f"SF eval of NN move: {neural_move_uci} (CP: {nn_move_data['value_cp']})")

    # Calculate evaluation difference (from player's perspective)
    sf_best_cp = sf_best_data['value_cp']
    nn_move_cp = nn_move_data['value_cp']
    
    eval_diff = sf_best_cp - nn_move_cp  # Positive means SF's move was better
    
    # Performance categorization
    performance = "UNKNOWN"
    if abs(eval_diff) <= 10:
        performance = "EXCELLENT"
    elif abs(eval_diff) <= 30:
        performance = "VERY_GOOD"
    elif abs(eval_diff) <= 60:
        performance = "GOOD"
    elif abs(eval_diff) <= 100:
        performance = "OKAY"
    elif abs(eval_diff) <= 200:
        performance = "POOR"
    else:
        performance = "BLUNDER"
    
    logger.info(f"Evaluation difference: {eval_diff} cp")
    logger.info(f"Performance: {performance}")

    return {
        "status": "OK",
        "position_name": position_name,
        "fen": fen,
        "color": player_color,
        "nn_move": neural_move_uci,
        "sf_best_move": sf_best_data['move_uci'],
        "sf_best_cp": sf_best_cp,
        "nn_move_cp": nn_move_cp,
        "eval_diff": eval_diff,
        "performance": performance
    }

def generate_balanced_test_positions(count_per_color: int = 25) -> List[Dict]:
    """Generate test positions balanced between White and Black to move"""
    positions = []
    
    logger.info(f"🎯 Generating {count_per_color} positions for each color...")
    
    # Start with some known good positions
    base_positions = [
        # Opening positions
        {"fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "name": "king_pawn_opening_white"},
        {"fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2", "name": "king_pawn_opening_black"},
        
        # Middlegame positions  
        {"fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4", "name": "italian_middlegame_white"},
        {"fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 4 4", "name": "italian_middlegame_black"},
        
        # Tactical positions
        {"fen": "r3k2r/pp3ppp/2n1bn2/2pp4/4P3/2NP1N2/PPP2PPP/R1B1K2R w KQkq - 0 8", "name": "tactical_white"},
        {"fen": "r3k2r/pp3ppp/2n1bn2/2pp4/4P3/2NP1N2/PPP2PPP/R1B1K2R b KQkq - 0 8", "name": "tactical_black"},
        
        # Endgame positions
        {"fen": "8/8/3k4/8/3K4/8/4P3/8 w - - 0 1", "name": "king_pawn_endgame_white"},
        {"fen": "8/4p3/8/3k4/8/3K4/8/8 b - - 0 1", "name": "king_pawn_endgame_black"},
    ]
    
    # Add the base positions
    positions.extend(base_positions)
    
    # Generate additional random positions
    white_count = len([p for p in positions if "white" in p["name"]])
    black_count = len([p for p in positions if "black" in p["name"]])
    
    # Generate more positions to reach target counts
    for attempt in range(200):  # Try many random games
        try:
            board = chess.Board()
            
            # Play random moves to get to interesting positions
            moves_played = 0
            max_moves = random.randint(10, 40)
            
            while moves_played < max_moves and not board.is_game_over():
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break
                    
                move = random.choice(legal_moves)
                board.push(move)
                moves_played += 1
                
                # Check if this is an interesting position to test
                if moves_played >= 8 and moves_played % 3 == 0:  # Sample every 3 moves after move 8
                    piece_count = len(board.piece_map())
                    
                    # Skip very unbalanced positions
                    if piece_count < 8:
                        continue
                        
                    current_fen = board.fen()
                    color = "white" if board.turn == chess.WHITE else "black"
                    
                    # Check if we need more positions of this color
                    if (color == "white" and white_count < count_per_color) or \
                       (color == "black" and black_count < count_per_color):
                        
                        # Quick check that Stockfish can analyze this position
                        if stockfish.is_fen_valid(current_fen):
                            stockfish.set_fen_position(current_fen)
                            if stockfish.get_top_moves(1):  # Ensure it's analyzable
                                position_name = f"random_{color}_{len(positions)+1}"
                                positions.append({
                                    "fen": current_fen,
                                    "name": position_name
                                })
                                
                                if color == "white":
                                    white_count += 1
                                else:
                                    black_count += 1
                                
                                logger.info(f"Added {position_name} (W:{white_count}, B:{black_count})")
                                
                                if white_count >= count_per_color and black_count >= count_per_color:
                                    return positions
                                    
        except Exception as e:
            continue  # Skip problematic positions
    
    logger.info(f"Generated {len(positions)} total positions (W:{white_count}, B:{black_count})")
    return positions

def run_color_bias_investigation():
    """Main investigation function"""
    logger.info("🔍 CHESS ENGINE COLOR BIAS INVESTIGATION")
    logger.info("="*60)
    
    # Generate test positions
    test_positions = generate_balanced_test_positions(count_per_color=20)
    
    # Test each position
    results = []
    white_results = []
    black_results = []
    
    for pos in test_positions:
        result = test_position_with_color_analysis(pos["fen"], pos["name"])
        if result["status"] == "OK":
            results.append(result)
            if result["color"] == "White":
                white_results.append(result)
            else:
                black_results.append(result)
    
    # Analyze results
    logger.info(f"\n{'='*80}")
    logger.info("📊 COLOR BIAS ANALYSIS RESULTS")
    logger.info(f"{'='*80}")
    
    logger.info(f"Total valid tests: {len(results)}")
    logger.info(f"White positions: {len(white_results)}")
    logger.info(f"Black positions: {len(black_results)}")
    
    # Performance by color
    def analyze_color_performance(results_list, color_name):
        if not results_list:
            return
            
        performance_counts = {}
        eval_diffs = []
        
        for result in results_list:
            perf = result["performance"]
            performance_counts[perf] = performance_counts.get(perf, 0) + 1
            eval_diffs.append(result["eval_diff"])
        
        avg_eval_diff = sum(eval_diffs) / len(eval_diffs)
        
        logger.info(f"\n🏳️ {color_name.upper()} PERFORMANCE:")
        logger.info(f"  Total positions: {len(results_list)}")
        logger.info(f"  Average eval difference: {avg_eval_diff:.1f} cp")
        
        excellent_count = performance_counts.get("EXCELLENT", 0)
        good_count = performance_counts.get("VERY_GOOD", 0) + performance_counts.get("GOOD", 0)
        poor_count = performance_counts.get("POOR", 0) + performance_counts.get("BLUNDER", 0)
        
        logger.info(f"  Excellent moves: {excellent_count}/{len(results_list)} ({excellent_count/len(results_list)*100:.1f}%)")
        logger.info(f"  Good+ moves: {excellent_count + good_count}/{len(results_list)} ({(excellent_count + good_count)/len(results_list)*100:.1f}%)")
        logger.info(f"  Poor/Blunder moves: {poor_count}/{len(results_list)} ({poor_count/len(results_list)*100:.1f}%)")
        
        return {
            "count": len(results_list),
            "avg_eval_diff": avg_eval_diff,
            "excellent_pct": excellent_count/len(results_list)*100,
            "good_plus_pct": (excellent_count + good_count)/len(results_list)*100,
            "poor_pct": poor_count/len(results_list)*100
        }
    
    white_stats = analyze_color_performance(white_results, "White")
    black_stats = analyze_color_performance(black_results, "Black")
    
    # Overall comparison
    if white_stats and black_stats:
        logger.info(f"\n🎯 OVERALL COMPARISON:")
        logger.info(f"  White avg eval diff: {white_stats['avg_eval_diff']:.1f} cp")
        logger.info(f"  Black avg eval diff: {black_stats['avg_eval_diff']:.1f} cp")
        logger.info(f"  Difference (W-B): {white_stats['avg_eval_diff'] - black_stats['avg_eval_diff']:.1f} cp")
        
        logger.info(f"\n  White excellent rate: {white_stats['excellent_pct']:.1f}%")
        logger.info(f"  Black excellent rate: {black_stats['excellent_pct']:.1f}%")
        logger.info(f"  Difference (B-W): {black_stats['excellent_pct'] - white_stats['excellent_pct']:.1f}%")
        
        # Statistical significance test (basic)
        if abs(white_stats['avg_eval_diff'] - black_stats['avg_eval_diff']) > 20:
            logger.info(f"\n⚠️  SIGNIFICANT COLOR BIAS DETECTED!")
            if black_stats['avg_eval_diff'] < white_stats['avg_eval_diff']:
                logger.info(f"   Engine performs better as BLACK")
            else:
                logger.info(f"   Engine performs better as WHITE")
        else:
            logger.info(f"\n✅ No significant color bias detected")
    
    # Save detailed results
    with open("color_bias_investigation_results.json", "w") as f:
        json.dump({
            "white_results": white_results,
            "black_results": black_results,
            "white_stats": white_stats,
            "black_stats": black_stats
        }, f, indent=2)
    
    logger.info(f"\n💾 Detailed results saved to: color_bias_investigation_results.json")

if __name__ == "__main__":
    run_color_bias_investigation() 