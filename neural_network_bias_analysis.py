#!/usr/bin/env python3
"""
Neural Network Color Bias Analysis
Based on understanding of the DeepChess model training methodology.

Key Insights from training code analysis:
1. Model is trained on position comparisons, not absolute evaluations
2. Training uses (WWinning, BWinning) position pairs 
3. Output: P(position1 is better for White than position2)
4. Critical: The interpretation of comparison results affects color bias
"""

import chess
from chess_engine import Engine, NNEvaluator
import numpy as np
from util import *

def analyze_nn_comparison_behavior():
    """Analyze how the NN comparison function behaves with different color positions"""
    print("🔍 NEURAL NETWORK COMPARISON BEHAVIOR ANALYSIS")
    print("="*70)
    
    nn_eval = NNEvaluator()
    
    # Test positions
    test_cases = [
        # Case 1: Starting position vs slight White advantage
        {
            "name": "Starting vs White advantage",
            "fen1": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting
            "fen2": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",  # e4 played
        },
        # Case 2: Starting position vs slight Black advantage  
        {
            "name": "Starting vs Black advantage", 
            "fen1": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting
            "fen2": "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",  # Nf6 developing
        },
        # Case 3: White to move tactical vs Black to move tactical
        {
            "name": "White tactical vs Black tactical",
            "fen1": "r3k2r/pp1b1ppp/2n2n2/1B2N3/Q2Pq3/2P1B3/P4PPP/R3K2R w KQkq - 1 13",  # White to move
            "fen2": "r3k2r/p4ppp/2p1b3/q2pQ3/1b2n3/2N2N2/PP1B1PPP/R3K2R b KQkq - 1 13",  # Black to move (artificial flip)
        }
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        print(f"FEN1: {case['fen1']}")
        print(f"FEN2: {case['fen2']}")
        
        try:
            # Get raw NN comparison
            comparison = nn_eval.compare_positions(case['fen1'], case['fen2'])
            print(f"Raw NN comparison: {comparison:.4f}")
            print(f"Interpretation: FEN1 is {'BETTER' if comparison > 0.5 else 'WORSE'} for White than FEN2")
            
            results.append({
                'name': case['name'],
                'fen1': case['fen1'], 
                'fen2': case['fen2'],
                'comparison': comparison,
                'fen1_better_for_white': comparison > 0.5
            })
            
        except Exception as e:
            print(f"Error: {e}")
    
    return results

def analyze_move_evaluation_bias():
    """Analyze how move evaluation differs between White and Black"""
    print(f"\n{'='*70}")
    print("🎯 MOVE EVALUATION BIAS ANALYSIS")
    print("="*70)
    
    nn_eval = NNEvaluator()
    
    # Test the same tactical position for both colors
    test_positions = [
        {
            "name": "Italian Game - White to move",
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4",
            "expected_color": "White"
        },
        {
            "name": "Italian Game - Black to move", 
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 4 4",
            "expected_color": "Black"
        }
    ]
    
    for pos in test_positions:
        print(f"\n--- {pos['name']} ---")
        print(f"FEN: {pos['fen']}")
        
        try:
            # Test with our engine
            engine = Engine(pos['fen'])
            
            # Get some move options
            board = chess.Board(pos['fen'])
            legal_moves = list(board.legal_moves)[:5]  # First 5 legal moves
            
            print(f"To move: {pos['expected_color']}")
            print(f"Testing {len(legal_moves)} moves:")
            
            current_fen = pos['fen']
            move_fens = []
            
            for move in legal_moves:
                board.push(move)
                move_fens.append(board.fen())
                board.pop()
            
            # Test move comparison
            player_is_white = (board.turn == chess.WHITE)
            scores = nn_eval.evaluate_move_comparison(current_fen, move_fens, player_is_white)
            
            print(f"Move comparison scores (higher = better for {pos['expected_color']}):")
            for i, (move, score) in enumerate(zip(legal_moves, scores)):
                print(f"  {move}: {score:.4f}")
            
            # Analyze distribution
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            
            print(f"Score statistics:")
            print(f"  Average: {avg_score:.4f}")
            print(f"  Range: {min_score:.4f} - {max_score:.4f}")
            print(f"  Spread: {max_score - min_score:.4f}")
            
        except Exception as e:
            print(f"Error: {e}")

def analyze_interpretation_logic():
    """Analyze how we interpret NN comparison results for move evaluation"""
    print(f"\n{'='*70}")
    print("🧮 INTERPRETATION LOGIC ANALYSIS")
    print("="*70)
    
    print("Current interpretation in evaluate_move_comparison():")
    print("""
    if player_is_white:
        # For white: if current > move_result, we want lower score (bad move)
        # If move_result > current, we want higher score (good move)  
        score = 1.0 - comparison  # Invert because we want high scores for good moves
    else:
        # For black: similar logic but inverted perspective
        score = comparison
    """)
    
    print("\nThis interpretation assumes:")
    print("1. comparison = P(move_result is better for White than current)")
    print("2. For White: high comparison (move helps White) → high score (good move)")  
    print("3. For Black: low comparison (move hurts White) → high score (good move)")
    
    print("\nPOTENTIAL BIAS SOURCE:")
    print("If the NN model has any systematic bias in how it evaluates")
    print("White vs Black positions, this interpretation will amplify it.")
    
    # Test with a simple example
    nn_eval = NNEvaluator()
    
    # Starting position
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    # After 1.e4
    after_e4_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    
    print(f"\nTest Case: Starting position vs 1.e4")
    comparison = nn_eval.compare_positions(after_e4_fen, start_fen)
    print(f"NN comparison (after_e4 vs start): {comparison:.4f}")
    print(f"Interpretation: 1.e4 is {'GOOD' if comparison > 0.5 else 'BAD'} for White")
    
    # Now test the reverse (this should be neutral/symmetric)
    comparison_reverse = nn_eval.compare_positions(start_fen, after_e4_fen) 
    print(f"NN comparison (start vs after_e4): {comparison_reverse:.4f}")
    print(f"Expected: ~{1.0 - comparison:.4f} (should be symmetric)")
    print(f"Actual difference: {abs(comparison_reverse - (1.0 - comparison)):.4f}")
    
    if abs(comparison_reverse - (1.0 - comparison)) > 0.05:
        print("⚠️  NON-SYMMETRIC BEHAVIOR DETECTED - Possible NN bias!")
    else:
        print("✅ Symmetric behavior - NN comparison logic is sound")

def run_bias_analysis():
    """Run comprehensive bias analysis"""
    print("🔍 COMPREHENSIVE NEURAL NETWORK COLOR BIAS ANALYSIS")
    print("Based on DeepChess model training methodology")
    print("="*70)
    
    print("\nKEY FINDINGS FROM CODE ANALYSIS:")
    print("1. Model trained on (WWinning, BWinning) position pairs")
    print("2. Output: P(position1 better for White than position2)")
    print("3. Training data: 2,000,000 positions from CCRL database")
    print("4. Training method: Pairwise comparison with random label flipping")
    
    # Run analyses
    comparison_results = analyze_nn_comparison_behavior()
    analyze_move_evaluation_bias()
    analyze_interpretation_logic()
    
    print(f"\n{'='*70}")
    print("🎯 CONCLUSIONS AND RECOMMENDATIONS")
    print("="*70)
    
    print("Potential sources of color bias:")
    print("1. Training data imbalance (more/better White positions)")
    print("2. Model architecture bias toward certain patterns")
    print("3. Interpretation logic amplifying small biases")
    print("4. Bitboard encoding favoring piece placement")
    
    print("\nRecommended fixes:")
    print("1. Add randomization to position comparison order")
    print("2. Test with color-flipped position pairs")
    print("3. Adjust interpretation confidence thresholds")
    print("4. Consider ensemble with color-flipped predictions")

if __name__ == "__main__":
    run_bias_analysis() 