#!/usr/bin/env python3
"""
Diagnostic test to see what the engine is thinking
"""

import chess
from chess_engine_v4_fixed import EngineV4, evaluate_pos

def analyze_position(fen, description):
    """Analyze what the engine thinks about a position"""
    print(f"\n🔍 ANALYZING: {description}")
    print(f"FEN: {fen}")
    print("=" * 70)
    
    board = chess.Board(fen)
    engine = EngineV4(fen)
    
    print(f"Current turn: {'White' if board.turn == chess.WHITE else 'Black'}")
    print(f"Engine plays White: {engine.engine_plays_white}")
    print(f"Engine plays Black: {engine.engine_plays_black}")
    
    # Test neural network evaluation
    print(f"\nNeural network raw evaluation: {evaluate_pos(fen, fen):.3f}")
    print(f"Engine position evaluation: {engine.evaluate_position():.3f}")
    
    # Get top 5 legal moves and their evaluations
    legal_moves = list(board.legal_moves)
    move_evals = []
    
    print(f"\nEvaluating top moves:")
    for i, move in enumerate(legal_moves[:8]):  # Check first 8 moves
        board.push(move)
        after_fen = board.fen()
        
        # Evaluate position after move
        engine_after = EngineV4(after_fen)
        score = engine_after.evaluate_position()
        
        board.pop()
        
        move_evals.append((move, score))
        print(f"  {i+1}. {move} → {score:.3f}")
    
    # Sort by score (engine's perspective)
    move_evals.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nEngine's ranking of moves:")
    for i, (move, score) in enumerate(move_evals[:5]):
        print(f"  {i+1}. {move} (score: {score:.3f})")
    
    # Get engine's actual choice
    print(f"\nEngine's actual choice:")
    chosen_move = engine.get_move()
    print(f"Chosen: {chosen_move}")
    
    return chosen_move

def main():
    """Test various positions to see what's wrong"""
    print("🔧 ENGINE DIAGNOSTIC TEST")
    print("=" * 50)
    
    # Test obvious positions where engine should make good moves
    test_positions = [
        # Opening position - should develop pieces
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Opening - White to move"),
        
        # Black to move from opening
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "Opening - Black to move"),
        
        # Obvious capture available
        ("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2", "White can capture pawn"),
        
        # Black can capture
        ("rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 2", "Black can capture pawn"),
    ]
    
    for fen, description in test_positions:
        analyze_position(fen, description)

if __name__ == "__main__":
    main() 