#!/usr/bin/env python3
"""
Move Quality Test: Neural Network vs Stockfish Move Selection
This script tests how good the neural network's move choices are compared to Stockfish's recommendations.
"""

import chess
import chess.pgn
from stockfish import Stockfish
from chess_engine import Engine
import random

# Initialize Stockfish
try:
    stockfish_paths = [
        "/opt/homebrew/bin/stockfish",  # Homebrew on Apple Silicon
        "/usr/local/bin/stockfish",    # Homebrew on Intel Mac
        "/usr/bin/stockfish",          # Linux
        "stockfish"                    # System PATH
    ]
    
    stockfish = None
    for path in stockfish_paths:
        try:
            stockfish = Stockfish(path=path, depth=15, parameters={"Threads": 2, "Hash": 1024})
            print(f"✅ Stockfish initialized at: {path}")
            break
        except:
            continue
    
    if stockfish is None:
        raise Exception("Could not find Stockfish binary")
        
except Exception as e:
    print(f"❌ Failed to initialize Stockfish: {e}")
    exit(1)

def get_stockfish_top_moves(fen, num_moves=5):
    """Get top moves from Stockfish with their evaluations"""
    try:
        stockfish.set_fen_position(fen)
        top_moves = stockfish.get_top_moves(num_moves)
        
        # Convert to a more readable format
        moves_with_eval = []
        for move_info in top_moves:
            move = move_info['Move']
            if move_info['Mate'] is not None:
                if move_info['Mate'] > 0:
                    eval_score = f"Mate in {move_info['Mate']}"
                else:
                    eval_score = f"Mated in {abs(move_info['Mate'])}"
            else:
                eval_score = f"{move_info['Centipawn']/100.0:+.2f}"
            
            moves_with_eval.append({
                'move': move,
                'evaluation': eval_score,
                'centipawns': move_info['Centipawn'],
                'mate': move_info['Mate']
            })
        
        return moves_with_eval
    except Exception as e:
        print(f"Failed to get Stockfish moves: {e}")
        return []

def test_position(fen, position_name):
    """Test neural network move selection against Stockfish for a given position"""
    print(f"\n{'='*80}")
    print(f"🧠 TESTING POSITION: {position_name}")
    print(f"{'='*80}")
    print(f"FEN: {fen}")
    
    # Get board visualization
    board = chess.Board(fen)
    print(f"\nPosition:")
    print(board)
    print(f"To move: {'White' if board.turn == chess.WHITE else 'Black'}")
    
    # Get neural network's choice
    print(f"\n🧠 Neural Network Analysis:")
    try:
        engine = Engine(fen)
        neural_move = engine.get_move()
        print(f"Selected move: {neural_move}")
    except Exception as e:
        print(f"❌ Neural network failed: {e}")
        return
    
    # Get Stockfish's top moves
    print(f"\n🐟 Stockfish Analysis:")
    stockfish_moves = get_stockfish_top_moves(fen, 5)
    
    if not stockfish_moves:
        print("❌ Failed to get Stockfish analysis")
        return
    
    print("Top 5 moves according to Stockfish:")
    for i, move_info in enumerate(stockfish_moves, 1):
        print(f"  {i}. {move_info['move']} ({move_info['evaluation']})")
    
    # Check where neural network's move ranks
    neural_move_rank = None
    neural_move_eval = None
    
    for i, move_info in enumerate(stockfish_moves):
        if move_info['move'] == neural_move:
            neural_move_rank = i + 1
            neural_move_eval = move_info['evaluation']
            break
    
    # Evaluate the neural network's performance
    print(f"\n📊 Neural Network Performance:")
    if neural_move_rank:
        print(f"✅ Neural move '{neural_move}' ranked #{neural_move_rank} by Stockfish")
        print(f"   Stockfish evaluation: {neural_move_eval}")
        
        if neural_move_rank == 1:
            performance = "🏆 EXCELLENT - Found best move!"
        elif neural_move_rank <= 2:
            performance = "🥈 VERY GOOD - Top 2 move"
        elif neural_move_rank <= 3:
            performance = "🥉 GOOD - Top 3 move"
        elif neural_move_rank <= 5:
            performance = "⚠️  OKAY - Top 5 move"
        else:
            performance = "❌ POOR - Not in top 5"
    else:
        print(f"❌ Neural move '{neural_move}' NOT in Stockfish's top 5")
        
        # Get evaluation of the neural network's move
        try:
            test_board = chess.Board(fen)
            test_move = chess.Move.from_uci(neural_move)
            if test_move in test_board.legal_moves:
                test_board.push(test_move)
                stockfish.set_fen_position(test_board.fen())
                eval_info = stockfish.get_evaluation()
                
                if eval_info['type'] == 'cp':
                    # Flip evaluation since we made a move
                    neural_eval = f"{-eval_info['value']/100.0:+.2f}"
                elif eval_info['type'] == 'mate':
                    if eval_info['value'] > 0:
                        neural_eval = f"Mated in {eval_info['value']}"
                    else:
                        neural_eval = f"Mate in {abs(eval_info['value'])}"
                
                print(f"   Stockfish evaluation of neural move: {neural_eval}")
                
                # Compare to best move
                best_move_eval = stockfish_moves[0]['centipawns'] if stockfish_moves[0]['centipawns'] else 0
                neural_move_centipawns = -eval_info['value'] if eval_info['type'] == 'cp' else 0
                
                if abs(best_move_eval - neural_move_centipawns) <= 50:
                    performance = "✅ GOOD - Close to best move (±50cp)"
                elif abs(best_move_eval - neural_move_centipawns) <= 100:
                    performance = "⚠️  OKAY - Reasonable move (±100cp)"
                else:
                    performance = "❌ POOR - Significantly worse move"
            else:
                print(f"   ❌ Neural move is ILLEGAL!")
                performance = "❌ CRITICAL ERROR - Illegal move"
                
        except Exception as e:
            print(f"   Failed to evaluate neural move: {e}")
            performance = "❓ UNKNOWN"
    
    print(f"   {performance}")
    
    return {
        'position': position_name,
        'fen': fen,
        'neural_move': neural_move,
        'neural_rank': neural_move_rank,
        'stockfish_top': stockfish_moves[0]['move'] if stockfish_moves else None,
        'performance': performance
    }

def main():
    """Main function to test various positions"""
    print("🚀 Testing Neural Network Move Selection Quality")
    
    # Test positions - mix of opening, middlegame, and tactical positions
    test_positions = [
        {
            "name": "Opening: After 1.e4 e5 2.Nf3",
            "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
        },
        {
            "name": "Middlegame: Tactical position",
            "fen": "r3k2r/pp1b1ppp/2n2n2/1B2N3/Q2Pq3/2P1B3/P4PPP/R3K2R w KQkq - 1 13"
        },
        {
            "name": "Opening: Sicilian Defense",
            "fen": "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"
        },
        {
            "name": "Middlegame: Material imbalance",
            "fen": "7r/pp1k1p2/2pN1npp/8/8/BP6/P4PnP/2KR4 w - - 0 23"
        },
        {
            "name": "Endgame: King and pawn",
            "fen": "8/8/1k6/8/8/8/4P3/3K4 w - - 0 1"
        },
        {
            "name": "Tactical: Pin situation",
            "fen": "5k2/r2p4/3Np1RP/2PnP3/5P2/1p1N3P/1P1K4/r7 b - - 0 47"
        }
    ]
    
    results = []
    
    for position in test_positions:
        result = test_position(position['fen'], position['name'])
        if result:
            results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 SUMMARY OF NEURAL NETWORK PERFORMANCE")
    print(f"{'='*80}")
    
    if results:
        total_tests = len(results)
        top_1_count = sum(1 for r in results if r['neural_rank'] == 1)
        top_3_count = sum(1 for r in results if r['neural_rank'] and r['neural_rank'] <= 3)
        top_5_count = sum(1 for r in results if r['neural_rank'] and r['neural_rank'] <= 5)
        
        print(f"Total positions tested: {total_tests}")
        print(f"🏆 Best moves found: {top_1_count}/{total_tests} ({top_1_count/total_tests*100:.1f}%)")
        print(f"🥉 Top 3 moves: {top_3_count}/{total_tests} ({top_3_count/total_tests*100:.1f}%)")
        print(f"📋 Top 5 moves: {top_5_count}/{total_tests} ({top_5_count/total_tests*100:.1f}%)")
        
        print(f"\nDetailed results:")
        for i, result in enumerate(results, 1):
            rank_str = f"#{result['neural_rank']}" if result['neural_rank'] else "Not in top 5"
            print(f"  {i}. {result['position']}: {result['neural_move']} ({rank_str})")
        
        # Overall assessment
        if top_1_count / total_tests >= 0.5:
            assessment = "🏆 EXCELLENT - Neural network finds best moves frequently"
        elif top_3_count / total_tests >= 0.7:
            assessment = "✅ GOOD - Neural network finds strong moves consistently"  
        elif top_5_count / total_tests >= 0.6:
            assessment = "⚠️  AVERAGE - Neural network finds decent moves"
        else:
            assessment = "❌ POOR - Neural network struggles to find good moves"
        
        print(f"\n{assessment}")
    
    print(f"\n✅ Move quality test completed!")

if __name__ == "__main__":
    main() 