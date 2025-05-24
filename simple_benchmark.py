#!/usr/bin/env python3
import json
from pure_nn_engine import Engine
import chess.engine

def simple_benchmark():
    # Load benchmark positions
    with open('benchmark_positions.json', 'r') as f:
        positions = json.load(f)
    
    # Initialize Stockfish
    stockfish = chess.engine.SimpleEngine.popen_uci("/opt/homebrew/bin/stockfish")
    
    results = []
    
    for pos in positions:
        try:
            print(f"Testing {pos['name']}...")
            fen = pos['fen']
            
            # Get NN move
            engine = Engine(fen)
            nn_move_uci = engine.get_move()
            
            # Get Stockfish move and evaluations
            board = chess.Board(fen)
            sf_result = stockfish.analyse(board, chess.engine.Limit(depth=15))
            sf_move_uci = str(sf_result['pv'][0])
            sf_best_eval = sf_result['score'].white().score(mate_score=10000)
            
            # Get Stockfish evaluation of NN move
            board.push_uci(nn_move_uci)
            nn_move_result = stockfish.analyse(board, chess.engine.Limit(depth=15))
            nn_move_eval = nn_move_result['score'].white().score(mate_score=10000)
            board.pop()
            
            # Calculate difference
            eval_diff = abs(sf_best_eval - nn_move_eval)
            
            # Categorize
            if eval_diff <= 15:
                category = "EXCELLENT"
            elif eval_diff <= 50:
                category = "VERY GOOD"
            elif eval_diff <= 100:
                category = "GOOD"
            elif eval_diff <= 200:
                category = "OKAY"
            elif eval_diff <= 500:
                category = "POOR"
            else:
                category = "BLUNDER"
            
            result = {
                'position': pos['name'],
                'nn_move': nn_move_uci,
                'sf_move': sf_move_uci,
                'sf_eval': sf_best_eval,
                'nn_eval': nn_move_eval,
                'diff': eval_diff,
                'category': category
            }
            results.append(result)
            
            print(f"  NN: {nn_move_uci} ({nn_move_eval}), SF: {sf_move_uci} ({sf_best_eval}), Diff: {eval_diff}, {category}")
            
        except Exception as e:
            print(f"Error testing {pos['name']}: {e}")
    
    # Calculate summary
    excellence_count = len([r for r in results if r['category'] == 'EXCELLENT'])
    blunder_count = len([r for r in results if r['category'] == 'BLUNDER'])
    
    print(f"\n=== SUMMARY ===")
    print(f"Total positions: {len(results)}")
    if len(results) > 0:
        print(f"Excellence rate: {excellence_count/len(results)*100:.1f}%")
        print(f"Blunder rate: {blunder_count/len(results)*100:.1f}%")
    else:
        print("No valid results")
    
    # Category breakdown
    categories = {}
    for result in results:
        cat = result['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    for cat, count in sorted(categories.items()):
        print(f"{cat}: {count}")
    
    stockfish.quit()
    return results

if __name__ == "__main__":
    simple_benchmark()