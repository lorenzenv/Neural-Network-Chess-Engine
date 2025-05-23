#!/usr/bin/env python3
"""
⚔️ V4.0 vs V3.2 HEAD-TO-HEAD BATTLE
V4.0 Pure Neural Power vs V3.2 Anti-Repetition
"""

import chess
import time

# Import both engines
try:
    from chess_engine_v4 import EngineV4
    print("✅ Loaded V4.0 Pure Neural Power")
except ImportError:
    print("❌ Could not import V4.0")
    exit(1)

try:
    from chess_engine_v3 import EngineV3
    print("✅ Loaded V3.2 Anti-Repetition")
except ImportError:
    print("❌ Could not import V3.2")
    exit(1)

def play_head_to_head(engine1, engine2, max_moves=80):
    """Play a single game between two engines"""
    board = chess.Board()
    moves_played = 0
    move_times = {"engine1": [], "engine2": []}
    
    print(f"\n⚔️ BATTLE: {engine1.name} (Black) vs {engine2.name} (White)")
    print("=" * 60)
    
    while not board.is_game_over() and moves_played < max_moves:
        current_engine = engine1 if board.turn == chess.BLACK else engine2
        engine_key = "engine1" if board.turn == chess.BLACK else "engine2"
        
        print(f"\n{'Black' if board.turn == chess.BLACK else 'White'} to move ({current_engine.name})")
        
        # Set the position
        current_engine.board.set_fen(board.fen())
        
        # Time the move
        start_time = time.time()
        move_str = current_engine.get_move()
        end_time = time.time()
        
        move_time = end_time - start_time
        move_times[engine_key].append(move_time)
        
        if move_str == "checkmate":
            print(f"🏳️ {current_engine.name} resigned")
            break
        
        try:
            move = chess.Move.from_uci(move_str)
            if move in board.legal_moves:
                board.push(move)
                moves_played += 1
                print(f"⚡ {current_engine.name} played: {move_str} ({move_time:.2f}s)")
            else:
                print(f"❌ Illegal move by {current_engine.name}: {move_str}")
                break
        except:
            print(f"❌ Invalid move by {current_engine.name}: {move_str}")
            break
    
    # Determine result
    if board.is_checkmate():
        if board.turn == chess.BLACK:
            result = "1-0"  # White wins
            winner = engine2.name
        else:
            result = "0-1"  # Black wins
            winner = engine1.name
    elif board.is_stalemate() or board.is_insufficient_material() or moves_played >= max_moves:
        result = "1/2-1/2"  # Draw
        winner = "Draw"
    else:
        result = "1/2-1/2"  # Draw (incomplete)
        winner = "Draw"
    
    print(f"\n🏆 RESULT: {result} - {winner}")
    print(f"📊 Game length: {moves_played} moves")
    
    # Performance stats
    if move_times["engine1"]:
        avg_time_1 = sum(move_times["engine1"]) / len(move_times["engine1"])
        print(f"⏱️ {engine1.name} average time: {avg_time_1:.2f}s per move")
    
    if move_times["engine2"]:
        avg_time_2 = sum(move_times["engine2"]) / len(move_times["engine2"])
        print(f"⏱️ {engine2.name} average time: {avg_time_2:.2f}s per move")
    
    return result, winner, moves_played, move_times

def run_match(games=2):
    """Run a match between V4.0 and V3.2"""
    print("🏆" + "="*60 + "🏆")
    print("         V4.0 PURE NEURAL POWER vs V3.2 ANTI-REPETITION")
    print("🏆" + "="*60 + "🏆")
    
    # Create engines
    engine_v4 = EngineV4("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    engine_v3 = EngineV3("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    
    print(f"\n🧠 {engine_v4.name}")
    v4_info = engine_v4.get_version_info()
    print(f"   Version: {v4_info['version']}")
    print(f"   Key Features: {v4_info['features'][:3]}")  # Show first 3 features
    
    print(f"\n🔄 {engine_v3.name}")
    v3_info = engine_v3.get_version_info()
    print(f"   Version: {v3_info['version']}")
    print(f"   Key Features: {v3_info['features'][:3]}")  # Show first 3 features
    
    # Match results
    results = {
        engine_v4.name: {"wins": 0, "losses": 0, "draws": 0, "points": 0, "times": []},
        engine_v3.name: {"wins": 0, "losses": 0, "draws": 0, "points": 0, "times": []}
    }
    
    # Play games
    for game_num in range(games):
        print(f"\n🎮 GAME {game_num + 1}/{games}")
        
        if game_num % 2 == 0:
            # V4.0 plays as Black, V3.2 as White
            result, winner, moves, times = play_head_to_head(engine_v4, engine_v3)
            
            if times["engine1"]:  # V4.0 times
                results[engine_v4.name]["times"].extend(times["engine1"])
            if times["engine2"]:  # V3.2 times
                results[engine_v3.name]["times"].extend(times["engine2"])
        else:
            # V3.2 plays as Black, V4.0 as White
            result, winner, moves, times = play_head_to_head(engine_v3, engine_v4)
            
            if times["engine1"]:  # V3.2 times
                results[engine_v3.name]["times"].extend(times["engine1"])
            if times["engine2"]:  # V4.0 times
                results[engine_v4.name]["times"].extend(times["engine2"])
        
        # Update results
        if winner == "Draw":
            results[engine_v4.name]["draws"] += 1
            results[engine_v3.name]["draws"] += 1
            results[engine_v4.name]["points"] += 0.5
            results[engine_v3.name]["points"] += 0.5
        elif winner == engine_v4.name:
            results[engine_v4.name]["wins"] += 1
            results[engine_v3.name]["losses"] += 1
            results[engine_v4.name]["points"] += 1
        else:
            results[engine_v3.name]["wins"] += 1
            results[engine_v4.name]["losses"] += 1
            results[engine_v3.name]["points"] += 1
    
    # Final results
    print("\n" + "🏆" + "="*60 + "🏆")
    print("                    MATCH RESULTS")
    print("🏆" + "="*60 + "🏆")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]["points"], reverse=True)
    
    print(f"{'Engine':<35}{'W':<4}{'L':<4}{'D':<4}{'Points':<8}{'Avg Time':<10}")
    print("-" * 70)
    
    for engine_name, stats in sorted_results:
        avg_time = sum(stats["times"]) / len(stats["times"]) if stats["times"] else 0
        if engine_name == sorted_results[0][0]:
            trophy = "🥇"
        else:
            trophy = "🥈"
        
        print(f"{trophy} {engine_name:<33}{stats['wins']:<4}{stats['losses']:<4}{stats['draws']:<4}{stats['points']:<8}{avg_time:.2f}s")
    
    # Champion
    champion_name, champion_stats = sorted_results[0]
    print(f"\n🎉 WINNER: {champion_name}")
    print(f"🏆 Final Score: {champion_stats['points']}/{games} points")
    print(f"📊 Record: {champion_stats['wins']}W-{champion_stats['losses']}L-{champion_stats['draws']}D")
    
    # Performance comparison
    v4_times = results[engine_v4.name]["times"]
    v3_times = results[engine_v3.name]["times"]
    
    if v4_times and v3_times:
        v4_avg = sum(v4_times) / len(v4_times)
        v3_avg = sum(v3_times) / len(v3_times)
        
        print(f"\n⚡ PERFORMANCE COMPARISON:")
        print(f"   {engine_v4.name}: {v4_avg:.2f}s average")
        print(f"   {engine_v3.name}: {v3_avg:.2f}s average")
        
        if v4_avg < v3_avg:
            speed_diff = ((v3_avg - v4_avg) / v3_avg) * 100
            print(f"   🏃 V4.0 is {speed_diff:.1f}% FASTER than V3.2!")
        else:
            speed_diff = ((v4_avg - v3_avg) / v4_avg) * 100
            print(f"   🏃 V3.2 is {speed_diff:.1f}% faster than V4.0")
    
    return champion_name, results

if __name__ == "__main__":
    print("⚔️ Starting V4.0 vs V3.2 Head-to-Head Battle!")
    champion, results = run_match(games=2)  # 2 games for speed
    print(f"\n🎯 The champion is: {champion}!") 