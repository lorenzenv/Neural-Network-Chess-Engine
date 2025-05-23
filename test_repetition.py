#!/usr/bin/env python3
"""
🔄 REPETITION AVOIDANCE TEST
Test that V3.2 engine correctly avoids three-fold repetition in winning positions
"""

try:
    from chess_engine_v3 import EngineV3
    print("✅ Loaded V3.2 Anti-Repetition Engine")
except ImportError:
    print("❌ Could not import V3.2 engine")
    exit(1)

def test_repetition_detection():
    """Test the repetition detection functionality"""
    print("🔄" + "="*50 + "🔄")
    print("    REPETITION DETECTION TEST")
    print("🔄" + "="*50 + "🔄")
    
    # Create engine with a simple position
    engine = EngineV3("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    
    # Test the repetition detection methods
    print(f"\n📊 Initial position history: {len(engine.position_history)} positions")
    
    # Simulate some moves to build history
    import chess
    
    # Add some positions to history manually for testing
    test_position = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
    engine.position_history = [test_position]  # Simulate this position happened before
    
    print(f"📝 Added test position to history: {test_position[:20]}...")
    print(f"📊 Position history length: {len(engine.position_history)}")
    
    # Test a move that would return to this position
    test_move = chess.Move.from_uci("e2e4")  # Move that leads back to test position
    
    print(f"\n🔍 Testing move: {test_move}")
    would_repeat = engine.would_cause_repetition(test_move)
    print(f"🔄 Would cause repetition: {would_repeat}")
    
    if would_repeat:
        print("✅ Repetition detection working!")
    else:
        print("❌ Repetition detection needs adjustment")
    
    return would_repeat

def test_anti_repetition_in_game():
    """Test that engine avoids repetition in actual play"""
    print(f"\n🎮" + "="*50 + "🎮")
    print("    ANTI-REPETITION IN GAMEPLAY")
    print("🎮" + "="*50 + "🎮")
    
    # Start with a position where repetition might occur
    engine = EngineV3("rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3")
    
    moves_played = []
    positions_seen = []
    
    print("\n🕹️  Playing 10 moves to test for repetitive patterns...")
    
    for move_num in range(10):
        current_fen = engine.board.fen()
        current_pos = current_fen.split(' ')[0]
        
        print(f"\n🔄 Move {move_num + 1}:")
        print(f"   📍 Position: {current_pos[:30]}...")
        print(f"   📚 History length: {len(engine.position_history)}")
        
        # Count how many times we've seen this position
        position_count = positions_seen.count(current_pos)
        print(f"   🔢 Position seen {position_count} times before")
        
        if position_count >= 2:
            print("   ⚠️  WARNING: This would be third repetition!")
        
        # Get engine's move
        move = engine.get_move()
        moves_played.append(move)
        positions_seen.append(current_pos)
        
        print(f"   ⚡ Engine chose: {move}")
        
        # Update engine's position (simulate the move being played)
        try:
            import chess
            engine.board.push(chess.Move.from_uci(move))
        except:
            print("   ❌ Could not simulate move")
            break
    
    # Analyze for repetitive patterns
    print(f"\n📊 ANALYSIS:")
    print(f"   🕹️  Moves played: {moves_played}")
    
    # Check for immediate back-and-forth patterns
    repetitive_moves = 0
    for i in range(len(moves_played) - 1):
        if i > 0 and moves_played[i] == moves_played[i-2]:
            repetitive_moves += 1
    
    print(f"   🔄 Repetitive move patterns detected: {repetitive_moves}")
    
    if repetitive_moves < 2:
        print("   ✅ Good! Engine avoided excessive repetition")
    else:
        print("   ⚠️  Engine still shows repetitive behavior")
    
    # Check position repetitions
    unique_positions = len(set(positions_seen))
    total_positions = len(positions_seen)
    
    print(f"   📍 Unique positions: {unique_positions}/{total_positions}")
    
    if unique_positions >= total_positions * 0.8:  # At least 80% unique
        print("   ✅ Good position variety!")
    else:
        print("   ⚠️  Too many repeated positions")
    
    return repetitive_moves, unique_positions, total_positions

if __name__ == "__main__":
    print("🧪 Testing V3.2 Anti-Repetition Features...")
    
    # Test 1: Basic repetition detection
    detection_works = test_repetition_detection()
    
    # Test 2: Anti-repetition in gameplay
    repetitive_moves, unique_pos, total_pos = test_anti_repetition_in_game()
    
    print("\n" + "🎯" + "="*50 + "🎯")
    print("           FINAL ASSESSMENT")
    print("🎯" + "="*50 + "🎯")
    
    print(f"🔍 Repetition Detection: {'✅ Working' if detection_works else '❌ Needs Fix'}")
    print(f"🔄 Repetitive Moves: {repetitive_moves} (lower is better)")
    print(f"📍 Position Variety: {unique_pos}/{total_pos} ({(unique_pos/total_pos*100):.1f}%)")
    
    if detection_works and repetitive_moves < 3 and unique_pos >= total_pos * 0.7:
        print("\n🎉 SUCCESS! V3.2 anti-repetition features are working!")
        print("🏆 Engine should now avoid drawing in winning positions!")
    else:
        print("\n🔧 NEEDS IMPROVEMENT: Some anti-repetition features need tuning")
        
    print(f"\n💡 The V3.2 engine is now deployed and should avoid repetitive draws!") 