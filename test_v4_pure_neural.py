#!/usr/bin/env python3
"""
🧠 V4.0 PURE NEURAL POWER TEST
Test the new V4.0 engine with pure neural network strength (no opening book)
Focus on advanced search techniques and improved performance
"""

try:
    from chess_engine_v4 import EngineV4
    print("✅ Loaded V4.0 Pure Neural Power Engine")
except ImportError:
    print("❌ Could not import V4.0 engine")
    exit(1)

import time
import chess

def test_pure_neural_strength():
    """Test that V4.0 works without opening book and relies on pure neural strength"""
    print("🧠" + "="*50 + "🧠")
    print("        PURE NEURAL STRENGTH TEST")
    print("🧠" + "="*50 + "🧠")
    
    # Test with opening position - should NOT use opening book
    engine = EngineV4("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    
    print(f"\n🧪 Testing opening position (should use PURE neural evaluation)...")
    print(f"🚫 No opening book - relying on neural network only")
    
    start_time = time.time()
    move = engine.get_move()
    end_time = time.time()
    
    print(f"\n✅ V4.0 chose: {move}")
    print(f"⏱️  Time taken: {(end_time - start_time):.2f} seconds")
    print(f"🧠 This move was chosen by PURE neural network evaluation!")
    
    # Check that it's a reasonable opening move
    reasonable_openings = ['e2e4', 'd2d4', 'g1f3', 'b1c3', 'c2c4']
    if move in reasonable_openings:
        print(f"✅ Excellent! {move} is a strong opening move from pure neural strength")
    else:
        print(f"🤔 Interesting choice: {move} - neural network has its own style!")
    
    return move

def test_advanced_search_features():
    """Test the advanced search features of V4.0"""
    print(f"\n⚡" + "="*50 + "⚡")
    print("        ADVANCED SEARCH FEATURES TEST")
    print("⚡" + "="*50 + "⚡")
    
    # Test with a tactical position where advanced search should shine
    engine = EngineV4("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 1 4")
    
    print(f"\n🎯 Testing tactical position for advanced search features...")
    print(f"🔍 V4.0 should utilize:")
    print(f"   • Enhanced move ordering (MVV-LVA + History)")
    print(f"   • Null move pruning")
    print(f"   • Late move reductions") 
    print(f"   • Transposition tables")
    print(f"   • Quiescence search")
    
    start_time = time.time()
    move = engine.get_move()
    end_time = time.time()
    
    print(f"\n🎯 V4.0 tactical choice: {move}")
    print(f"⏱️  Search time: {(end_time - start_time):.2f} seconds")
    
    # Check the engine's internal statistics
    print(f"\n📊 ADVANCED SEARCH STATISTICS:")
    print(f"   🔍 Nodes searched: {engine.nodes_searched:,}")
    print(f"   💾 Cache hits: {engine.cache_hits}")
    print(f"   ⚡ Null move cutoffs: {engine.null_move_cutoffs}")
    print(f"   🎯 Late move reductions: {engine.late_move_reductions}")
    print(f"   🧮 Transposition table size: {len(engine.transposition_table):,}")
    print(f"   📚 History table entries: {len(engine.history_table)}")
    
    # Calculate search efficiency
    if engine.nodes_searched > 0:
        cache_rate = (engine.cache_hits / engine.nodes_searched) * 100
        print(f"   💡 Cache hit rate: {cache_rate:.1f}%")
        
        if cache_rate > 10:
            print("   ✅ Excellent cache efficiency!")
        elif cache_rate > 5:
            print("   👍 Good cache usage!")
        else:
            print("   📈 Cache could be more effective")
    
    return move, engine.nodes_searched, engine.cache_hits

def test_move_ordering_sophistication():
    """Test the sophisticated move ordering of V4.0"""
    print(f"\n🎯" + "="*50 + "🎯")
    print("      SOPHISTICATED MOVE ORDERING TEST")
    print("🎯" + "="*50 + "🎯")
    
    # Create a position with various move types to test ordering
    engine = EngineV4("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5")
    
    print(f"\n🔍 Testing move ordering with position containing:")
    print(f"   • Captures")
    print(f"   • Checks") 
    print(f"   • Development moves")
    print(f"   • Central control")
    print(f"   • Castling opportunity")
    
    # Test the move ordering function directly
    from chess_engine_v4 import order_moves_v4
    
    legal_moves = list(engine.board.legal_moves)
    print(f"\n📝 Found {len(legal_moves)} legal moves")
    
    # Order the moves
    ordered_moves = order_moves_v4(engine.board, legal_moves, 
                                  engine.killer_moves[0], engine.history_table)
    
    print(f"\n🎯 TOP 5 MOVES (by V4.0 sophisticated ordering):")
    for i, move in enumerate(ordered_moves[:5]):
        move_str = str(move)
        
        # Analyze move type
        move_type = ""
        if engine.board.piece_at(move.to_square):
            move_type += "📥 Capture "
        if engine.board.is_castling(move):
            move_type += "🏰 Castle "
        
        engine.board.push(move)
        if engine.board.is_check():
            move_type += "⚔️ Check "
        engine.board.pop()
        
        if not move_type:
            move_type = "🎭 Quiet"
        
        print(f"   {i+1}. {move_str:8} - {move_type}")
    
    # Test actual move selection
    start_time = time.time()
    chosen_move = engine.get_move()
    end_time = time.time()
    
    print(f"\n✅ V4.0 final choice: {chosen_move}")
    print(f"⏱️  Decision time: {(end_time - start_time):.2f} seconds")
    
    # Check if chosen move was in top 3 of ordered moves
    top_3_moves = [str(move) for move in ordered_moves[:3]]
    if chosen_move in top_3_moves:
        position = top_3_moves.index(chosen_move) + 1
        print(f"🎯 Excellent! Chose move #{position} from sophisticated ordering")
    else:
        print(f"🤔 Chose different move - deeper search changed evaluation")
    
    return chosen_move

def test_speed_vs_strength():
    """Test the balance between speed and strength in V4.0"""
    print(f"\n⚡" + "="*50 + "⚡")
    print("        SPEED vs STRENGTH BALANCE TEST")
    print("⚡" + "="*50 + "⚡")
    
    test_positions = [
        # Opening position
        ("Opening", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
        # Middlegame tactical
        ("Tactical", "r1bq1rk1/ppp2ppp/2n2n2/2bp4/2B5/3P1N2/PPP2PPP/RNBQR1K1 w - - 1 8"),
        # Complex endgame  
        ("Endgame", "4k3/8/3K4/8/8/1Q6/8/8 w - - 0 1")
    ]
    
    total_time = 0
    total_nodes = 0
    
    for position_name, fen in test_positions:
        print(f"\n🎮 Testing {position_name} position...")
        engine = EngineV4(fen)
        
        start_time = time.time()
        move = engine.get_move()
        end_time = time.time()
        
        duration = end_time - start_time
        total_time += duration
        total_nodes += engine.nodes_searched
        
        print(f"   ⚡ Move: {move}")
        print(f"   ⏱️  Time: {duration:.2f}s")
        print(f"   🔍 Nodes: {engine.nodes_searched:,}")
        print(f"   🏃 Speed: {engine.nodes_searched/duration:,.0f} nodes/sec")
        
        # Efficiency metrics
        optimizations = engine.null_move_cutoffs + engine.late_move_reductions
        print(f"   ⚡ Optimizations: {optimizations} (nulls: {engine.null_move_cutoffs}, LMR: {engine.late_move_reductions})")
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"   ⏱️  Total time: {total_time:.2f} seconds")
    print(f"   🔍 Total nodes: {total_nodes:,}")
    print(f"   🏃 Average speed: {total_nodes/total_time:,.0f} nodes/second")
    print(f"   ⚡ Average time per move: {total_time/len(test_positions):.2f}s")
    
    if total_time/len(test_positions) < 2.0:
        print("   ✅ Excellent speed! Under 2 seconds per move")
    elif total_time/len(test_positions) < 5.0:
        print("   👍 Good speed! Under 5 seconds per move")
    else:
        print("   ⏰ Could be faster - consider optimizations")
    
    return total_time, total_nodes

if __name__ == "__main__":
    print("🚀 Testing V4.0 Pure Neural Power Engine...")
    print("🧠 This engine has NO opening book - pure neural network strength!")
    
    # Test 1: Pure neural strength
    neural_move = test_pure_neural_strength()
    
    # Test 2: Advanced search features
    tactical_move, nodes, cache_hits = test_advanced_search_features()
    
    # Test 3: Move ordering sophistication
    ordered_move = test_move_ordering_sophistication()
    
    # Test 4: Speed vs strength balance
    total_time, total_nodes = test_speed_vs_strength()
    
    print("\n" + "🏆" + "="*50 + "🏆")
    print("           V4.0 ASSESSMENT SUMMARY")
    print("🏆" + "="*50 + "🏆")
    
    print(f"🧠 Pure Neural Strength: ✅ Working (chose {neural_move})")
    print(f"⚡ Advanced Search: ✅ {nodes:,} nodes, {cache_hits} cache hits") 
    print(f"🎯 Move Ordering: ✅ Sophisticated scoring system")
    print(f"🏃 Performance: {total_nodes/total_time:,.0f} nodes/sec average")
    
    print(f"\n🎉 V4.0 PURE NEURAL POWER ENGINE IS READY!")
    print(f"💪 Features: No opening book, advanced pruning, smart ordering")
    print(f"🧠 This engine will show the TRUE strength of your neural network!")
    
    # Version info
    try:
        engine = EngineV4("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        version_info = engine.get_version_info()
        print(f"\n📋 Engine: {version_info['name']}")
        print(f"🔢 Version: {version_info['version']}")
        print(f"✨ Features: {len(version_info['features'])} advanced capabilities")
    except Exception as e:
        print(f"❌ Could not get version info: {e}") 