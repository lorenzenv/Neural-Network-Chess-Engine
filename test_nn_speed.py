#!/usr/bin/env python3
"""
🧠 NEURAL NETWORK SPEED BENCHMARK
Test the real speed of neural network evaluation and caching
"""

import time
import sys
import os

# Add current directory to path to import our engine
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from chess_engine import EngineV3, ENGINE_VERSION, ENGINE_NAME
    print(f"✅ Loaded: {ENGINE_NAME} v{ENGINE_VERSION}")
except ImportError as e:
    print(f"❌ Could not import engine: {e}")
    sys.exit(1)

def benchmark_neural_network():
    """Benchmark neural network performance"""
    print("🧠" + "="*60 + "🧠")
    print("         NEURAL NETWORK SPEED BENCHMARK")
    print("🧠" + "="*60 + "🧠")
    
    # Test positions
    test_positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
        "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3",  # Italian Game
        "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",  # Spanish Opening
        "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 5",  # Middle game
        "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/3P1N2/PPP1QPPP/RNB2RK1 w - - 0 8"  # Castled position
    ]
    
    total_time = 0
    total_moves = 0
    cache_performance = []
    
    print(f"\n🎯 Testing {len(test_positions)} positions...")
    
    for i, fen in enumerate(test_positions, 1):
        print(f"\n📍 Position {i}: {fen[:30]}...")
        
        # Create fresh engine for each test
        engine = EngineV3(fen)
        
        # Test move calculation with timing
        start_time = time.time()
        move = engine.get_move()
        calc_time = time.time() - start_time
        
        total_time += calc_time
        total_moves += 1
        
        # Get performance stats
        nodes = getattr(engine, 'nodes_searched', 0)
        cache_hits = getattr(engine, 'cache_hits', 0)
        cache_size = len(getattr(engine, 'position_cache', {}))
        
        cache_performance.append({
            'position': i,
            'time': calc_time,
            'nodes': nodes,
            'cache_hits': cache_hits,
            'cache_size': cache_size
        })
        
        print(f"⚡ Move: {move}")
        print(f"⏱️  Time: {calc_time:.3f}s")
        print(f"🔍 Nodes: {nodes:,}")
        print(f"💾 Cache hits: {cache_hits}")
        print(f"📦 Cache size: {cache_size}")
        
        if calc_time > 0:
            nps = nodes / calc_time if calc_time > 0 else 0
            print(f"🚀 Speed: {nps:,.0f} nodes/second")
    
    # Overall performance analysis
    print("\n" + "📊" + "="*60 + "📊")
    print("              PERFORMANCE ANALYSIS")
    print("📊" + "="*60 + "📊")
    
    avg_time = total_time / total_moves if total_moves > 0 else 0
    total_nodes = sum(p['nodes'] for p in cache_performance)
    total_cache_hits = sum(p['cache_hits'] for p in cache_performance)
    
    print(f"⏱️  Average time per move: {avg_time:.3f}s")
    print(f"🔍 Total nodes searched: {total_nodes:,}")
    print(f"💾 Total cache hits: {total_cache_hits}")
    
    if total_nodes > 0:
        cache_hit_rate = (total_cache_hits / total_nodes) * 100
        print(f"📈 Cache hit rate: {cache_hit_rate:.1f}%")
    
    if total_time > 0:
        overall_nps = total_nodes / total_time
        print(f"🚀 Overall speed: {overall_nps:,.0f} nodes/second")
    
    # Speed assessment
    print(f"\n🎯 SPEED ASSESSMENT:")
    if avg_time < 0.1:
        grade = "🥇 EXCELLENT"
        assessment = "Lightning fast! Great user experience."
    elif avg_time < 0.3:
        grade = "🥈 VERY GOOD"
        assessment = "Fast response time, users will be happy."
    elif avg_time < 0.5:
        grade = "🥉 GOOD"
        assessment = "Acceptable speed for online play."
    elif avg_time < 1.0:
        grade = "⚡ MODERATE"
        assessment = "Noticeable delay but still playable."
    else:
        grade = "🐌 SLOW"
        assessment = "Users may get impatient waiting for moves."
    
    print(f"   {grade}")
    print(f"   💭 {assessment}")
    
    # Optimization suggestions
    if cache_hit_rate < 30:
        print(f"\n💡 OPTIMIZATION OPPORTUNITY:")
        print(f"   🔧 Low cache hit rate ({cache_hit_rate:.1f}%) - consider better caching")
    
    if avg_time > 0.5:
        print(f"\n⚡ SPEED IMPROVEMENT IDEAS:")
        print(f"   🔍 Reduce search depth for faster response")
        print(f"   💾 Improve position caching")
        print(f"   🧠 Optimize neural network calls")
    
    return {
        'avg_time': avg_time,
        'total_nodes': total_nodes,
        'cache_hit_rate': cache_hit_rate if total_nodes > 0 else 0,
        'grade': grade
    }

def quick_stress_test():
    """Quick stress test with multiple rapid moves"""
    print(f"\n🏃‍♂️ STRESS TEST: 10 rapid moves...")
    
    engine = EngineV3("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    times = []
    
    for i in range(10):
        start_time = time.time()
        move = engine.get_move()
        calc_time = time.time() - start_time
        times.append(calc_time)
        print(f"   Move {i+1}: {move} ({calc_time:.3f}s)")
    
    avg_stress_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n📊 Stress test results:")
    print(f"   ⏱️  Average: {avg_stress_time:.3f}s")
    print(f"   🚀 Fastest: {min_time:.3f}s")
    print(f"   🐌 Slowest: {max_time:.3f}s")
    
    return avg_stress_time

if __name__ == "__main__":
    # Run comprehensive benchmark
    results = benchmark_neural_network()
    
    # Run stress test
    stress_avg = quick_stress_test()
    
    print("\n" + "🎉" + "="*60 + "🎉")
    print(f"🏆 FINAL VERDICT: {results['grade']}")
    print(f"⚡ Average speed: {results['avg_time']:.3f}s per move")
    print(f"💾 Cache efficiency: {results['cache_hit_rate']:.1f}%")
    print(f"🏃‍♂️ Stress test average: {stress_avg:.3f}s")
    print("🎉" + "="*60 + "🎉")
    
    if results['avg_time'] < 0.2:
        print("🎊 AMAZING! Your engine is optimized for great online play!")
    elif results['avg_time'] < 0.5:
        print("👍 GOOD! Solid performance for online chess.")
    else:
        print("🔧 Consider further optimizations for better user experience.") 