#!/usr/bin/env python3
"""
🎯 TACTICAL STRENGTH TEST
Compare engines on challenging tactical positions to see which is truly stronger!
"""

import time
from collections import defaultdict

# Test positions with known best moves (or good moves)
TACTICAL_TESTS = [
    {
        "name": "Opening Development",
        "fen": "rnbqkbnr/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR b KQkq - 1 2",
        "description": "Black should develop knights, not move random pieces",
        "excellent_moves": ["g8f6", "b8c6"],  # Developing moves
        "good_moves": ["f8e7", "d7d6", "f7f5"],  # Acceptable
        "bad_moves": ["a7a6", "h7h6", "a8b8"],  # Wasted moves
        "points": {"excellent": 10, "good": 5, "bad": -5, "other": 0}
    },
    {
        "name": "Central Control",
        "fen": "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 4",
        "description": "Black should challenge the center or develop actively",
        "excellent_moves": ["f8b4", "d7d5"],  # Pin or central break
        "good_moves": ["f8e7", "b8c6", "d7d6"],  # Solid development
        "bad_moves": ["h7h6", "a7a6", "g7g6"],  # Passive
        "points": {"excellent": 10, "good": 5, "bad": -5, "other": 0}
    },
    {
        "name": "Material vs Position",
        "fen": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP1QPPP/RNB1K2R b KQkq - 4 6",
        "description": "Complex middlegame position requiring careful evaluation",
        "excellent_moves": ["f8e7", "d7d6", "c8e6"],  # Consolidating
        "good_moves": ["b8d7", "h7h6"],  # Reasonable
        "bad_moves": ["c5d4", "h8g8"],  # Losing material or awkward
        "points": {"excellent": 10, "good": 5, "bad": -10, "other": 0}
    },
    {
        "name": "King Safety",
        "fen": "r1bqk2r/pppp1p1p/2n2np1/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 7",
        "description": "King safety is paramount - should castle",
        "excellent_moves": ["e8g8"],  # Castling
        "good_moves": ["d7d6", "h7h6"],  # Preparing to castle
        "bad_moves": ["e8f7", "d8c7"],  # Exposing king
        "points": {"excellent": 15, "good": 3, "bad": -15, "other": -5}
    },
    {
        "name": "Piece Activity",
        "fen": "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP1QPPP/RNB2RK1 b - - 8 8",
        "description": "Both sides castled - now improve piece positions",
        "excellent_moves": ["c8e6", "d7d6", "a8d8"],  # Active development
        "good_moves": ["b8d7", "f8e8"],  # Solid
        "bad_moves": ["a7a6", "h7h6"],  # Slow
        "points": {"excellent": 10, "good": 5, "bad": -3, "other": 0}
    }
]

def score_move(move_str, test_case):
    """Score a move based on the test case criteria"""
    move_lower = move_str.lower()
    
    if move_lower in [m.lower() for m in test_case["excellent_moves"]]:
        return test_case["points"]["excellent"], "excellent"
    elif move_lower in [m.lower() for m in test_case["good_moves"]]:
        return test_case["points"]["good"], "good"
    elif move_lower in [m.lower() for m in test_case["bad_moves"]]:
        return test_case["points"]["bad"], "bad"
    else:
        return test_case["points"]["other"], "other"

def test_engine_strength():
    """Test the current production engine's tactical strength"""
    print("🎯" + "="*60 + "🎯")
    print("           TACTICAL STRENGTH ASSESSMENT")
    print("🎯" + "="*60 + "🎯")
    
    # Import the current production engine
    try:
        from chess_engine import EngineV3 as CurrentEngine, ENGINE_VERSION, ENGINE_NAME
        print(f"🤖 Testing: {ENGINE_NAME} v{ENGINE_VERSION}")
    except ImportError:
        print("❌ Could not import current engine!")
        return
    
    total_score = 0
    total_possible = 0
    results = []
    
    print(f"\n📊 Running {len(TACTICAL_TESTS)} tactical tests...")
    print("-" * 70)
    
    for i, test in enumerate(TACTICAL_TESTS, 1):
        print(f"\n🧩 Test {i}: {test['name']}")
        print(f"📝 {test['description']}")
        print(f"🏁 Position: {test['fen']}")
        
        # Create engine and get move
        engine = CurrentEngine(test['fen'])
        
        start_time = time.time()
        try:
            move = engine.get_move()
            calc_time = time.time() - start_time
        except Exception as e:
            print(f"❌ Engine error: {e}")
            move = "error"
            calc_time = 0
        
        # Score the move
        score, category = score_move(move, test)
        max_possible = test["points"]["excellent"]
        
        total_score += score
        total_possible += max_possible
        
        # Display result
        if category == "excellent":
            emoji = "🎯"
            color = "EXCELLENT"
        elif category == "good":
            emoji = "✅"
            color = "GOOD"
        elif category == "bad":
            emoji = "❌"
            color = "BAD"
        else:
            emoji = "🤔"
            color = "UNCLEAR"
        
        print(f"{emoji} Move: {move} ({calc_time:.2f}s)")
        print(f"📊 Score: {score}/{max_possible} ({color})")
        
        if category == "excellent":
            print("💡 Perfect choice! Engine found the best move.")
        elif category == "good":
            print("👍 Solid choice! Engine made a reasonable move.")
        elif category == "bad":
            print("⚠️  Questionable choice. Engine could improve here.")
        else:
            print("🤷 Unclear move. Not clearly good or bad.")
        
        results.append({
            "test": test['name'],
            "move": move,
            "score": score,
            "max_possible": max_possible,
            "category": category,
            "time": calc_time
        })
    
    # Final assessment
    print("\n" + "🏆" + "="*60 + "🏆")
    print("              FINAL TACTICAL ASSESSMENT")
    print("🏆" + "="*60 + "🏆")
    
    percentage = (total_score / total_possible * 100) if total_possible > 0 else 0
    avg_time = sum(r['time'] for r in results) / len(results)
    
    print(f"📊 Total Score: {total_score}/{total_possible} ({percentage:.1f}%)")
    print(f"⏱️  Average Time: {avg_time:.2f} seconds per move")
    
    # Grade the engine
    if percentage >= 90:
        grade = "🥇 GRANDMASTER"
        assessment = "Exceptional tactical strength!"
    elif percentage >= 75:
        grade = "🥈 MASTER"
        assessment = "Strong tactical play with minor weaknesses."
    elif percentage >= 60:
        grade = "🥉 EXPERT"
        assessment = "Good tactical understanding with room for improvement."
    elif percentage >= 40:
        grade = "📚 INTERMEDIATE"
        assessment = "Basic tactical awareness but misses key ideas."
    else:
        grade = "🔰 BEGINNER"
        assessment = "Needs significant improvement in tactical play."
    
    print(f"\n🎖️  Engine Grade: {grade}")
    print(f"💭 Assessment: {assessment}")
    
    # Detailed breakdown
    excellent_count = sum(1 for r in results if r['category'] == 'excellent')
    good_count = sum(1 for r in results if r['category'] == 'good')
    bad_count = sum(1 for r in results if r['category'] == 'bad')
    
    print(f"\n📈 Move Quality Breakdown:")
    print(f"   🎯 Excellent: {excellent_count}/{len(results)}")
    print(f"   ✅ Good: {good_count}/{len(results)}")
    print(f"   ❌ Bad: {bad_count}/{len(results)}")
    
    return {
        "total_score": total_score,
        "total_possible": total_possible,
        "percentage": percentage,
        "grade": grade,
        "avg_time": avg_time,
        "results": results
    }

if __name__ == "__main__":
    test_result = test_engine_strength()
    
    print(f"\n🎯 ENGINE STRENGTH: {test_result['percentage']:.1f}%")
    print(f"⚡ SPEED: {test_result['avg_time']:.2f}s average")
    print(f"🏆 GRADE: {test_result['grade']}")
    
    print("\n" + "="*60)
    print("🚀 Test complete! This gives you an objective measure of engine strength.")
    print("💡 Run this test after each improvement to track progress!") 