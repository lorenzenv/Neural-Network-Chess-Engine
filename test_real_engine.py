import chess
import time
import sys

# Try to import the real engine
try:
    from chess_engine import Engine
    REAL_ENGINE_AVAILABLE = True
    print("✅ Real engine with neural network available!")
except Exception as e:
    print(f"❌ Could not load real engine: {e}")
    REAL_ENGINE_AVAILABLE = False

class RealEngineTest:
    def __init__(self):
        # Test positions with clearly good/bad moves for Black
        self.test_positions = [
            {
                "fen": "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR b KQkq - 2 3",
                "description": "Black should avoid Nxe4 (hangs knight)",
                "good_moves": ["d6", "Be7", "Nc6", "Bd6"],
                "bad_moves": ["Nxe4"],
                "best_move": "d6"
            },
            {
                "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 4",
                "description": "Black should develop naturally",
                "good_moves": ["Be7", "Bb4+", "d6"],
                "bad_moves": ["Nxe4", "h6", "a6"],
                "best_move": "Be7"
            },
            {
                "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
                "description": "Simple development",
                "good_moves": ["Nf6", "Be7", "f5"],
                "bad_moves": ["Qh4", "Ke7"],
                "best_move": "Nf6"
            },
            {
                "fen": "rnbqkbnr/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR b KQkq - 1 2",
                "description": "Opening position - should develop",
                "good_moves": ["Nf6", "Nc6", "d6", "Be7"],
                "bad_moves": ["h6", "a6", "g6"],
                "best_move": "Nf6"
            },
            {
                "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 1 4",
                "description": "White threatens mate - Black must defend",
                "good_moves": ["Qf6", "g6", "Nf6"],
                "bad_moves": ["d6", "Be7", "h6"],
                "best_move": "Qf6"
            }
        ]
    
    def test_real_engine(self):
        """Test the real engine with neural network"""
        if not REAL_ENGINE_AVAILABLE:
            print("Cannot test real engine - not available")
            return
        
        print(f"\n{'='*60}")
        print("TESTING RESTORED ORIGINAL ENGINE WITH NEURAL NETWORK")
        print(f"{'='*60}")
        
        total_score = 0
        total_time = 0
        
        for i, test in enumerate(self.test_positions):
            print(f"\n{'='*40}")
            print(f"Test {i+1}: {test['description']}")
            print(f"Position: {test['fen']}")
            
            # Show the position
            board = chess.Board(test['fen'])
            print()
            print(board)
            print()
            
            try:
                # Create engine and get move
                engine = Engine(test['fen'])
                print("Engine thinking...")
                start_time = time.time()
                move = engine.get_move()
                end_time = time.time()
                
                move_time = end_time - start_time
                total_time += move_time
                
                print(f"\n🤖 Engine chose: {move}")
                print(f"⏱️  Time taken: {move_time:.2f}s")
                
                # Score the move
                score = 0
                if move == test['best_move']:
                    score = 3
                    print("🎯 BEST MOVE!")
                elif move in test['good_moves']:
                    score = 2
                    print("✅ GOOD MOVE!")
                elif move in test['bad_moves']:
                    score = -2
                    print("❌ BAD MOVE!")
                else:
                    score = 0
                    print("⚪ NEUTRAL MOVE")
                
                total_score += score
                print(f"📊 Score: {score}/3")
                print(f"💡 Best move: {test['best_move']}")
                print(f"✅ Good moves: {test['good_moves']}")
                print(f"❌ Bad moves: {test['bad_moves']}")
                
            except Exception as e:
                print(f"❌ Error testing position: {e}")
                score = 0
                total_score += score
        
        max_score = len(self.test_positions) * 3
        avg_time = total_time / len(self.test_positions) if len(self.test_positions) > 0 else 0
        
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        print(f"📊 Total Score: {total_score}/{max_score}")
        print(f"📈 Percentage: {(total_score/max_score)*100:.1f}%")
        print(f"⏱️  Average Time: {avg_time:.2f}s")
        
        if total_score >= max_score * 0.7:
            print("🎉 EXCELLENT! Engine is playing well!")
        elif total_score >= max_score * 0.4:
            print("👍 GOOD! Engine shows decent play")
        elif total_score >= 0:
            print("😐 OK! Engine needs improvement")
        else:
            print("😞 POOR! Engine is making bad moves")
        
        return total_score, avg_time

def quick_test():
    """Quick single position test"""
    if not REAL_ENGINE_AVAILABLE:
        print("Cannot run quick test - real engine not available")
        return
    
    print("🚀 QUICK TEST - Simple Position")
    test_fen = "rnbqkbnr/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR b KQkq - 1 2"
    
    board = chess.Board(test_fen)
    print(f"Position: {test_fen}")
    print()
    print(board)
    print()
    
    try:
        engine = Engine(test_fen)
        print("Engine thinking...")
        start_time = time.time()
        move = engine.get_move()
        end_time = time.time()
        
        print(f"🤖 Engine chose: {move}")
        print(f"⏱️  Time: {end_time - start_time:.2f}s")
        
        good_moves = ["Nf6", "Nc6", "d6", "Be7"]
        if move in good_moves:
            print("✅ Looks like a reasonable move!")
        else:
            print("❓ Unusual choice...")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_test()
    else:
        tester = RealEngineTest()
        tester.test_real_engine() 