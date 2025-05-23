import chess
from chess_engine import Engine
import time

class EngineTest:
    def __init__(self):
        # Test positions with clearly good/bad moves
        self.test_positions = [
            {
                "fen": "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR b KQkq - 2 3",
                "description": "Black should avoid Nxe4 (hangs knight), prefer d6 or Be7",
                "good_moves": ["d6", "Be7", "Nc6"],
                "bad_moves": ["Nxe4"]
            },
            {
                "fen": "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4",
                "description": "Black should capture Bxf2+ (check and wins material)",
                "good_moves": ["Bxf2+"],
                "bad_moves": ["d6", "0-0", "h6"]
            },
            {
                "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
                "description": "Black should develop pieces, not hang material",
                "good_moves": ["Nf6", "Be7", "d6"],
                "bad_moves": ["f5", "Qh4"]
            },
            {
                "fen": "rnbqkbnr/ppp2ppp/3p4/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 3",
                "description": "Simple development position",
                "good_moves": ["Nf6", "Be7", "Nd7"],
                "bad_moves": ["Qg5", "Ke7"]
            },
            {
                "fen": "rnbqkb1r/pppppppp/5n2/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2",
                "description": "Opening position - Black should develop normally",
                "good_moves": ["e5", "d5", "c5"],
                "bad_moves": ["h6", "a6", "g6"]
            }
        ]
    
    def test_engine_move_quality(self, engine_class):
        """Test engine on predefined positions"""
        print(f"\n=== Testing Engine ===")
        total_score = 0
        total_tests = 0
        
        for i, test in enumerate(self.test_positions):
            print(f"\nTest {i+1}: {test['description']}")
            print(f"Position: {test['fen']}")
            
            # Create engine and get move
            engine = engine_class(test['fen'])
            start_time = time.time()
            move = engine.get_move()
            end_time = time.time()
            
            print(f"Engine chose: {move} (took {end_time - start_time:.2f}s)")
            
            # Score the move
            score = 0
            if move in test['good_moves']:
                score = 2
                print("✅ GOOD MOVE!")
            elif move in test['bad_moves']:
                score = -1
                print("❌ BAD MOVE!")
            else:
                score = 0
                print("⚪ NEUTRAL MOVE")
            
            total_score += score
            total_tests += 1
            
            print(f"Good moves: {test['good_moves']}")
            print(f"Bad moves: {test['bad_moves']}")
        
        avg_score = total_score / total_tests if total_tests > 0 else 0
        print(f"\n=== FINAL SCORE ===")
        print(f"Total Score: {total_score}/{total_tests * 2}")
        print(f"Average Score: {avg_score:.2f}")
        print(f"Percentage: {((avg_score + 1) / 3) * 100:.1f}%")
        
        return avg_score
    
    def compare_engines(self, engine1_class, engine2_class, name1="Engine 1", name2="Engine 2"):
        """Compare two engine versions"""
        print(f"\n{'='*50}")
        print(f"COMPARING {name1} vs {name2}")
        print(f"{'='*50}")
        
        print(f"\nTesting {name1}:")
        score1 = self.test_engine_move_quality(engine1_class)
        
        print(f"\nTesting {name2}:")
        score2 = self.test_engine_move_quality(engine2_class)
        
        print(f"\n{'='*50}")
        print(f"COMPARISON RESULTS:")
        print(f"{name1}: {score1:.2f}")
        print(f"{name2}: {score2:.2f}")
        
        if score2 > score1:
            print(f"🎉 {name2} is BETTER by {score2 - score1:.2f} points!")
        elif score1 > score2:
            print(f"📉 {name2} is WORSE by {score1 - score2:.2f} points")
        else:
            print(f"🔄 Both engines perform equally")
        print(f"{'='*50}")
        
        return score2 - score1

if __name__ == "__main__":
    # Test current engine
    tester = EngineTest()
    tester.test_engine_move_quality(Engine) 