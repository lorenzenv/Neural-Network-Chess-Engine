import chess
import time

# Simple mock evaluation function for testing without neural network
def mock_evaluate_pos(first, second):
    """Mock evaluation that returns a simple score based on material"""
    board = chess.Board(second)
    
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
    score = 0
    for piece_type in piece_values:
        white_pieces = len(board.pieces(piece_type, chess.WHITE))
        black_pieces = len(board.pieces(piece_type, chess.BLACK))
        score += (white_pieces - black_pieces) * piece_values[piece_type]
    
    return score

class TestableEngine:
    """Original working algorithm with mock evaluation for testing"""
    def __init__(self, fen):
        self.board = chess.Board()
        self.board.set_fen(fen)

    def get_move(self):
        best_move = self.make_move()
        return str(best_move)

    def make_move(self):
        all_pos = {}
        black_legal_moves = self.board.legal_moves
        current_fen_x = self.board.fen()
        black_response = {}
        
        for black_move in black_legal_moves:
            white_response = {}
            self.board.push(black_move)
            if self.board.is_checkmate():
                self.board.pop()
                return str(black_move)
            
            white_legal_moves = self.board.legal_moves
            for white_move in white_legal_moves:
                self.board.push(white_move)
                if self.board.is_checkmate():
                    white_response[white_move] = 0
                    self.board.pop()
                    break
                
                black_legal_moves_depth_2 = self.board.legal_moves
                black_response_depth_2 = {}
                
                for black_move_depth_2 in black_legal_moves_depth_2:
                    self.board.push(black_move_depth_2)
                    if self.board.is_checkmate():
                        black_response_depth_2[black_move_depth_2] = 1
                        self.board.pop()
                        break
                    
                    next_fen_x = self.board.fen()
                    if next_fen_x in all_pos:
                        prediction_number = all_pos[next_fen_x]
                    else:
                        prediction_number = mock_evaluate_pos(current_fen_x, next_fen_x)
                        all_pos[next_fen_x] = prediction_number
                    
                    if len(white_response) > 0:
                        if prediction_number > white_response[max(white_response, key=white_response.get)]:
                            black_response_depth_2[black_move_depth_2] = prediction_number
                            self.board.pop()
                            break
                        else:
                            black_response_depth_2[black_move_depth_2] = prediction_number
                            self.board.pop()
                    else:
                        black_response_depth_2[black_move_depth_2] = prediction_number
                        self.board.pop()
                
                if len(black_response) > 0 and len(white_response) > 0:
                    if white_response[min(white_response, key=white_response.get)] < black_response[max(black_response, key=black_response.get)]:
                        white_response[white_move] = black_response_depth_2[max(black_response_depth_2, key=black_response_depth_2.get)]
                        self.board.pop()
                        break
                    else:
                        if len(black_response_depth_2) > 0:
                            white_response[white_move] = black_response_depth_2[max(black_response_depth_2, key=black_response_depth_2.get)]
                        self.board.pop()
                else:
                    if len(black_response_depth_2) > 0:
                        white_response[white_move] = black_response_depth_2[max(black_response_depth_2, key=black_response_depth_2.get)]
                    self.board.pop()
            
            if len(white_response) > 0:
                black_response[black_move] = white_response[min(white_response, key=white_response.get)]
            self.board.pop()
        
        if len(black_response) > 0:
            best_move = max(black_response, key=black_response.get)
        else:
            return "checkmate"

        return str(best_move)

class EngineComparison:
    def __init__(self):
        # Test positions with clearly good/bad moves for Black
        self.test_positions = [
            {
                "fen": "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR b KQkq - 2 3",
                "description": "Black should avoid Nxe4 (hangs knight)",
                "good_moves": ["d6", "Be7", "Nc6", "Bd6"],
                "bad_moves": ["Nxe4"],
                "best_move": "d6"  # Most natural development
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
            }
        ]
    
    def test_engine_quality(self, engine_class, name="Engine"):
        """Test engine on tactical positions"""
        print(f"\n=== Testing {name} ===")
        total_score = 0
        total_time = 0
        
        for i, test in enumerate(self.test_positions):
            print(f"\nTest {i+1}: {test['description']}")
            print(f"Position: {test['fen']}")
            
            # Create engine and get move
            engine = engine_class(test['fen'])
            start_time = time.time()
            move = engine.get_move()
            end_time = time.time()
            
            move_time = end_time - start_time
            total_time += move_time
            
            print(f"{name} chose: {move} (took {move_time:.2f}s)")
            
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
            print(f"Score: {score}/3")
            print(f"Best: {test['best_move']}, Good: {test['good_moves']}, Bad: {test['bad_moves']}")
        
        max_score = len(self.test_positions) * 3
        avg_time = total_time / len(self.test_positions)
        
        print(f"\n=== {name} SUMMARY ===")
        print(f"Total Score: {total_score}/{max_score}")
        print(f"Percentage: {(total_score/max_score)*100:.1f}%")
        print(f"Average Time: {avg_time:.2f}s")
        
        return total_score, avg_time
    
    def compare_engines(self, engine1_class, engine2_class, name1="Engine 1", name2="Engine 2"):
        """Compare two engines"""
        print(f"\n{'='*60}")
        print(f"COMPARING {name1} vs {name2}")
        print(f"{'='*60}")
        
        score1, time1 = self.test_engine_quality(engine1_class, name1)
        score2, time2 = self.test_engine_quality(engine2_class, name2)
        
        print(f"\n{'='*60}")
        print(f"FINAL COMPARISON:")
        print(f"{name1}: Score {score1}, Time {time1:.2f}s")
        print(f"{name2}: Score {score2}, Time {time2:.2f}s")
        
        if score2 > score1:
            print(f"🎉 {name2} plays BETTER! (+{score2-score1} points)")
        elif score1 > score2:
            print(f"📉 {name2} plays WORSE! ({score2-score1} points)")
        else:
            print(f"🔄 Both engines play equally well")
            
        if time2 < time1:
            print(f"⚡ {name2} is FASTER! ({time1-time2:.2f}s faster)")
        elif time1 < time2:
            print(f"🐌 {name2} is SLOWER! ({time2-time1:.2f}s slower)")
        else:
            print(f"⏱️ Both engines have similar speed")
        
        print(f"{'='*60}")
        
        return score2 - score1, time1 - time2

if __name__ == "__main__":
    # Test the original algorithm
    comparison = EngineComparison()
    comparison.test_engine_quality(TestableEngine, "Original Algorithm") 