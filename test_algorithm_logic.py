import chess
import time

def simple_evaluate_pos(first, second):
    """
    Simple evaluation that mimics how a neural network might work
    Returns positive values for positions good for Black
    """
    board = chess.Board(second)
    
    # Material count
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 300,
        chess.BISHOP: 300,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0
    }
    
    material_score = 0
    for piece_type in piece_values:
        white_pieces = len(board.pieces(piece_type, chess.WHITE))
        black_pieces = len(board.pieces(piece_type, chess.BLACK))
        # Positive score is good for Black
        material_score += (black_pieces - white_pieces) * piece_values[piece_type]
    
    # Positional factors
    positional_score = 0
    
    # Control of center squares
    center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
    for square in center_squares:
        piece = board.piece_at(square)
        if piece:
            if piece.color == chess.BLACK:
                positional_score += 30
            else:
                positional_score -= 30
    
    # Development (pieces off back rank)
    for color in [chess.WHITE, chess.BLACK]:
        back_rank = 0 if color == chess.WHITE else 7
        developed = 0
        for square in [chess.B1, chess.C1, chess.F1, chess.G1] if color == chess.WHITE else [chess.B8, chess.C8, chess.F8, chess.G8]:
            piece = board.piece_at(square)
            if piece is None or piece.piece_type == chess.KING:
                developed += 1
        
        if color == chess.BLACK:
            positional_score += developed * 20
        else:
            positional_score -= developed * 20
    
    # King safety (not in center)
    king_square = board.king(chess.BLACK)
    if king_square in center_squares:
        positional_score -= 100
    
    white_king_square = board.king(chess.WHITE)
    if white_king_square in center_squares:
        positional_score += 100
    
    total_score = material_score + positional_score
    
    # Add some randomness to avoid identical evaluations
    import random
    total_score += random.randint(-5, 5)
    
    return total_score

class TestEngine:
    """Original algorithm with improved evaluation function"""
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
        print("calculating")
        for black_move in black_legal_moves:
            print(".", end="", flush=True)
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
                        prediction_number = simple_evaluate_pos(current_fen_x, next_fen_x)
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
        
        print()  # New line after dots
        
        if len(black_response) > 0:
            best_move = max(black_response, key=black_response.get)
            print(f"best move found: {best_move}")
            return str(best_move)
        else:
            print("CHECKMATE")
            return str("checkmate")

def test_specific_position():
    """Test on a specific position where we can verify the logic"""
    print("🧪 TESTING ALGORITHM LOGIC")
    print("=" * 50)
    
    # Simple position where Black should develop
    test_fen = "rnbqkbnr/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR b KQkq - 1 2"
    
    board = chess.Board(test_fen)
    print(f"Position: {test_fen}")
    print()
    print(board)
    print()
    
    print("Legal moves for Black:")
    legal_moves = list(board.legal_moves)
    for i, move in enumerate(legal_moves):
        print(f"{i+1:2d}. {move}")
    
    print(f"\nTotal legal moves: {len(legal_moves)}")
    print()
    
    engine = TestEngine(test_fen)
    start_time = time.time()
    chosen_move = engine.get_move()
    end_time = time.time()
    
    print(f"\n🤖 Engine chose: {chosen_move}")
    print(f"⏱️  Time taken: {end_time - start_time:.2f}s")
    
    # Analyze the choice
    good_developing_moves = ["Nf6", "Nc6", "d6", "Be7", "Nd7"]
    bad_moves = ["h6", "a6", "g6", "h5", "a5"]
    
    if chosen_move in good_developing_moves:
        print("✅ GOOD! This is a developing move.")
    elif chosen_move in bad_moves:
        print("❌ BAD! This is a wasteful move.")
    else:
        print("⚪ NEUTRAL. Not obviously good or bad.")
    
    print(f"\nGood developing moves: {good_developing_moves}")
    print(f"Bad moves: {bad_moves}")

def test_tactical_position():
    """Test on a position with a clear tactical opportunity"""
    print("\n" + "=" * 50)
    print("🎯 TESTING TACTICAL AWARENESS")
    print("=" * 50)
    
    # Position where Black can win material
    test_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 1 4"
    
    board = chess.Board(test_fen)
    print(f"Position: {test_fen}")
    print("(White threatens Qxf7# mate!)")
    print()
    print(board)
    print()
    
    engine = TestEngine(test_fen)
    start_time = time.time()
    chosen_move = engine.get_move()
    end_time = time.time()
    
    print(f"\n🤖 Engine chose: {chosen_move}")
    print(f"⏱️  Time taken: {end_time - start_time:.2f}s")
    
    # Check if it found the right defensive move
    good_defensive_moves = ["Qf6", "g6"]  # Blocks mate threat
    if chosen_move in good_defensive_moves:
        print("✅ EXCELLENT! Engine found the defensive move.")
    else:
        print("❌ MISSED! Engine didn't defend against mate.")
        print(f"Good defensive moves were: {good_defensive_moves}")

if __name__ == "__main__":
    test_specific_position()
    test_tactical_position() 