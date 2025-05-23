import chess
import time

# Mock evaluation function for testing
def mock_evaluate_pos(first, second):
    """Mock evaluation that returns a simple score based on material and position"""
    board = chess.Board(second)
    
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
        material_score += (black_pieces - white_pieces) * piece_values[piece_type]
    
    # Add positional factors
    positional_score = 0
    
    # Control of center
    center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
    for square in center_squares:
        piece = board.piece_at(square)
        if piece:
            if piece.color == chess.BLACK:
                positional_score += 30
            else:
                positional_score -= 30
    
    # Development bonus
    for color in [chess.WHITE, chess.BLACK]:
        developed = 0
        back_rank_squares = [chess.B1, chess.C1, chess.F1, chess.G1] if color == chess.WHITE else [chess.B8, chess.C8, chess.F8, chess.G8]
        for square in back_rank_squares:
            piece = board.piece_at(square)
            if piece is None or piece.piece_type == chess.KING:
                developed += 1
        
        if color == chess.BLACK:
            positional_score += developed * 20
        else:
            positional_score -= developed * 20
    
    return material_score + positional_score

def order_moves(board, moves):
    """Order moves to search better moves first"""
    move_scores = []
    
    for move in moves:
        score = 0
        
        # Prioritize captures
        if board.piece_at(move.to_square) is not None:
            captured_piece = board.piece_at(move.to_square)
            moving_piece = board.piece_at(move.from_square)
            
            piece_values = {
                chess.PAWN: 1,
                chess.KNIGHT: 3,
                chess.BISHOP: 3,
                chess.ROOK: 5,
                chess.QUEEN: 9,
                chess.KING: 0
            }
            
            if captured_piece and moving_piece:
                score += piece_values.get(captured_piece.piece_type, 0) * 10
                score -= piece_values.get(moving_piece.piece_type, 0)
        
        # Prioritize checks
        board.push(move)
        if board.is_check():
            score += 50
        board.pop()
        
        # Prioritize central moves
        if move.to_square in [chess.E4, chess.E5, chess.D4, chess.D5]:
            score += 20
        
        # Prioritize development
        if board.piece_at(move.from_square):
            piece = board.piece_at(move.from_square)
            if piece.color == chess.BLACK:
                if chess.square_rank(move.from_square) == 7:
                    score += 15
        
        move_scores.append((score, move))
    
    move_scores.sort(key=lambda x: x[0], reverse=True)
    return [move for score, move in move_scores]

class EngineOriginal:
    """Original working engine"""
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
            return str(best_move)
        else:
            return "checkmate"

class EngineImproved:
    """Improved engine with move ordering and caching"""
    def __init__(self, fen):
        self.board = chess.Board()
        self.board.set_fen(fen)
        self.position_cache = {}

    def get_move(self):
        best_move = self.make_move()
        return str(best_move)

    def make_move(self):
        current_fen_x = self.board.fen()
        black_response = {}
        
        # Order moves for better search
        black_legal_moves = order_moves(self.board, list(self.board.legal_moves))
        
        for black_move in black_legal_moves:
            white_response = {}
            self.board.push(black_move)
            if self.board.is_checkmate():
                self.board.pop()
                return str(black_move)
            
            # Order white moves too
            white_legal_moves = order_moves(self.board, list(self.board.legal_moves))
            
            for white_move in white_legal_moves:
                self.board.push(white_move)
                if self.board.is_checkmate():
                    white_response[white_move] = 0
                    self.board.pop()
                    break
                
                black_legal_moves_depth_2 = order_moves(self.board, list(self.board.legal_moves))
                black_response_depth_2 = {}
                
                for black_move_depth_2 in black_legal_moves_depth_2:
                    self.board.push(black_move_depth_2)
                    if self.board.is_checkmate():
                        black_response_depth_2[black_move_depth_2] = 1
                        self.board.pop()
                        break
                    
                    next_fen_x = self.board.fen()
                    if next_fen_x in self.position_cache:
                        prediction_number = self.position_cache[next_fen_x]
                    else:
                        prediction_number = mock_evaluate_pos(current_fen_x, next_fen_x)
                        self.position_cache[next_fen_x] = prediction_number
                    
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
            return str(best_move)
        else:
            return "checkmate"

def compare_engines():
    test_positions = [
        {
            "fen": "rnbqkbnr/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR b KQkq - 1 2",
            "description": "Opening position - should develop",
            "good_moves": ["Nf6", "Nc6", "d6", "Be7"],
            "best_move": "Nf6"
        },
        {
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 4",
            "description": "Black should develop naturally",
            "good_moves": ["Be7", "Bb4+", "d6"],
            "best_move": "Be7"
        },
        {
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 1 4",
            "description": "White threatens mate - defend!",
            "good_moves": ["Qf6", "g6"],
            "best_move": "Qf6"
        }
    ]
    
    print("🏆 ORIGINAL vs IMPROVED ENGINE COMPARISON")
    print("=" * 60)
    
    original_score = 0
    improved_score = 0
    original_time = 0
    improved_time = 0
    
    for i, test in enumerate(test_positions):
        print(f"\n📍 Test {i+1}: {test['description']}")
        print(f"Position: {test['fen']}")
        
        board = chess.Board(test['fen'])
        print()
        print(board)
        print()
        
        # Test original engine
        print("🔹 Testing Original Engine...")
        engine_orig = EngineOriginal(test['fen'])
        start_time = time.time()
        move_orig = engine_orig.get_move()
        orig_time = time.time() - start_time
        original_time += orig_time
        
        # Test improved engine
        print("🔸 Testing Improved Engine...")
        engine_impr = EngineImproved(test['fen'])
        start_time = time.time()
        move_impr = engine_impr.get_move()
        impr_time = time.time() - start_time
        improved_time += impr_time
        
        print(f"\n📊 RESULTS:")
        print(f"Original: {move_orig} ({orig_time:.2f}s)")
        print(f"Improved: {move_impr} ({impr_time:.2f}s)")
        
        # Score moves
        def score_move(move, test_data):
            if move == test_data['best_move']:
                return 3
            elif move in test_data['good_moves']:
                return 2
            else:
                return 0
        
        orig_score = score_move(move_orig, test)
        impr_score = score_move(move_impr, test)
        
        original_score += orig_score
        improved_score += impr_score
        
        print(f"Original score: {orig_score}/3")
        print(f"Improved score: {impr_score}/3")
        
        if impr_score > orig_score:
            print("🎉 Improved engine chose better!")
        elif orig_score > impr_score:
            print("📉 Original engine chose better!")
        else:
            print("🔄 Both engines chose equally well")
    
    print(f"\n{'='*60}")
    print("🏁 FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"📊 Original Total Score: {original_score}/{len(test_positions)*3}")
    print(f"📊 Improved Total Score: {improved_score}/{len(test_positions)*3}")
    print(f"⏱️  Original Average Time: {original_time/len(test_positions):.2f}s")
    print(f"⏱️  Improved Average Time: {improved_time/len(test_positions):.2f}s")
    
    if improved_score > original_score:
        print(f"🎉 IMPROVED ENGINE WINS! (+{improved_score - original_score} points)")
    elif original_score > improved_score:
        print(f"😞 Original engine performed better ({original_score - improved_score} points)")
    else:
        print("🔄 Both engines performed equally well")
    
    if improved_time < original_time:
        print(f"⚡ Improved engine is FASTER! ({original_time - improved_time:.2f}s faster)")
    elif original_time < improved_time:
        print(f"🐌 Improved engine is slower ({improved_time - original_time:.2f}s slower)")
    else:
        print("⏱️ Both engines have similar speed")

if __name__ == "__main__":
    compare_engines() 