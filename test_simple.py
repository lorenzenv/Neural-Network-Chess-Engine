import chess

# Simple mock evaluation function for testing
def mock_evaluate_pos(first_fen, second_fen):
    """Mock evaluation that returns a simple score based on material"""
    board = chess.Board(second_fen)
    
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
    
    # Add some positional bonus
    if board.turn == chess.BLACK:
        score = -score
    
    return score

class TestEngineOriginal:
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

class TestEngineNew:
    def __init__(self, fen):
        self.board = chess.Board()
        self.board.set_fen(fen)
        self.position_cache = {}

    def get_move(self):
        best_move = self.make_move()
        return str(best_move)

    def make_move(self):
        current_fen = self.board.fen()
        
        black_legal_moves = list(self.board.legal_moves)
        if not black_legal_moves:
            return "no legal moves"
        
        best_move = None
        best_value = float('-inf')
        
        for black_move in black_legal_moves:
            self.board.push(black_move)
            
            if self.board.is_checkmate():
                self.board.pop()
                return str(black_move)
            
            white_value = self.get_white_response(current_fen, float('-inf'), float('inf'))
            self.board.pop()
            
            if white_value > best_value:
                best_value = white_value
                best_move = black_move
        
        if best_move is None:
            return "checkmate"
        
        return str(best_move)
    
    def get_white_response(self, original_fen, alpha, beta):
        white_legal_moves = list(self.board.legal_moves)
        if not white_legal_moves:
            return 0
        
        min_value = float('inf')
        
        for white_move in white_legal_moves:
            self.board.push(white_move)
            
            if self.board.is_checkmate():
                self.board.pop()
                return -10000
            
            black_value = self.get_black_counter_response(original_fen, alpha, beta)
            self.board.pop()
            
            min_value = min(min_value, black_value)
            beta = min(beta, black_value)
            
            if beta <= alpha:
                break
                
        return min_value
    
    def get_black_counter_response(self, original_fen, alpha, beta):
        black_legal_moves = list(self.board.legal_moves)
        if not black_legal_moves:
            return 0
        
        max_value = float('-inf')
        
        for black_move in black_legal_moves:
            self.board.push(black_move)
            
            if self.board.is_checkmate():
                self.board.pop()
                return 10000
            
            current_fen = self.board.fen()
            if current_fen in self.position_cache:
                evaluation = self.position_cache[current_fen]
            else:
                evaluation = mock_evaluate_pos(original_fen, current_fen)
                self.position_cache[current_fen] = evaluation
                
            self.board.pop()
            
            max_value = max(max_value, evaluation)
            alpha = max(alpha, evaluation)
            
            if beta <= alpha:
                break
                
        return max_value

def simple_test():
    # Test position where Black can capture material
    test_fen = "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR b KQkq - 2 3"
    
    print("Testing Original Engine:")
    engine1 = TestEngineOriginal(test_fen)
    move1 = engine1.get_move()
    print(f"Original chose: {move1}")
    
    print("\nTesting New Engine:")
    engine2 = TestEngineNew(test_fen)
    move2 = engine2.get_move()
    print(f"New chose: {move2}")
    
    print(f"\nMoves are {'SAME' if move1 == move2 else 'DIFFERENT'}")

if __name__ == "__main__":
    simple_test() 