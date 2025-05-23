import chess

def debug_position():
    # Simpler test position - Black can win material by taking on c4
    test_fen = "rnbqkbnr/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR b KQkq - 1 2"
    
    board = chess.Board(test_fen)
    print(f"Position: {test_fen}")
    print(f"Turn: {'Black' if board.turn == chess.BLACK else 'White'}")
    print()
    print(board)
    print()
    
    legal_moves = list(board.legal_moves)
    print(f"Legal moves for Black: {len(legal_moves)}")
    for i, move in enumerate(legal_moves):
        print(f"{i+1}. {move}")
    
    print()
    # Check if any move captures material (correctly this time)
    for move in legal_moves:
        if board.piece_at(move.to_square) is not None:
            captured_piece = board.piece_at(move.to_square)
            print(f"CAPTURE: {move} captures {captured_piece}")

def debug_simple_position():
    # Very simple position - Black knight can take White pawn
    test_fen = "8/8/8/4n3/3P4/8/8/8 b - - 0 1"
    
    board = chess.Board(test_fen)
    print(f"\nSimple Position: {test_fen}")
    print(board)
    
    legal_moves = list(board.legal_moves)
    print(f"Legal moves: {[str(m) for m in legal_moves]}")
    
    for move in legal_moves:
        if board.piece_at(move.to_square) is not None:
            captured_piece = board.piece_at(move.to_square)
            print(f"CAPTURE: {move} captures {captured_piece}")

if __name__ == "__main__":
    debug_position()
    debug_simple_position() 