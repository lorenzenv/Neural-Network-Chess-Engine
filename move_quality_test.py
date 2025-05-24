#!/usr/bin/env python3
"""
Move Quality Test: Neural Network vs Stockfish Move Selection
This script tests how good the neural network's move choices are compared to Stockfish's recommendations,
focusing on the evaluation difference.
"""

import chess
import chess.pgn
from stockfish import Stockfish
from chess_engine import Engine # Assuming your engine is in chess_engine.py
import random
import sys
import os
import logging

# Configure logging for less verbose output during normal operation
logging.basicConfig(level=logging.INFO, format='%(message)s') # Simpler format for test output
logger = logging.getLogger(__name__)


# Initialize Stockfish
try:
    stockfish_paths = [
        "/opt/homebrew/bin/stockfish",
        "/usr/local/bin/stockfish",
        "/usr/bin/stockfish",
        "stockfish"
    ]
    stockfish = None
    for path in stockfish_paths:
        try:
            # Increased depth for more stable evaluations from Stockfish
            stockfish = Stockfish(path=path, depth=18, parameters={"Threads": 2, "Hash": 1024, "UCI_LimitStrength": "false"})
            logger.info(f"✅ Stockfish initialized at: {path} with depth 18")
            break
        except Exception as e:
            logger.debug(f"Stockfish not found at {path}: {e}") # Debug if path fails
            continue
    if stockfish is None:
        raise Exception("Could not find Stockfish binary in common paths.")
except Exception as e:
    logger.error(f"❌ Failed to initialize Stockfish: {e}")
    sys.exit(1)

def get_stockfish_eval_for_move(fen: str, move_uci: str) -> dict | None:
    """Gets Stockfish's evaluation for a specific FEN after a given move."""
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return {"type": "illegal", "value": None, "display": "Illegal Move"}
        
        board.push(move)
        stockfish.set_fen_position(board.fen())
        eval_data = stockfish.get_evaluation()

        # Stockfish eval is for the position *after* the move, from current player's perspective.
        # We want the value of the move *itself* from the perspective of the player who *made* the move.
        # So, if it's White's turn in 'fen', and White makes 'move_uci', then eval_data is for Black's turn.
        # If eval_data['value'] is +100 (good for current player, Black), it means White's move was bad.
        # Thus, we need to flip the sign for centipawn values.
        
        display_eval = ""
        value_cp = None # Represents the value of the move from the original player's perspective

        if eval_data['type'] == 'cp':
            value_cp = -eval_data['value'] # Flip sign
            display_eval = f"{value_cp / 100.0:+.2f}"
        elif eval_data['type'] == 'mate':
            # If positive, it's mate FOR the current player (opponent of mover)
            # So, if eval_data['value'] = 1, it means opponent can mate in 1. This is bad for mover.
            # If negative, it's mate BY the current player (opponent of mover)
            # So, if eval_data['value'] = -1, it means opponent is mated in 1. This is good for mover.
            value_cp = -eval_data['value'] * 10000 # Assign large value, flip sign
            if eval_data['value'] > 0: # Opponent will mate
                display_eval = f"Mated in {eval_data['value']}"
            else: # Opponent is mated
                display_eval = f"Mate in {abs(eval_data['value'])}"
        
        return {"type": eval_data['type'], "value_cp": value_cp, "display": display_eval, "original_sf_value": eval_data['value']}

    except Exception as e:
        logger.error(f"Error getting Stockfish eval for move {move_uci} on FEN {fen}: {e}")
        return None


def get_stockfish_best_move_eval(fen: str) -> dict | None:
    """Gets Stockfish's best move and its evaluation from the perspective of the player to move."""
    try:
        stockfish.set_fen_position(fen)
        # Get top move, which is what Stockfish itself would play
        top_move_info = stockfish.get_top_moves(1)
        if not top_move_info:
            return None
        
        best_move_uci = top_move_info[0]['Move']
        eval_data = top_move_info[0] # This eval is already from player-to-move's perspective
        
        display_eval = ""
        value_cp = None

        if eval_data['Mate'] is not None:
            value_cp = eval_data['Mate'] * 10000 # Positive if SF mates, negative if SF gets mated
            if eval_data['Mate'] > 0:
                display_eval = f"Mate in {eval_data['Mate']}"
            else: # Should not happen for SF's best move unless it's a forced mate against SF
                display_eval = f"Mated in {abs(eval_data['Mate'])}"
        elif eval_data['Centipawn'] is not None:
            value_cp = eval_data['Centipawn']
            display_eval = f"{value_cp / 100.0:+.2f}"
        
        return {"move_uci": best_move_uci, "type": "cp" if eval_data['Mate'] is None else "mate", 
                "value_cp": value_cp, "display": display_eval}

    except Exception as e:
        logger.error(f"Error getting Stockfish best move for FEN {fen}: {e}")
        return None

def test_position(fen: str, position_name: str, results_summary: list):
    logger.info(f"\n{'='*80}")
    logger.info(f"🧠 TESTING POSITION: {position_name}")
    logger.info(f"{'='*80}")
    logger.info(f"FEN: {fen}")
    
    board = chess.Board(fen)
    logger.info(f"\nPosition:\n{board}")
    logger.info(f"To move: {'White' if board.turn == chess.WHITE else 'Black'}")
    
    # Get Neural Network's choice
    logger.info(f"\n🧠 Neural Network Analysis:")
    neural_move_uci = None
    try:
        engine = Engine(fen) # Assuming Engine is correctly imported
        neural_move_uci = engine.get_move()
        if not neural_move_uci or neural_move_uci in ["checkmate", "draw", "no_legal_moves_or_error"]: # Handle special returns
            logger.warning(f"Engine returned special status: {neural_move_uci}")
            if board.is_game_over():
                 logger.info("Game is already over according to board state.")
            # We might need to handle these cases better if they occur in non-terminal positions
            neural_move_uci = None # Don't proceed with eval if no actual move
    except Exception as e:
        logger.error(f"❌ Neural network failed: {e}")
        results_summary.append({'position': position_name, 'status': 'NN_ERROR'})
        return

    if not neural_move_uci:
        logger.error(f"❌ Neural network did not produce a valid move for FEN: {fen}")
        results_summary.append({'position': position_name, 'status': 'NN_NO_MOVE'})
        return
    logger.info(f"NN Selected move: {neural_move_uci}")

    # Get Stockfish's best move and its evaluation
    logger.info(f"\n🐟 Stockfish Analysis:")
    sf_best_move_data = get_stockfish_best_move_eval(fen)
    if not sf_best_move_data or sf_best_move_data.get("value_cp") is None: # Check if value_cp is None
        logger.error("❌ Failed to get Stockfish best move analysis.")
        results_summary.append({'position': position_name, 'status': 'SF_BEST_MOVE_ERROR'})
        return
    
    logger.info(f"Stockfish best move: {sf_best_move_data['move_uci']} ({sf_best_move_data['display']})")

    # Get Stockfish's evaluation of the Neural Network's chosen move
    nn_move_eval_data = get_stockfish_eval_for_move(fen, neural_move_uci)
    if not nn_move_eval_data or nn_move_eval_data.get("value_cp") is None: # Check if value_cp is None
        logger.error(f"❌ Failed to get Stockfish evaluation for NN move: {neural_move_uci}.")
        results_summary.append({'position': position_name, 'status': 'SF_NN_MOVE_EVAL_ERROR', 'nn_move': neural_move_uci})
        return

    logger.info(f"Stockfish evaluation of NN move '{neural_move_uci}': {nn_move_eval_data['display']}")

    # Performance Comparison
    logger.info(f"\n📊 Neural Network Performance vs Stockfish Best:")
    
    performance_category = "❓ UNKNOWN"
    eval_diff_cp = None

    # Handle cases where one or both evaluations involve mate
    is_sf_best_mate = "mate" in sf_best_move_data['type'] and sf_best_move_data['value_cp'] is not None and abs(sf_best_move_data['value_cp']) > 5000
    is_nn_move_mate = "mate" in nn_move_eval_data['type'] and nn_move_eval_data['value_cp'] is not None and abs(nn_move_eval_data['value_cp']) > 5000


    if is_sf_best_mate and sf_best_move_data['value_cp'] > 0: # Stockfish finds a mate for itself
        if is_nn_move_mate and nn_move_eval_data['value_cp'] > 0:
            if nn_move_eval_data['value_cp'] >= sf_best_move_data['value_cp']: # NN finds same or faster mate
                 performance_category = "🏆 EXCELLENT - Found optimal mate!"
            else: # NN finds a mate, but slower
                 performance_category = "🥈 GOOD - Found a mate, but slower than optimal."
        else: # Stockfish finds mate, NN does not
            performance_category = "❌ POOR - Missed a forced mate."
    elif is_nn_move_mate and nn_move_eval_data['value_cp'] < 0 : # NN gets itself mated by playing the move
        performance_category = "💣 BLUNDER - Plays into a mate."
    elif is_sf_best_mate and sf_best_move_data['value_cp'] < 0: # Stockfish is getting mated
        if is_nn_move_mate and nn_move_eval_data['value_cp'] < 0 and nn_move_eval_data['value_cp'] <= sf_best_move_data['value_cp']:
            performance_category = "✅ OKAY - Gets mated, but optimally delays or chooses same losing line."
        else:
            performance_category = "⚠️  POOR - Gets mated faster or avoids best losing line."
    # Centipawn comparison if no decisive mates are involved for SF's best move
    elif sf_best_move_data['value_cp'] is not None and nn_move_eval_data['value_cp'] is not None:
        eval_diff_cp = sf_best_move_data['value_cp'] - nn_move_eval_data['value_cp']
        
        # Adjust for player to move (SF eval is always from player-to-move's perspective)
        # No, both value_cp are now from original player-to-move's perspective.
        
        logger.info(f"   SF Best Eval (CP): {sf_best_move_data['value_cp']:.0f}, NN Move Eval (CP): {nn_move_eval_data['value_cp']:.0f}")
        logger.info(f"   Evaluation difference (SF_best - NN_move): {eval_diff_cp:.0f} cp")

        if eval_diff_cp <= 10: # Effectively same or better
            performance_category = "🏆 EXCELLENT - Optimal or near-optimal move!"
        elif eval_diff_cp <= 30:
            performance_category = "🥈 VERY GOOD - Strong move, very close to optimal."
        elif eval_diff_cp <= 60:
            performance_category = "🥉 GOOD - Solid move."
        elif eval_diff_cp <= 100:
            performance_category = "⚠️  OKAY - Playable, but noticeably weaker."
        elif eval_diff_cp <= 200:
            performance_category = "❌ POOR - Significant error."
        else:
            performance_category = "💣 BLUNDER - Very serious error."
    else:
        logger.warning("Could not compare evaluations due to missing centipawn values or unhandled mate logic.")


    logger.info(f"   Performance: {performance_category}")
    results_summary.append({
        'position': position_name,
        'fen': fen,
        'nn_move': neural_move_uci,
        'sf_best_move': sf_best_move_data['move_uci'],
        'sf_best_eval_display': sf_best_move_data['display'],
        'nn_move_eval_display': nn_move_eval_data['display'],
        'sf_best_cp': sf_best_move_data['value_cp'],
        'nn_move_cp': nn_move_eval_data['value_cp'],
        'eval_diff_cp': eval_diff_cp,
        'category': performance_category,
        'status': 'OK'
    })


def main():
    logger.info("🚀 Testing Neural Network Move Selection Quality (Comparison by Evaluation Difference)")
    
    test_positions = [
        {"name": "test1_opening_blacks_reply", "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"},
        {"name": "test2_middlegame_tactical", "fen": "r3k2r/pp1b1ppp/2n2n2/1B2N3/Q2Pq3/2P1B3/P4PPP/R3K2R w KQkq - 1 13"},
        {"name": "test3_sicilian_opening_white", "fen": "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"},
        {"name": "test4_middlegame_material_imbalance", "fen": "7r/pp1k1p2/2pN1npp/8/8/BP6/P4PnP/2KR4 w - - 0 23"},
        {"name": "test5_endgame_king_pawn", "fen": "8/8/1k6/8/8/8/4P3/3K4 w - - 0 1"},
        {"name": "test6_tactical_pin_black", "fen": "5k2/r2p4/3Np1RP/2PnP3/5P2/1p1N3P/1P1K4/r7 b - - 0 47"},
        {"name": "test7_endgame_promotion_race_black", "fen": "5k2/r2p3P/3Np1R1/2PnP3/5P2/1p1N3P/1P1K4/7r b - - 0 48"},
        {"name": "test8_middlegame_queen_attack_black", "fen": "k3r3/5p2/pqbR4/5Ppp/3B4/1P3P2/1Q4PP/6K1 b - - 2 29"},
        {"name": "test9_kings_indian_attack_white", "fen": "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP3PPP/R1BQKB1R w KQ - 2 6"},
        {"name": "test10_endgame_rook_pawns_white", "fen": "8/5p2/R7/5k2/8/8/P4P2/6K1 w - - 1 36"}
    ]
    
    overall_results_summary = []
    
    for position_data in test_positions:
        test_position(position_data['fen'], position_data['name'], overall_results_summary)
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("📊 FINAL SUMMARY OF NEURAL NETWORK PERFORMANCE")
    logger.info(f"{'='*80}")
    
    if overall_results_summary:
        total_tests = len([r for r in overall_results_summary if r['status'] == 'OK'])
        logger.info(f"Total valid positions tested: {total_tests}")

        categories_count = {
            "🏆 EXCELLENT - Optimal or near-optimal move!": 0,
            "🏆 EXCELLENT - Found optimal mate!": 0,
            "🥈 VERY GOOD - Strong move, very close to optimal.": 0,
            "🥉 GOOD - Solid move.": 0,
            "🥈 GOOD - Found a mate, but slower than optimal.":0,
            "⚠️  OKAY - Playable, but noticeably weaker.": 0,
            "✅ OKAY - Gets mated, but optimally delays or chooses same losing line.":0,
            "❌ POOR - Significant error.": 0,
            "❌ POOR - Missed a forced mate.":0,
            "⚠️  POOR - Gets mated faster or avoids best losing line.":0,
            "💣 BLUNDER - Very serious error.": 0,
            "💣 BLUNDER - Plays into a mate.":0,
            "❓ UNKNOWN": 0
        }
        
        successful_tests = 0
        total_eval_diff = 0

        for r in overall_results_summary:
            if r['status'] == 'OK':
                successful_tests +=1
                if r['category'] in categories_count:
                    categories_count[r['category']] += 1
                if r['eval_diff_cp'] is not None and not ("mate" in r['sf_best_eval_display'].lower() or "mate" in r['nn_move_eval_display'].lower()): # Only average non-mate situations
                    total_eval_diff += r['eval_diff_cp']
                
                logger.info(f"  Pos: {r['position'][:30]:<30} | NN: {r['nn_move']:<7} (Eval: {r['nn_move_eval_display']:>8}) | SF: {r['sf_best_move']:<7} (Eval: {r['sf_best_eval_display']:>8}) | Diff: {(str(int(r['eval_diff_cp']))+'cp') if r['eval_diff_cp'] is not None else 'N/A':>7} | Result: {r['category']}")

        if successful_tests > 0:
            non_mate_diffs = [r['eval_diff_cp'] for r in overall_results_summary if r['status'] == 'OK' and r['eval_diff_cp'] is not None and not ("mate" in r['sf_best_eval_display'].lower() or "mate" in r['nn_move_eval_display'].lower())]
            if non_mate_diffs:
                 average_eval_diff = sum(non_mate_diffs) / len(non_mate_diffs)
                 logger.info(f"\nAverage Evaluation Difference (SF_best - NN_move, non-mate positions): {average_eval_diff:.0f} cp")
            else:
                logger.info("\nNo non-mate positions with comparable centipawn evaluations to average.")


        logger.info("\nCategory Counts:")
        for cat, count in categories_count.items():
            if count > 0 : # Only print categories that occurred
                logger.info(f"  {cat}: {count}")

        # Simplified overall assessment based on average difference or mate performance
        # This part can be made more sophisticated
        if successful_tests > 0:
            excellent_cats = ["🏆 EXCELLENT - Optimal or near-optimal move!", "🏆 EXCELLENT - Found optimal mate!"]
            good_cats = ["🥈 VERY GOOD - Strong move, very close to optimal.", "🥉 GOOD - Solid move.", "🥈 GOOD - Found a mate, but slower than optimal."]
            
            if sum(categories_count[cat] for cat in excellent_cats) / successful_tests >= 0.25: # 25% excellent
                logger.info("\nOverall: Commendable performance, often finding top-tier moves.")
            elif (sum(categories_count[cat] for cat in excellent_cats) + sum(categories_count[cat] for cat in good_cats)) / successful_tests >= 0.5: # 50% good or better
                logger.info("\nOverall: Solid performance, consistently finding reasonable moves.")
            else:
                logger.info("\nOverall: Room for improvement, struggles with consistency.")


    logger.info(f"\n✅ Move quality test completed!")

if __name__ == "__main__":
    main()