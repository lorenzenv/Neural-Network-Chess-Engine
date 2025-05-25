#!/usr/bin/env python3
"""
Move Quality Test: Neural Network vs Stockfish Move Selection
This script tests how good the neural network's move choices are compared to Stockfish's recommendations,
focusing on the evaluation difference.

Enhanced with automatic position generation from multiple sources:
- PGN game analysis
- EPD test suites  
- Tactical position generation
- Famous games database
- Online position fetching
"""

import chess
import chess.pgn
import chess.engine
from stockfish import Stockfish
from search_coordinator import PureNeuralNetworkEngine as Engine
import random
import sys
import os
import logging
import argparse
import json
import requests
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re

# Configure logging for less verbose output during normal operation
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class PositionGenerator:
    """Generates test positions from various sources"""
    
    def __init__(self):
        self.positions = []
        
    def load_epd_file(self, epd_path: str) -> List[Dict]:
        """Load positions from EPD (Extended Position Description) files"""
        positions = []
        try:
            with open(epd_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                        
                    # Basic FEN is first 4 parts
                    fen = ' '.join(parts[:4])
                    
                    # Parse EPD operations (like bm for best move, id for identifier)
                    operations = {}
                    for part in parts[4:]:
                        if '=' in part:
                            key, value = part.split('=', 1)
                            operations[key] = value.strip('"')
                    
                    positions.append({
                        'name': f'epd_{Path(epd_path).stem}_line_{line_num}',
                        'fen': fen,
                        'source': 'epd',
                        'best_move': operations.get('bm', ''),
                        'id': operations.get('id', f'position_{line_num}'),
                        'operations': operations
                    })
                    
        except FileNotFoundError:
            logger.warning(f"EPD file not found: {epd_path}")
        except Exception as e:
            logger.error(f"Error loading EPD file {epd_path}: {e}")
            
        return positions
    
    def extract_from_pgn(self, pgn_path: str, max_positions: int = 50) -> List[Dict]:
        """Extract interesting positions from PGN games"""
        positions = []
        try:
            with open(pgn_path, 'r') as f:
                game_count = 0
                while len(positions) < max_positions:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                        
                    game_count += 1
                    board = game.board()
                    move_count = 0
                    
                    for move in game.mainline_moves():
                        move_count += 1
                        board.push(move)
                        
                        # Extract positions at interesting moments
                        if self._is_interesting_position(board, move_count):
                            positions.append({
                                'name': f'pgn_game{game_count}_move{move_count}',
                                'fen': board.fen(),
                                'source': 'pgn',
                                'game_info': {
                                    'white': game.headers.get('White', 'Unknown'),
                                    'black': game.headers.get('Black', 'Unknown'),
                                    'result': game.headers.get('Result', '*'),
                                    'move_number': move_count
                                }
                            })
                            
                        if len(positions) >= max_positions:
                            break
                            
        except FileNotFoundError:
            logger.warning(f"PGN file not found: {pgn_path}")
        except Exception as e:
            logger.error(f"Error parsing PGN file {pgn_path}: {e}")
            
        return positions
    
    def _is_interesting_position(self, board: chess.Board, move_count: int) -> bool:
        """Determine if a position is worth testing"""
        # Skip very early opening moves
        if move_count < 8:
            return False
            
        # Skip very long games (likely drawish)
        if move_count > 80:
            return False
            
        # Include positions with:
        # - Captures
        # - Checks  
        # - Castling rights about to be lost
        # - Piece development completed (middlegame)
        # - Endgame with few pieces
        
        piece_count = len(board.piece_map())
        
        # Middlegame positions (piece development phase)
        if 12 <= move_count <= 25 and piece_count >= 20:
            return random.random() < 0.3
            
        # Tactical positions (fewer pieces, likely more tactics)
        if piece_count <= 16:
            return random.random() < 0.4
            
        # Random sampling for variety
        return random.random() < 0.1
    
    def generate_tactical_positions(self, count: int = 20) -> List[Dict]:
        """Generate tactical positions using Stockfish analysis of random games"""
        positions = []
        
        try:
            logger.info(f"Generating {count} tactical positions using Stockfish analysis...")
            
            # Start from random starting positions and let Stockfish find tactical moments
            for i in range(count * 3):  # Generate more than needed, filter best
                # Create random middle-game-ish positions by playing random moves
                board = chess.Board()
                
                # Play 8-15 random moves to get past opening
                opening_moves = random.randint(8, 15)
                for _ in range(opening_moves):
                    if board.legal_moves:
                        move = random.choice(list(board.legal_moves))
                        board.push(move)
                    else:
                        break
                
                if board.is_game_over():
                    continue
                    
                fen = board.fen()
                
                # Use Stockfish to analyze if this position has tactical content
                stockfish.set_fen_position(fen)
                top_moves = stockfish.get_top_moves(3)
                
                if not top_moves:
                    continue
                    
                # Look for positions where there's a big difference between best and 2nd best move
                best_eval = top_moves[0].get('Centipawn', 0)
                if len(top_moves) > 1:
                    second_eval = top_moves[1].get('Centipawn', 0)
                    eval_gap = abs(best_eval - second_eval)
                    
                    # If there's a big gap (>100cp), it's likely tactical
                    if eval_gap > 100:
                        positions.append({
                            'name': f'stockfish_tactical_{len(positions)+1}',
                            'fen': fen,
                            'source': 'stockfish_generator',
                            'pattern': 'tactical',
                            'eval_gap': eval_gap,
                            'best_move': top_moves[0]['Move']
                        })
                        
                        if len(positions) >= count:
                            break
            
        except Exception as e:
            logger.warning(f"Error generating Stockfish tactical positions: {e}")
            
        logger.info(f"Generated {len(positions)} tactical positions from Stockfish analysis")
        return positions[:count]
    
    def fetch_lichess_games(self, count: int = 10) -> List[Dict]:
        """Fetch real games from Lichess API"""
        positions = []
        
        try:
            import requests
            import time
            
            logger.info(f"Fetching {count} positions from Lichess API...")
            
            # Fetch recent games from strong players
            strong_players = ['hikaru', 'daniil_dubov', 'liem_chess', 'penguingm1', 'grandelius']
            
            for player in strong_players[:2]:  # Limit to avoid rate limiting
                try:
                    # Get recent games from this player
                    url = f"https://lichess.org/api/games/user/{player}"
                    params = {
                        'max': 5,
                        'rated': 'true',
                        'perfType': 'blitz,rapid,classical',
                        'format': 'pgn'
                    }
                    
                    headers = {'Accept': 'application/x-ndjson'}
                    response = requests.get(url, params=params, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        # Parse the PGN response
                        games_text = response.text.strip()
                        if games_text:
                            # Create a temporary file and extract positions
                            import tempfile
                            with tempfile.NamedTemporaryFile(mode='w', suffix='.pgn', delete=False) as f:
                                f.write(games_text)
                                temp_pgn_path = f.name
                            
                            # Extract positions from this PGN
                            try:
                                game_positions = self.extract_from_pgn(temp_pgn_path, max_positions=5)
                                for pos in game_positions:
                                    pos['source'] = 'lichess_api'
                                    pos['player'] = player
                                positions.extend(game_positions)
                                
                                if len(positions) >= count:
                                    break
                                    
                            finally:
                                # Clean up temp file
                                import os
                                try:
                                    os.unlink(temp_pgn_path)
                                except:
                                    pass
                    
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    logger.warning(f"Error fetching games from {player}: {e}")
                    continue
                    
        except ImportError:
            logger.warning("requests module not available for Lichess API")
        except Exception as e:
            logger.warning(f"Error fetching from Lichess API: {e}")
            
        logger.info(f"Fetched {len(positions)} positions from Lichess API")
        return positions[:count]
    
    def download_famous_test_suites(self) -> List[Dict]:
        """Download real chess test suites from the internet"""
        positions = []
        
        try:
            import requests
            import tempfile
            import os
            
            # Famous test suites with direct download URLs
            test_suites = {
                'wac': {
                    'url': 'https://raw.githubusercontent.com/official-stockfish/books/master/wac.epd',
                    'description': 'Win At Chess - 300 tactical positions'
                },
                'bratko_kopec': {
                    'url': 'https://raw.githubusercontent.com/official-stockfish/books/master/bratko-kopec.epd', 
                    'description': 'Bratko-Kopec Test - 24 strategic positions'
                }
            }
            
            for suite_name, suite_info in test_suites.items():
                try:
                    logger.info(f"Downloading {suite_info['description']}...")
                    response = requests.get(suite_info['url'], timeout=30)
                    
                    if response.status_code == 200:
                        # Save to temp file and load
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.epd', delete=False) as f:
                            f.write(response.text)
                            temp_epd_path = f.name
                        
                        try:
                            # Load positions from downloaded EPD
                            epd_positions = self.load_epd_file(temp_epd_path)
                            
                            # Take a sample to avoid overwhelming
                            sample_size = min(10, len(epd_positions))
                            sampled_positions = random.sample(epd_positions, sample_size)
                            
                            for pos in sampled_positions:
                                pos['source'] = f'{suite_name}_downloaded'
                                pos['test_suite'] = suite_info['description']
                            
                            positions.extend(sampled_positions)
                            logger.info(f"Loaded {len(sampled_positions)} positions from {suite_name}")
                            
                        finally:
                            try:
                                os.unlink(temp_epd_path)
                            except:
                                pass
                                
                except Exception as e:
                    logger.warning(f"Could not download {suite_name}: {e}")
                    
        except ImportError:
            logger.warning("requests module not available for downloading test suites")
        except Exception as e:
            logger.warning(f"Error downloading test suites: {e}")
            
        return positions
    
    def fetch_online_positions(self, count: int = 10) -> List[Dict]:
        """Fetch positions from multiple real online sources"""
        positions = []
        
        # Try multiple real sources
        sources = [
            ('lichess_games', lambda: self.fetch_lichess_games(count // 2)),
            ('downloaded_suites', lambda: self.download_famous_test_suites()),
        ]
        
        for source_name, fetch_func in sources:
            try:
                source_positions = fetch_func()
                positions.extend(source_positions)
                
                if len(positions) >= count:
                    break
                    
            except Exception as e:
                logger.warning(f"Error fetching from {source_name}: {e}")
                
        return positions[:count]
    
    def get_all_positions(self, 
                         include_default: bool = True,
                         pgn_files: List[str] = None,
                         epd_files: List[str] = None,
                         tactical_count: int = 20,
                         online_count: int = 10,
                         max_total: int = 100) -> List[Dict]:
        """Get comprehensive test positions from all sources"""
        
        all_positions = []
        
        # Include default hardcoded positions
        if include_default:
            default_positions = [
                {"name": "test1_opening_blacks_reply", "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2", "source": "default"},
                {"name": "test2_middlegame_tactical", "fen": "r3k2r/pp1b1ppp/2n2n2/1B2N3/Q2Pq3/2P1B3/P4PPP/R3K2R w KQkq - 1 13", "source": "default"},
                {"name": "test4_middlegame_material_imbalance", "fen": "7r/pp1k1p2/2pN1npp/8/8/BP6/P4PnP/2KR4 w - - 0 23", "source": "default"},
                {"name": "test5_endgame_king_pawn", "fen": "8/8/1k6/8/8/8/4P3/3K4 w - - 0 1", "source": "default"},
                {"name": "test6_tactical_pin_black", "fen": "5k2/r2p4/3Np1RP/2PnP3/5P2/1p1N3P/1P1K4/r7 b - - 0 47", "source": "default"},
                {"name": "test7_endgame_promotion_race_black", "fen": "5k2/r2p3P/3Np1R1/2PnP3/5P2/1p1N3P/1P1K4/7r b - - 0 48", "source": "default"},
                {"name": "test8_middlegame_queen_attack_black", "fen": "k3r3/5p2/pqbR4/5Ppp/3B4/1P3P2/1Q4PP/6K1 b - - 2 29", "source": "default"},
                {"name": "test9_kings_indian_attack_white", "fen": "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP3PPP/R1BQKB1R w KQ - 2 6", "source": "default"},
                {"name": "test10_endgame_rook_pawns_white", "fen": "8/5p2/R7/5k2/8/8/P4P2/6K1 w - - 1 36", "source": "default"}
            ]
            all_positions.extend(default_positions)
        
        # Load from PGN files
        if pgn_files:
            for pgn_file in pgn_files:
                if os.path.exists(pgn_file):
                    pgn_positions = self.extract_from_pgn(pgn_file, max_positions=20)
                    all_positions.extend(pgn_positions)
                    logger.info(f"Loaded {len(pgn_positions)} positions from {pgn_file}")
        
        # Load from EPD files
        if epd_files:
            for epd_file in epd_files:
                if os.path.exists(epd_file):
                    epd_positions = self.load_epd_file(epd_file)
                    all_positions.extend(epd_positions)
                    logger.info(f"Loaded {len(epd_positions)} positions from {epd_file}")
        
        # Generate tactical positions
        if tactical_count > 0:
            tactical_positions = self.generate_tactical_positions(tactical_count)
            all_positions.extend(tactical_positions)
            logger.info(f"Generated {len(tactical_positions)} tactical positions")
        
        # Fetch online positions
        if online_count > 0:
            online_positions = self.fetch_online_positions(online_count)
            all_positions.extend(online_positions)
            logger.info(f"Fetched {len(online_positions)} online positions")
        
        # Shuffle and limit
        random.shuffle(all_positions)
        final_positions = all_positions[:max_total]
        
        logger.info(f"Total positions available: {len(all_positions)}")
        logger.info(f"Running test on: {len(final_positions)} positions")
        
        # Show distribution by source
        sources = {}
        for pos in final_positions:
            source = pos.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        
        logger.info("Position sources:")
        for source, count in sources.items():
            logger.info(f"  {source}: {count}")
        
        return final_positions

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
    """Gets Stockfish's evaluation for a specific FEN after a given move using search-based evaluation."""
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return {"type": "illegal", "value_cp": None, "display": "Illegal Move"}
        
        board.push(move)
        
        # Use the same search-based evaluation method as get_stockfish_best_move_eval
        stockfish.set_fen_position(board.fen())
        top_move_info = stockfish.get_top_moves(1)
        
        if not top_move_info:
            # Position might be terminal or Stockfish couldn't find moves
            return {"type": "draw", "value_cp": 0, "display": "Draw/Terminal"}
        
        eval_data = top_move_info[0]  # Get eval from the position after the move
        
        # This eval is from the perspective of the current player (opponent of the original mover)
        # We want it from the original mover's perspective, so flip the sign
        display_eval = ""
        value_cp = None

        if eval_data['Mate'] is not None:
            value_cp = eval_data['Mate'] * 10000  # No sign flip
            if eval_data['Mate'] > 0:  # Positive mate value
                display_eval = f"Mate in {eval_data['Mate']}"
            else:  # Negative mate value  
                display_eval = f"Mated in {abs(eval_data['Mate'])}"
        elif eval_data['Centipawn'] is not None:
            value_cp = eval_data['Centipawn']  # No sign flip
            display_eval = f"{value_cp / 100.0:+.2f}"
        
        return {"type": "cp" if eval_data['Mate'] is None else "mate", 
                "value_cp": value_cp, "display": display_eval}

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
        
        # Ensure both evaluations are from White's Point of View (WPOV) for consistent comparison
        sf_best_cp_wpov = sf_best_move_data['value_cp']
        nn_move_cp_wpov = nn_move_eval_data['value_cp']
        current_player_is_black = (board.turn == chess.BLACK)

        # The *_move_data['value_cp'] fields are from the perspective of the player whose turn it was at the original FEN.
        # So, if it was Black's turn, a positive value_cp is good for Black.
        # For WPOV comparison, we need to negate Black's scores.
        if current_player_is_black:
            sf_best_cp_wpov = -sf_best_cp_wpov
            nn_move_cp_wpov = -nn_move_cp_wpov

        eval_diff_cp = sf_best_cp_wpov - nn_move_cp_wpov # Difference of WPOV scores
        
        logger.info(f"   SF Best Eval (WPOV CP): {sf_best_cp_wpov:.0f}, NN Move Eval (WPOV CP): {nn_move_cp_wpov:.0f}")
        logger.info(f"   Evaluation difference (SF_best_WPOV - NN_move_WPOV): {eval_diff_cp:.0f} cp")

        # Thresholds are based on the magnitude of loss relative to SF's WPOV best.
        # A positive eval_diff_cp means SF's best (WPOV) was better than NN's move (WPOV).
        # A negative eval_diff_cp means NN's move (WPOV) was better than SF's best (WPOV).

        if eval_diff_cp <= 10: # NN is same or better (or negligibly worse)
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
        # Store original display values, but WPOV cp values for consistent summary calculation
        'sf_best_eval_display': sf_best_move_data['display'], 
        'nn_move_eval_display': nn_move_eval_data['display'],
        'sf_best_cp_wpov': sf_best_cp_wpov if 'sf_best_cp_wpov' in locals() else sf_best_move_data.get('value_cp'), # fallback for mate cases
        'nn_move_cp_wpov': nn_move_cp_wpov if 'nn_move_cp_wpov' in locals() else nn_move_eval_data.get('value_cp'), # fallback for mate cases
        'eval_diff_cp': eval_diff_cp,
        'category': performance_category,
        'status': 'OK'
    })

def test_speed_modes(fen: str, position_name: str):
    """Test the same position with different speed modes"""
    logger.info(f"\n🏃 SPEED MODE TESTING: {position_name}")
    logger.info(f"FEN: {fen}")
    
    speed_modes = ["fast", "balanced", "strong"]
    results = {}
    
    for mode in speed_modes:
        logger.info(f"\n--- Testing {mode.upper()} mode ---")
        
        # Temporarily modify the speed mode in chess_engine.py
        import importlib
        import chess_engine
        
        # Store original values
        original_mode = chess_engine.Config.SPEED_MODE
        original_time_limit = chess_engine.Config.ITERATIVE_DEEPENING_TIME_LIMIT_PER_MOVE
        original_depth = chess_engine.Config.MAX_SEARCH_DEPTH_ID
        original_blend = chess_engine.Config.NN_EVALUATION_BLEND
        
        try:
            # Set new speed mode
            chess_engine.Config.SPEED_MODE = mode
            if mode == "fast":
                chess_engine.Config.MAX_SEARCH_DEPTH_ID = 6
                chess_engine.Config.ITERATIVE_DEEPENING_TIME_LIMIT_PER_MOVE = 3.0
                chess_engine.Config.NN_EVALUATION_BLEND = 0.9
            elif mode == "balanced":
                chess_engine.Config.MAX_SEARCH_DEPTH_ID = 6
                chess_engine.Config.ITERATIVE_DEEPENING_TIME_LIMIT_PER_MOVE = 5.0
                chess_engine.Config.NN_EVALUATION_BLEND = 0.8
            else:  # strong
                chess_engine.Config.MAX_SEARCH_DEPTH_ID = 8
                chess_engine.Config.ITERATIVE_DEEPENING_TIME_LIMIT_PER_MOVE = 15.0
                chess_engine.Config.NN_EVALUATION_BLEND = 0.7
            
            # Test the position
            start_time = time.time()
            engine = chess_engine.Engine(fen)
            neural_move_uci = engine.get_move()
            total_time = time.time() - start_time
            
            # Get Stockfish evaluation
            sf_best_move_data = get_stockfish_best_move_eval(fen)
            nn_move_eval_data = get_stockfish_eval_for_move(fen, neural_move_uci)
            
            eval_diff = None
            if (sf_best_move_data and nn_move_eval_data and 
                sf_best_move_data.get('value_cp') is not None and 
                nn_move_eval_data.get('value_cp') is not None):
                
                board = chess.Board(fen)
                current_player_is_black = (board.turn == chess.BLACK)
                
                sf_best_cp_wpov = sf_best_move_data['value_cp']
                nn_move_cp_wpov = nn_move_eval_data['value_cp']
                
                if current_player_is_black:
                    sf_best_cp_wpov = -sf_best_cp_wpov
                    nn_move_cp_wpov = -nn_move_cp_wpov
                
                eval_diff = sf_best_cp_wpov - nn_move_cp_wpov
            
            results[mode] = {
                'move': neural_move_uci,
                'time': total_time,
                'sf_best': sf_best_move_data['move_uci'] if sf_best_move_data else None,
                'eval_diff': eval_diff
            }
            
            logger.info(f"  Move: {neural_move_uci}")
            logger.info(f"  Time: {total_time:.2f}s")
            logger.info(f"  SF Best: {sf_best_move_data['move_uci'] if sf_best_move_data else 'N/A'}")
            logger.info(f"  Eval diff: {eval_diff:.0f}cp" if eval_diff is not None else "  Eval diff: N/A")
            
        finally:
            # Restore original values
            chess_engine.Config.SPEED_MODE = original_mode
            chess_engine.Config.ITERATIVE_DEEPENING_TIME_LIMIT_PER_MOVE = original_time_limit
            chess_engine.Config.MAX_SEARCH_DEPTH_ID = original_depth
            chess_engine.Config.NN_EVALUATION_BLEND = original_blend
    
    # Summary
    logger.info(f"\n📊 Speed Mode Comparison:")
    logger.info(f"{'-'*50}")
    
    for mode, data in results.items():
        eval_str = f"{data['eval_diff']:.0f}cp" if data['eval_diff'] is not None else "N/A"
        logger.info(f"{mode:<10} {data['move']:<8} {data['time']:<6.1f}s {data['sf_best']:<8} {eval_str:<10}")
    
    return results

def setup_test_databases():
    """Download and setup real chess test databases"""
    logger.info("🔧 Setting up real chess test databases...")
    
    setup_dir = Path("test_databases")
    setup_dir.mkdir(exist_ok=True)
    
    try:
        import requests
        
        # Download real chess databases
        databases = {
            "bratko_kopec.epd": {
                "url": "https://raw.githubusercontent.com/ChrisWhittington/Chess-EPDs/master/bratko-kopec.epd",
                "description": "Bratko-Kopec Test - 24 strategic positions"
            },
            "kaufman.epd": {
                "url": "https://raw.githubusercontent.com/ChrisWhittington/Chess-EPDs/master/kaufman.epd",
                "description": "Larry Kaufman Test Suite - Strategic positions"
            },
            "lct2.epd": {
                "url": "https://raw.githubusercontent.com/ChrisWhittington/Chess-EPDs/master/lct2.epd",
                "description": "Lomonosov Chessbase Test 2 - Famous tactical suite"
            },
            "silent_but_deadly.epd": {
                "url": "https://raw.githubusercontent.com/ChrisWhittington/Chess-EPDs/master/silent-but-deadly.epd",
                "description": "Dann Corbitt Quiet Positions - Strategic test suite"
            }
        }
        
        for filename, db_info in databases.items():
            filepath = setup_dir / filename
            if not filepath.exists():
                try:
                    logger.info(f"Downloading {db_info['description']}...")
                    response = requests.get(db_info['url'], timeout=60)
                    
                    if response.status_code == 200:
                        filepath.write_text(response.text)
                        logger.info(f"✅ Downloaded {filename} ({len(response.text)} bytes)")
                    else:
                        logger.warning(f"❌ Failed to download {filename}: HTTP {response.status_code}")
                        
                except Exception as e:
                    logger.warning(f"❌ Error downloading {filename}: {e}")
            else:
                logger.info(f"📁 {filename} already exists, skipping download")
        
        # Download sample PGN from Lichess API
        sample_pgn = setup_dir / "lichess_sample.pgn"
        if not sample_pgn.exists():
            try:
                logger.info("Downloading sample games from Lichess...")
                # Get games from a strong player
                url = "https://lichess.org/api/games/user/hikaru"
                params = {
                    'max': 10,
                    'rated': 'true',
                    'perfType': 'classical,rapid',
                    'format': 'pgn'
                }
                headers = {'Accept': 'application/x-chess-pgn'}
                
                response = requests.get(url, params=params, headers=headers, timeout=30)
                
                if response.status_code == 200 and response.text.strip():
                    sample_pgn.write_text(response.text)
                    logger.info(f"✅ Downloaded sample games from Lichess")
                else:
                    logger.info("⚠️  Could not download Lichess games, will generate via Stockfish instead")
                    
            except Exception as e:
                logger.warning(f"❌ Error downloading Lichess games: {e}")
        
        logger.info(f"\n🎯 Test databases setup complete in: {setup_dir}")
        logger.info("\n📖 Usage examples:")
        
        # Show actual files that exist
        epd_files = list(setup_dir.glob("*.epd"))
        pgn_files = list(setup_dir.glob("*.pgn"))
        
        if epd_files:
            logger.info(f"  python3 move_quality_test.py --epd {epd_files[0]}")
        if pgn_files:
            logger.info(f"  python3 move_quality_test.py --pgn {pgn_files[0]}")
            
        logger.info("  python3 move_quality_test.py --tactical 20   # Stockfish-generated positions")
        logger.info("  python3 move_quality_test.py --online 10     # Live Lichess games + downloaded suites")
        
    except ImportError:
        logger.error("❌ 'requests' module required for downloading. Install with: pip install requests")
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
    
    return setup_dir

def fetch_recent_lichess_positions(count: int = 10) -> List[Dict]:
    """Fetch recent positions from real Lichess games"""
    positions = []
    
    try:
        import requests
        import tempfile
        import os
        
        logger.info(f"🌐 Fetching {count} recent positions from Lichess...")
        
        # Get recent games from various strong players
        players = ['hikaru', 'daniil_dubov', 'liem_chess', 'penguingm1', 'grandelius', 'firouzja2003']
        
        for player in players[:3]:  # Limit API calls
            try:
                url = f"https://lichess.org/api/games/user/{player}"
                params = {
                    'max': 3,
                    'rated': 'true',
                    'perfType': 'blitz,rapid,classical',
                    'format': 'pgn'
                }
                
                headers = {'Accept': 'application/x-ndjson'}
                response = requests.get(url, params=params, headers=headers, timeout=15)
                
                if response.status_code == 200 and response.text.strip():
                    # Split by double newline to get individual games
                    games = response.text.strip().split('\n\n\n')
                    
                    for game_text in games[:2]:  # Max 2 games per player
                        if not game_text.strip():
                            continue
                            
                        try:
                            # Parse the game
                            import io
                            game_io = io.StringIO(game_text)
                            game = chess.pgn.read_game(game_io)
                            
                            if game is None:
                                continue
                                
                            board = game.board()
                            move_count = 0
                            
                            # Extract interesting positions from the game
                            for move in game.mainline_moves():
                                move_count += 1
                                board.push(move)
                                
                                # Focus on middlegame positions (moves 10-30)
                                if 10 <= move_count <= 30:
                                    piece_count = len(board.piece_map())
                                    
                                    # Select positions with good piece activity
                                    if piece_count >= 20 and not board.is_game_over():
                                        positions.append({
                                            'name': f'lichess_{player}_move{move_count}',
                                            'fen': board.fen(),
                                            'source': 'lichess_recent',
                                            'player': player,
                                            'move_number': move_count,
                                            'game_info': {
                                                'white': game.headers.get('White', 'Unknown'),
                                                'black': game.headers.get('Black', 'Unknown'),
                                                'result': game.headers.get('Result', '*'),
                                                'event': game.headers.get('Event', 'Lichess'),
                                                'time_control': game.headers.get('TimeControl', 'Unknown')
                                            }
                                        })
                                        
                                        if len(positions) >= count:
                                            break
                                
                                if len(positions) >= count:
                                    break
                                    
                        except Exception as e:
                            logger.debug(f"Error parsing game from {player}: {e}")
                            continue
                            
                time.sleep(1)  # Rate limiting
                
                if len(positions) >= count:
                    break
                    
            except Exception as e:
                logger.warning(f"Error fetching games from {player}: {e}")
                continue
                
    except ImportError:
        logger.warning("requests module required for Lichess API")
    except Exception as e:
        logger.warning(f"Error fetching recent Lichess positions: {e}")
        
    logger.info(f"✅ Fetched {len(positions)} recent Lichess positions")
    return positions

def calculate_strength_score(results_summary: list) -> dict:
    """Calculate a comprehensive strength score from test results"""
    if not results_summary:
        return {"total_score": 0, "breakdown": {}, "grade": "N/A"}
    
    valid_results = [r for r in results_summary if r['status'] == 'OK']
    if not valid_results:
        return {"total_score": 0, "breakdown": {}, "grade": "N/A"}
    
    # Scoring system based on move quality categories
    category_scores = {
        "🏆 EXCELLENT - Optimal or near-optimal move!": 100,
        "🏆 EXCELLENT - Found optimal mate!": 100,
        "🥈 VERY GOOD - Strong move, very close to optimal.": 85,
        "🥉 GOOD - Solid move.": 70,
        "🥈 GOOD - Found a mate, but slower than optimal.": 75,
        "⚠️  OKAY - Playable, but noticeably weaker.": 50,
        "✅ OKAY - Gets mated, but optimally delays or chooses same losing line.": 45,
        "❌ POOR - Significant error.": 25,
        "❌ POOR - Missed a forced mate.": 20,
        "⚠️  POOR - Gets mated faster or avoids best losing line.": 15,
        "💣 BLUNDER - Very serious error.": 5,
        "💣 BLUNDER - Plays into a mate.": 0,
        "❓ UNKNOWN": 30
    }
    
    total_score = 0
    category_breakdown = {}
    
    for result in valid_results:
        category = result.get('category', '❓ UNKNOWN')
        score = category_scores.get(category, 30)
        total_score += score
        category_breakdown[category] = category_breakdown.get(category, 0) + 1
    
    # Calculate average score (0-100 scale)
    average_score = total_score / len(valid_results) if valid_results else 0
    
    # Assign letter grades
    if average_score >= 90:
        grade = "A+"
    elif average_score >= 85:
        grade = "A"
    elif average_score >= 80:
        grade = "A-"
    elif average_score >= 75:
        grade = "B+"
    elif average_score >= 70:
        grade = "B"
    elif average_score >= 65:
        grade = "B-"
    elif average_score >= 60:
        grade = "C+"
    elif average_score >= 55:
        grade = "C"
    elif average_score >= 50:
        grade = "C-"
    elif average_score >= 40:
        grade = "D"
    else:
        grade = "F"
    
    # Calculate ELO estimate (rough approximation)
    # Based on centipawn loss correlation with rating
    non_mate_results = [r for r in valid_results if r.get('eval_diff_cp') is not None]
    avg_centipawn_loss = 0
    if non_mate_results:
        total_cp_loss = sum(max(0, r['eval_diff_cp']) for r in non_mate_results)
        avg_centipawn_loss = total_cp_loss / len(non_mate_results)
    
    # Rough ELO estimate based on average centipawn loss
    # Professional studies show ~50cp loss = ~400 ELO difference from perfect play
    if avg_centipawn_loss <= 20:
        estimated_elo = "2600+"
    elif avg_centipawn_loss <= 35:
        estimated_elo = "2400-2600"
    elif avg_centipawn_loss <= 50:
        estimated_elo = "2200-2400"
    elif avg_centipawn_loss <= 75:
        estimated_elo = "2000-2200"
    elif avg_centipawn_loss <= 100:
        estimated_elo = "1800-2000"
    elif avg_centipawn_loss <= 150:
        estimated_elo = "1600-1800"
    else:
        estimated_elo = "1400-1600"
    
    return {
        "total_score": round(average_score, 1),
        "total_positions": len(valid_results),
        "grade": grade,
        "estimated_elo": estimated_elo,
        "avg_centipawn_loss": round(avg_centipawn_loss, 1),
        "breakdown": category_breakdown,
        "excellence_rate": round(sum(1 for r in valid_results if "EXCELLENT" in r.get('category', '')) / len(valid_results) * 100, 1),
        "blunder_rate": round(sum(1 for r in valid_results if "BLUNDER" in r.get('category', '')) / len(valid_results) * 100, 1)
    }

def save_benchmark_results(score_data: dict, config: dict, timestamp: str = None):
    """Save benchmark results for comparison tracking"""
    if timestamp is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    benchmark_file = Path("engine_benchmarks.json")
    
    # Load existing benchmarks
    benchmarks = []
    if benchmark_file.exists():
        try:
            with open(benchmark_file, 'r') as f:
                benchmarks = json.load(f)
        except:
            benchmarks = []
    
    # Add new benchmark
    new_benchmark = {
        "timestamp": timestamp,
        "engine_version": "2.2.0",  # From chess_engine.py
        "config": config,
        "results": score_data
    }
    
    benchmarks.append(new_benchmark)
    
    # Keep only last 50 benchmarks
    benchmarks = benchmarks[-50:]
    
    # Save updated benchmarks
    with open(benchmark_file, 'w') as f:
        json.dump(benchmarks, f, indent=2)
    
    logger.info(f"💾 Benchmark saved to {benchmark_file}")
    
    # Show improvement comparison if previous benchmarks exist
    if len(benchmarks) > 1:
        prev_benchmark = benchmarks[-2]
        prev_score = prev_benchmark['results']['total_score']
        curr_score = score_data['total_score']
        
        improvement = curr_score - prev_score
        if improvement > 0:
            logger.info(f"📈 Improvement: +{improvement:.1f} points from previous benchmark")
        elif improvement < 0:
            logger.info(f"📉 Regression: {improvement:.1f} points from previous benchmark")
        else:
            logger.info(f"🔄 No change from previous benchmark")

def main():
    logger.info("🚀 Testing Neural Network Move Selection Quality (Comparison by Evaluation Difference)")

    parser = argparse.ArgumentParser(description="Test neural network move selection quality against Stockfish.")
    parser.add_argument("--setup", action="store_true", help="Setup test databases and example files")
    parser.add_argument("--test", type=int, metavar="N", help="Run only a specific test number (e.g., 7 for test7_...).")
    parser.add_argument("--count", type=int, default=25, help="Maximum number of positions to test (default: 25)")
    parser.add_argument("--pgn", action="append", help="PGN file(s) to extract positions from")
    parser.add_argument("--epd", action="append", help="EPD file(s) to load test positions from")
    parser.add_argument("--no-default", action="store_true", help="Skip default hardcoded positions")
    parser.add_argument("--tactical", type=int, default=10, help="Number of tactical positions to generate (default: 10)")
    parser.add_argument("--online", type=int, default=5, help="Number of online positions to fetch (default: 5)")
    parser.add_argument("--sources", help="Comma-separated list of sources to include: default,pgn,epd,tactical,online")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "mixed"], default="mixed", 
                       help="Difficulty level of positions to test")
    parser.add_argument("--pattern", choices=["pin", "fork", "discovery", "endgame", "promotion", "all"], default="all",
                       help="Specific tactical pattern to focus on")
    parser.add_argument("--save-results", help="Save detailed results to JSON file")
    parser.add_argument("--benchmark", action="store_true", help="Run comprehensive benchmark with all position types")
    parser.add_argument("--track-improvement", action="store_true", help="Save results for tracking improvements over time")
    args = parser.parse_args()
    
    # Handle setup command
    if args.setup:
        setup_test_databases()
        return
        
    # Handle benchmark mode
    if args.benchmark:
        logger.info("🏁 Running comprehensive benchmark...")
        args.count = 100
        args.tactical = 25
        args.online = 10
        
    # Initialize position generator
    generator = PositionGenerator()
    
    # Handle specific test selection (legacy behavior)
    if args.test is not None:
        target_test_prefix = f"test{args.test}_"
        default_positions = [
            {"name": "test1_opening_blacks_reply", "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"},
            {"name": "test2_middlegame_tactical", "fen": "r3k2r/pp1b1ppp/2n2n2/1B2N3/Q2Pq3/2P1B3/P4PPP/R3K2R w KQkq - 1 13"},
            {"name": "test4_middlegame_material_imbalance", "fen": "7r/pp1k1p2/2pN1npp/8/8/BP6/P4PnP/2KR4 w - - 0 23"},
            {"name": "test5_endgame_king_pawn", "fen": "8/8/1k6/8/8/8/4P3/3K4 w - - 0 1"},
            {"name": "test6_tactical_pin_black", "fen": "5k2/r2p4/3Np1RP/2PnP3/5P2/1p1N3P/1P1K4/r7 b - - 0 47"},
            {"name": "test7_endgame_promotion_race_black", "fen": "5k2/r2p3P/3Np1R1/2PnP3/5P2/1p1N3P/1P1K4/7r b - - 0 48"},
            {"name": "test8_middlegame_queen_attack_black", "fen": "k3r3/5p2/pqbR4/5Ppp/3B4/1P3P2/1Q4PP/6K1 b - - 2 29"},
            {"name": "test9_kings_indian_attack_white", "fen": "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP3PPP/R1BQKB1R w KQ - 2 6"},
            {"name": "test10_endgame_rook_pawns_white", "fen": "8/5p2/R7/5k2/8/8/P4P2/6K1 w - - 1 36"}
        ]
        
        selected_tests = [p for p in default_positions if p['name'].startswith(target_test_prefix)]
        if not selected_tests:
            logger.error(f"❌ Test number {args.test} (looking for prefix '{target_test_prefix}') not found in defined tests.")
            logger.info("Available test names are:")
            for p in default_positions:
                logger.info(f"  - {p['name']}")
            sys.exit(1)
            
        test_positions_to_run = [selected_tests[0]]
        logger.info(f"🎯 Running only specified test: {test_positions_to_run[0]['name']}")
    else:
        # Use comprehensive position generation
        test_positions_to_run = generator.get_all_positions(
            include_default=not args.no_default,
            pgn_files=args.pgn or [],
            epd_files=args.epd or [],
            tactical_count=args.tactical,
            online_count=args.online,
            max_total=args.count
        )
        
        logger.info(f"\n🎯 Testing Configuration:")
        logger.info(f"   Max positions: {args.count}")
        logger.info(f"   Difficulty: {args.difficulty}")
        logger.info(f"   Pattern focus: {args.pattern}")
        if args.sources:
            logger.info(f"   Sources: {args.sources}")
    
    overall_results_summary = []
    
    for position_data in test_positions_to_run:
        fen = position_data.get('fen')
        name = position_data.get('name', 'unknown_position')
        if fen:
            test_position(fen, name, overall_results_summary)
            # Add metadata to results
            if overall_results_summary:
                overall_results_summary[-1].update({
                    'source': position_data.get('source', 'unknown'),
                    'pattern': position_data.get('pattern', 'unknown'),
                    'game_info': position_data.get('game_info', {}),
                    'operations': position_data.get('operations', {})
                })
    
    # Enhanced Summary with source analysis
    logger.info(f"\n{'='*80}")
    logger.info("📊 FINAL SUMMARY OF NEURAL NETWORK PERFORMANCE")
    logger.info(f"{'='*80}")
    
    if overall_results_summary:
        total_tests = len([r for r in overall_results_summary if r['status'] == 'OK'])
        logger.info(f"Total valid positions tested: {total_tests}")

        # Enhanced category tracking
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
        
        # Source-based analysis
        source_performance = {}
        pattern_performance = {}
        successful_tests = 0
        total_eval_diff_cp = 0
        valid_comparisons = 0

        for r in overall_results_summary:
            if r['status'] == 'OK':
                successful_tests +=1
                if r['category'] in categories_count:
                    categories_count[r['category']] += 1
                    
                # Track performance by source
                source = r.get('source', 'unknown')
                if source not in source_performance:
                    source_performance[source] = {'total': 0, 'excellent': 0, 'good': 0, 'poor': 0}
                source_performance[source]['total'] += 1
                
                # Track performance by pattern
                pattern = r.get('pattern', 'unknown')
                if pattern not in pattern_performance:
                    pattern_performance[pattern] = {'total': 0, 'excellent': 0, 'good': 0, 'poor': 0}
                pattern_performance[pattern]['total'] += 1
                
                if "EXCELLENT" in r['category']:
                    source_performance[source]['excellent'] += 1
                    pattern_performance[pattern]['excellent'] += 1
                elif "GOOD" in r['category'] or "VERY GOOD" in r['category']:
                    source_performance[source]['good'] += 1
                    pattern_performance[pattern]['good'] += 1
                elif "POOR" in r['category'] or "BLUNDER" in r['category']:
                    source_performance[source]['poor'] += 1
                    pattern_performance[pattern]['poor'] += 1
                    
                if r['eval_diff_cp'] is not None and not ("mate" in r['sf_best_eval_display'].lower() or "mate" in r['nn_move_eval_display'].lower()):
                    total_eval_diff_cp += r['eval_diff_cp']
                    valid_comparisons += 1
                
                logger.info(f"  Pos: {r['position'][:30]:<30} | NN: {r['nn_move']:<7} (WPOV Eval: {r['nn_move_cp_wpov']/100.0 if r['nn_move_cp_wpov'] is not None else 'N/A':>8}) | SF: {r['sf_best_move']:<7} (WPOV Eval: {r['sf_best_cp_wpov']/100.0 if r['sf_best_cp_wpov'] is not None else 'N/A':>8}) | Diff: {(str(int(r['eval_diff_cp']))+'cp') if r['eval_diff_cp'] is not None else 'N/A':>7} | Result: {r['category']}")

        if valid_comparisons > 0:
            average_diff_cp = total_eval_diff_cp / valid_comparisons
            logger.info(f"\nAverage Evaluation Difference (SF_best_WPOV - NN_move_WPOV, non-mate positions): {average_diff_cp:.0f} cp")

        logger.info("\nCategory Counts:")
        for cat, count in categories_count.items():
            if count > 0:
                logger.info(f"  {cat}: {count}")

        # Source performance analysis
        if len(source_performance) > 1:
            logger.info("\n📈 Performance by Source:")
            for source, perf in source_performance.items():
                total = perf['total']
                if total > 0:
                    excellent_pct = (perf['excellent'] / total) * 100
                    good_pct = (perf['good'] / total) * 100
                    poor_pct = (perf['poor'] / total) * 100
                    logger.info(f"  {source}: {total} positions - {excellent_pct:.0f}% excellent, {good_pct:.0f}% good, {poor_pct:.0f}% poor")

        # Pattern performance analysis
        if len(pattern_performance) > 1:
            logger.info("\n🎯 Performance by Pattern:")
            for pattern, perf in pattern_performance.items():
                total = perf['total']
                if total > 0 and pattern != 'unknown':
                    excellent_pct = (perf['excellent'] / total) * 100
                    good_pct = (perf['good'] / total) * 100
                    poor_pct = (perf['poor'] / total) * 100
                    logger.info(f"  {pattern}: {total} positions - {excellent_pct:.0f}% excellent, {good_pct:.0f}% good, {poor_pct:.0f}% poor")

        # Overall assessment
        if successful_tests > 0:
            excellent_cats = ["🏆 EXCELLENT - Optimal or near-optimal move!", "🏆 EXCELLENT - Found optimal mate!"]
            good_cats = ["🥈 VERY GOOD - Strong move, very close to optimal.", "🥉 GOOD - Solid move.", "🥈 GOOD - Found a mate, but slower than optimal."]
            
            excellent_ratio = sum(categories_count[cat] for cat in excellent_cats) / successful_tests
            good_ratio = (sum(categories_count[cat] for cat in excellent_cats) + sum(categories_count[cat] for cat in good_cats)) / successful_tests
            
            if excellent_ratio >= 0.6:
                logger.info("\nOverall: 🎯 Outstanding performance! Consistently finding optimal moves.")
            elif excellent_ratio >= 0.4:
                logger.info("\nOverall: 🎯 Commendable performance, often finding top-tier moves.")
            elif good_ratio >= 0.6:
                logger.info("\nOverall: ✅ Solid performance, consistently finding reasonable moves.")
            else:
                logger.info("\nOverall: ⚠️ Room for improvement, struggles with consistency.")

        # Calculate comprehensive strength score
        strength_data = calculate_strength_score(overall_results_summary)
        
        logger.info(f"\n🏆 ENGINE STRENGTH ASSESSMENT")
        logger.info(f"{'='*50}")
        logger.info(f"Overall Score: {strength_data['total_score']}/100 (Grade: {strength_data['grade']})")
        logger.info(f"Estimated Playing Strength: {strength_data['estimated_elo']} ELO")
        logger.info(f"Average Centipawn Loss: {strength_data['avg_centipawn_loss']} cp")
        logger.info(f"Excellence Rate: {strength_data['excellence_rate']}%")
        logger.info(f"Blunder Rate: {strength_data['blunder_rate']}%")
        
        # Save benchmark results if requested
        if args.track_improvement or args.benchmark:
            config_used = {
                'count': args.count,
                'difficulty': args.difficulty,
                'pattern': args.pattern,
                'sources_used': list(source_performance.keys()) if source_performance else [],
                'benchmark_mode': args.benchmark,
                'tactical_count': args.tactical,
                'online_count': args.online
            }
            save_benchmark_results(strength_data, config_used)

        # Save results if requested
        if args.save_results:
            try:
                with open(args.save_results, 'w') as f:
                    json.dump({
                        'test_config': {
                            'count': args.count,
                            'difficulty': args.difficulty,
                            'pattern': args.pattern,
                            'sources_used': list(source_performance.keys()) if source_performance else [],
                            'benchmark_mode': args.benchmark
                        },
                        'strength_assessment': strength_data,
                        'summary': {
                            'total_positions': total_tests,
                            'average_eval_diff_cp': average_diff_cp if valid_comparisons > 0 else None,
                            'category_counts': categories_count,
                            'source_performance': source_performance,
                            'pattern_performance': pattern_performance,
                            'excellent_ratio': excellent_ratio if successful_tests > 0 else 0,
                            'good_ratio': good_ratio if successful_tests > 0 else 0
                        },
                        'detailed_results': overall_results_summary
                    }, f, indent=2)
                logger.info(f"\n💾 Detailed results saved to: {args.save_results}")
            except Exception as e:
                logger.error(f"❌ Failed to save results: {e}")

    logger.info(f"\n✅ Move quality test completed!")

if __name__ == "__main__":
    main()