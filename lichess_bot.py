#!/usr/bin/env python3
"""
🤖 LICHESS BOT - V4.1 ENHANCED STRENGTH
Advanced chess bot for Lichess using the V4.1 Enhanced Strength engine
"""

import os
import sys
import time
import threading
import logging
from typing import Dict, Optional, Any
import chess
import chess.pgn
import berserk
from chess_engine import Engine, ENGINE_VERSION, ENGINE_NAME, ENGINE_FEATURES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lichess_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class LichessBot:
    """Advanced Lichess bot using V4.1 Enhanced Strength engine"""
    
    def __init__(self, token: str):
        """Initialize the Lichess bot"""
        self.token = token
        self.session = berserk.TokenSession(token)
        self.client = berserk.Client(session=self.session)
        
        # Bot configuration
        self.bot_name = "valibot"
        self.engine_version = ENGINE_VERSION
        self.engine_name = ENGINE_NAME
        
        # Game management
        self.active_games: Dict[str, Dict] = {}
        self.game_threads: Dict[str, threading.Thread] = {}
        
        # Bot settings
        self.max_concurrent_games = 3
        self.time_controls = ["rapid", "blitz", "classical"]  # Supported time controls
        self.min_rating = 800   # Minimum opponent rating
        self.max_rating = 2800  # Maximum opponent rating
        
        logger.info(f"🤖 Initialized {self.engine_name} v{self.engine_version}")
        logger.info(f"✨ Features: {', '.join(ENGINE_FEATURES)}")
    
    def start(self):
        """Start the bot and begin listening for events"""
        try:
            # Verify bot account
            account = self.client.account.get()
            logger.info(f"✅ Connected as {account['username']} (Bot Account)")
            
            # Stream incoming events
            logger.info("🔄 Starting event stream...")
            self.stream_events()
            
        except Exception as e:
            logger.error(f"❌ Failed to start bot: {e}")
            raise
    
    def stream_events(self):
        """Stream and handle incoming Lichess events"""
        try:
            for event in self.client.bots.stream_incoming_events():
                event_type = event.get('type')
                
                if event_type == 'challenge':
                    self.handle_challenge(event['challenge'])
                elif event_type == 'gameStart':
                    self.handle_game_start(event['game'])
                elif event_type == 'gameFinish':
                    self.handle_game_finish(event['game'])
                elif event_type == 'challengeCanceled':
                    logger.info(f"🚫 Challenge canceled: {event['challenge']['id']}")
                elif event_type == 'challengeDeclined':
                    logger.info(f"❌ Challenge declined: {event['challenge']['id']}")
                else:
                    logger.debug(f"🔔 Unknown event: {event_type}")
                    
        except Exception as e:
            logger.error(f"💥 Event stream error: {e}")
            time.sleep(5)  # Wait before reconnecting
            self.stream_events()
    
    def handle_challenge(self, challenge: Dict[str, Any]):
        """Handle incoming challenge"""
        challenge_id = challenge['id']
        challenger = challenge['challenger']
        variant = challenge['variant']['key']
        speed = challenge['speed']
        rated = challenge['rated']
        
        logger.info(f"⚔️  Challenge from {challenger['name']} ({speed}, {variant}, {'rated' if rated else 'casual'})")
        
        # Decision logic
        should_accept = self.should_accept_challenge(challenge)
        
        if should_accept:
            try:
                self.client.bots.accept_challenge(challenge_id)
                logger.info(f"✅ Accepted challenge from {challenger['name']}")
            except Exception as e:
                logger.error(f"❌ Failed to accept challenge: {e}")
        else:
            try:
                self.client.bots.decline_challenge(challenge_id)
                logger.info(f"🚫 Declined challenge from {challenger['name']}")
            except Exception as e:
                logger.error(f"❌ Failed to decline challenge: {e}")
    
    def should_accept_challenge(self, challenge: Dict[str, Any]) -> bool:
        """Determine whether to accept a challenge"""
        # Check if we have too many active games
        if len(self.active_games) >= self.max_concurrent_games:
            logger.info(f"🎮 Too many active games ({len(self.active_games)}/{self.max_concurrent_games})")
            return False
        
        # Check variant (only standard chess)
        if challenge['variant']['key'] != 'standard':
            logger.info(f"🚫 Unsupported variant: {challenge['variant']['key']}")
            return False
        
        # Check time control
        speed = challenge['speed']
        if speed not in self.time_controls:
            logger.info(f"⏱️  Unsupported time control: {speed}")
            return False
        
        # Accept all rated games within reasonable rating range for better ELO calibration
        rated = challenge['rated']
        if rated:
            logger.info(f"✅ Accepting rated game for ELO calibration")
        
        # Check opponent rating - be more lenient for rated games
        challenger = challenge['challenger']
        if 'rating' in challenger:
            rating = challenger['rating']
            # For rated games, accept wider rating range to get better ELO assessment
            min_rating = 600 if rated else self.min_rating
            max_rating = 3200 if rated else self.max_rating
            
            if rating < min_rating or rating > max_rating:
                logger.info(f"📊 Rating out of range: {rating} (accepted: {min_rating}-{max_rating})")
                return False
        
        # Fixed time control parsing - don't check for UltraBullet if speed is already classified as blitz/rapid/classical
        time_control = challenge.get('timeControl', {})
        logger.info(f"🔍 Time control debug: {time_control}, speed: {speed}")
        
        # Only reject actual UltraBullet (speed will be 'ultraBullet' from Lichess)
        if speed == 'ultraBullet':
            logger.info(f"⚡ UltraBullet not supported")
            return False
        
        # Additional check for very fast time controls that might slip through
        if time_control.get('type') == 'clock':
            # Lichess API returns time in seconds, not milliseconds
            initial_time = time_control.get('limit', 0)
            increment = time_control.get('increment', 0)
            
            logger.info(f"⏰ Time details: {initial_time}s + {increment}s increment")
            
            # Only reject if it's truly under 30 seconds total time
            if initial_time < 15 and increment == 0:  # Very conservative check
                logger.info(f"⚡ Very fast time control rejected: {initial_time}s")
                return False
        
        return True
    
    def handle_game_start(self, game: Dict[str, Any]):
        """Handle the start of a new game"""
        game_id = game['gameId']
        color = game['color']
        opponent = game.get('opponent', {}).get('username', 'Unknown')
        
        logger.info(f"🎮 Game started: {game_id} as {color} vs {opponent}")
        
        # Store game info
        self.active_games[game_id] = {
            'color': color,
            'opponent': opponent,
            'board': chess.Board(),
            'engine': None,
            'move_count': 0
        }
        
        # Start game thread
        game_thread = threading.Thread(target=self.play_game, args=(game_id,))
        game_thread.daemon = True
        game_thread.start()
        self.game_threads[game_id] = game_thread
    
    def handle_game_finish(self, game: Dict[str, Any]):
        """Handle the end of a game"""
        game_id = game['gameId']
        
        if game_id in self.active_games:
            game_info = self.active_games[game_id]
            opponent = game_info['opponent']
            result = game.get('status', 'finished')
            
            logger.info(f"🏁 Game finished: {game_id} vs {opponent} ({result})")
            
            # Clean up
            del self.active_games[game_id]
            if game_id in self.game_threads:
                del self.game_threads[game_id]
    
    def play_game(self, game_id: str):
        """Main game loop for a specific game"""
        try:
            logger.info(f"🧠 Starting neural engine for game {game_id}")
            
            # Stream game state
            for event in self.client.bots.stream_game_state(game_id):
                if event['type'] == 'gameFull':
                    self.handle_game_full(game_id, event)
                elif event['type'] == 'gameState':
                    self.handle_game_state(game_id, event)
                elif event['type'] == 'chatLine':
                    self.handle_chat(game_id, event)
                elif event['type'] == 'opponentGone':
                    logger.info(f"👻 Opponent gone in game {game_id}")
                    
        except Exception as e:
            logger.error(f"💥 Game error {game_id}: {e}")
        finally:
            # Clean up if game ended unexpectedly
            if game_id in self.active_games:
                del self.active_games[game_id]
            if game_id in self.game_threads:
                del self.game_threads[game_id]
    
    def handle_game_full(self, game_id: str, event: Dict[str, Any]):
        """Handle full game data"""
        if game_id not in self.active_games:
            return
        
        game_info = self.active_games[game_id]
        
        # Initialize board with starting position or FEN
        initial_fen = event.get('initialFen', 'startpos')
        if initial_fen == 'startpos':
            game_info['board'] = chess.Board()
        else:
            game_info['board'] = chess.Board(initial_fen)
        
        # Process initial moves
        state = event.get('state', {})
        moves = state.get('moves', '')
        if moves:
            self.apply_moves(game_id, moves)
        
        # Check if it's our turn
        self.check_and_make_move(game_id, state)
    
    def handle_game_state(self, game_id: str, event: Dict[str, Any]):
        """Handle game state updates"""
        if game_id not in self.active_games:
            return
        
        moves = event.get('moves', '')
        self.apply_moves(game_id, moves)
        self.check_and_make_move(game_id, event)
    
    def apply_moves(self, game_id: str, moves_str: str):
        """Apply moves to the board"""
        if not moves_str or game_id not in self.active_games:
            return
        
        game_info = self.active_games[game_id]
        board = game_info['board']
        
        # Reset to starting position
        initial_fen = board.starting_fen if hasattr(board, 'starting_fen') else chess.STARTING_FEN
        board.set_fen(initial_fen)
        
        # Apply all moves
        moves = moves_str.split()
        for move_str in moves:
            try:
                move = chess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    logger.error(f"❌ Illegal move in game {game_id}: {move_str}")
                    break
            except Exception as e:
                logger.error(f"💥 Move parsing error in game {game_id}: {e}")
                break
        
        game_info['move_count'] = len(moves)
    
    def check_and_make_move(self, game_id: str, state: Dict[str, Any]):
        """Check if it's our turn and make a move"""
        if game_id not in self.active_games:
            return
        
        game_info = self.active_games[game_id]
        board = game_info['board']
        our_color = game_info['color']
        
        # Check game status
        status = state.get('status')
        if status not in ['started', 'created']:
            logger.info(f"🏁 Game {game_id} ended with status: {status}")
            return
        
        # Check if it's our turn
        our_turn = (our_color == 'white' and board.turn == chess.WHITE) or \
                   (our_color == 'black' and board.turn == chess.BLACK)
        
        if not our_turn:
            return
        
        # Check for game ending conditions
        if board.is_game_over():
            logger.info(f"🏁 Game {game_id} is over: {board.result()}")
            return
        
        # Make our move
        self.make_move(game_id)
    
    def make_move(self, game_id: str):
        """Generate and make a move using the V4.1 Enhanced Strength engine"""
        if game_id not in self.active_games:
            return
        
        try:
            game_info = self.active_games[game_id]
            board = game_info['board']
            
            logger.info(f"🧠 Calculating move for game {game_id} (move {game_info['move_count'] + 1})")
            
            # Initialize engine for this position
            if game_info['engine'] is None:
                # Pass the bot's color to the engine
                # The new Engine class determines color from FEN, bot_color parameter is not used by Engine itself
                # but lichess_bot uses game_info['color'] to know its own playing color.
                game_info['engine'] = Engine(board.fen())
            else:
                # Update engine with current position
                game_info['engine'].board.set_fen(board.fen())
            
            # Get move from V4.1 Enhanced Strength engine
            start_time = time.time()
            move_str = game_info['engine'].get_move()
            calc_time = time.time() - start_time
            
            if move_str == "checkmate":
                logger.info(f"😵 No legal moves in game {game_id} - resigning")
                self.client.bots.resign_game(game_id)
                return
            
            # Validate move
            try:
                move = chess.Move.from_uci(move_str)
                if move not in board.legal_moves:
                    logger.error(f"❌ Illegal move generated: {move_str}")
                    # Fallback to first legal move
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        move = legal_moves[0]
                        move_str = move.uci()
                    else:
                        self.client.bots.resign_game(game_id)
                        return
            except Exception as e:
                logger.error(f"💥 Move parsing error: {e}")
                return
            
            # Make the move on Lichess
            self.client.bots.make_move(game_id, move_str)
            
            logger.info(f"✅ Played {move_str} in game {game_id} ({calc_time:.2f}s)")
            
            # Send a friendly message occasionally
            if game_info['move_count'] == 0:  # First move
                self.send_intro_message(game_id)
            
        except Exception as e:
            logger.error(f"💥 Move generation error in game {game_id}: {e}")
    
    def handle_chat(self, game_id: str, event: Dict[str, Any]):
        """Handle chat messages"""
        username = event.get('username')
        text = event.get('text', '').lower()
        room = event.get('room', 'player')
        
        if username == self.bot_name:
            return  # Ignore our own messages
        
        logger.info(f"💬 Chat in {game_id} ({room}): {username}: {text}")
        
        # Auto-respond to certain messages
        if any(word in text for word in ['hi', 'hello', 'hey', 'good luck']):
            self.send_chat_message(game_id, "Good luck and have fun! 🤖", room)
        elif any(word in text for word in ['gg', 'good game', 'well played']):
            self.send_chat_message(game_id, "Good game! Thanks for playing! 🎯", room)
        elif 'engine' in text or 'bot' in text:
            self.send_chat_message(game_id, f"I'm running {ENGINE_NAME} v{ENGINE_VERSION} with pure neural network strength! 🧠", room)
    
    def send_intro_message(self, game_id: str):
        """Send introduction message at game start"""
        intro = f"Hello! I'm {ENGINE_NAME} v{ENGINE_VERSION} 🤖 Good luck!"
        self.send_chat_message(game_id, intro, "player")
    
    def send_chat_message(self, game_id: str, message: str, room: str = "player"):
        """Send a chat message"""
        try:
            self.client.bots.post_message(game_id, message, room)
            logger.debug(f"💬 Sent message to {game_id} ({room}): {message}")
        except Exception as e:
            logger.error(f"❌ Failed to send message: {e}")
    
    def stop(self):
        """Stop the bot gracefully"""
        logger.info("🛑 Stopping bot...")
        
        # Resign all active games
        for game_id in list(self.active_games.keys()):
            try:
                self.client.bots.resign_game(game_id)
                logger.info(f"🏳️  Resigned game {game_id}")
            except Exception as e:
                logger.error(f"❌ Failed to resign game {game_id}: {e}")
        
        logger.info("👋 Bot stopped")


def main():
    """Main function to run the Lichess bot"""
    # Get API token from environment variable (check both possible names)
    token = os.getenv('LICHESS_TOKEN') or os.getenv('LICHESS_API_TOKEN')
    if not token:
        logger.error("❌ Please set LICHESS_TOKEN or LICHESS_API_TOKEN environment variable")
        sys.exit(1)
    
    # Create and start bot
    bot = LichessBot(token)
    
    try:
        logger.info("🚀 Starting V4.1 Enhanced Strength Lichess Bot...")
        bot.start()
    except KeyboardInterrupt:
        logger.info("⏹️  Received interrupt signal")
    except Exception as e:
        logger.error(f"💥 Bot crashed: {e}")
    finally:
        bot.stop()


if __name__ == "__main__":
    main() 