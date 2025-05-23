# 🤖 Lichess Bot Setup - V4.0 Pure Neural Power

## Prerequisites

1. **Bot Account**: Your `valibot` account must be upgraded to a Bot account
2. **API Token**: You need a Lichess API token with "Play bot moves" permission
3. **Dependencies**: Install required packages

## Setup Steps

### 1. Install Dependencies
```bash
pip install berserk chess python-chess
```

### 2. Get API Token
1. Go to https://lichess.org/account/oauth/token
2. Create a new token with "Play bot moves" permission
3. Copy the token (it will be shown only once!)

### 3. Upgrade Account to Bot (if not done already)
```bash
curl -d '' https://lichess.org/api/bot/account/upgrade -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

### 4. Set Environment Variable
```bash
export LICHESS_TOKEN="your_lichess_api_token_here"
```

### 5. Run the Bot
```bash
python3 lichess_bot.py
```

## Bot Configuration

The bot is configured with these settings:

- **Max Concurrent Games**: 3
- **Supported Time Controls**: Rapid, Blitz, Classical
- **Rating Range**: 800-2800
- **Variants**: Standard chess only
- **Engine**: V4.0 Pure Neural Power with advanced features

## Bot Features

### 🧠 **Advanced Engine**
- Pure neural network strength (no opening book)
- Speed-optimized alpha-beta search (depths 2-3)
- Sophisticated move ordering (MVV-LVA + killers)
- Enhanced transposition tables
- Null move & late move reductions
- Three-fold repetition avoidance

### 🎮 **Game Management**
- Accepts challenges based on configured criteria
- Plays multiple games simultaneously
- Handles all game states (start, moves, finish)
- Automatic resignation when in hopeless positions
- Graceful error handling and recovery

### 💬 **Chat Features**
- Friendly introduction at game start
- Auto-responds to common greetings
- Explains engine capabilities when asked
- Good sportsmanship messages

### 📊 **Logging & Monitoring**
- Comprehensive logging to file and console
- Performance metrics (calculation time, nodes searched)
- Game statistics and results tracking
- Error reporting and debugging info

## Command Line Usage

### Basic Usage
```bash
# Set token and run
export LICHESS_TOKEN="lip_abc123..."
python3 lichess_bot.py
```

### With Custom Settings
You can modify the bot settings by editing `lichess_bot.py`:

```python
# In LichessBot.__init__()
self.max_concurrent_games = 5  # Play more games
self.time_controls = ["bullet", "blitz", "rapid"]  # Add bullet
self.min_rating = 1200  # Higher minimum rating
self.max_rating = 2500  # Lower maximum rating
```

## Monitoring

### Log Files
- **lichess_bot.log**: Detailed bot activity log
- Watch in real-time: `tail -f lichess_bot.log`

### Console Output
The bot provides real-time status updates:
```
2025-01-23 14:30:15 - INFO - 🤖 Initialized Neural Chess Engine V4.0 - Pure Neural Power v4.0
2025-01-23 14:30:15 - INFO - ✅ Connected as valibot (Bot Account)
2025-01-23 14:30:16 - INFO - 🔄 Starting event stream...
2025-01-23 14:30:45 - INFO - ⚔️  Challenge from player123 (rapid, standard, rated)
2025-01-23 14:30:45 - INFO - ✅ Accepted challenge from player123
2025-01-23 14:30:46 - INFO - 🎮 Game started: abc12345 as white vs player123
2025-01-23 14:30:47 - INFO - 🧠 Calculating move for game abc12345 (move 1)
2025-01-23 14:30:48 - INFO - ✅ Played e2e4 in game abc12345 (0.73s)
```

## Safety Features

### Rate Limiting Compliance
- Respects Lichess API rate limits
- Automatic reconnection on stream errors
- Graceful error handling

### Bot Rules Compliance
- No UltraBullet games (¼+0)
- Only challenge games (no pools/tournaments)
- Follows Lichess Terms of Service
- Fair play principles

### Emergency Controls
- **Ctrl+C**: Graceful shutdown (resigns active games)
- **Timeout Handling**: Automatic recovery from network issues
- **Error Recovery**: Continues running despite individual game errors

## Troubleshooting

### Common Issues

**"No module named 'berserk'"**
```bash
pip install berserk
```

**"Token authentication failed"**
- Check your token is correct
- Ensure account is upgraded to Bot
- Verify token has "Play bot moves" permission

**"Engine import error"**
```bash
# Make sure V4 engine is available
python3 -c "from chess_engine_v4 import EngineV4; print('✅ Engine OK')"
```

**Bot not accepting challenges**
- Check rating ranges in configuration
- Verify time control settings
- Look at logs for rejection reasons

### Debug Mode
Enable debug logging by modifying the logging level:
```python
logging.basicConfig(level=logging.DEBUG, ...)
```

## Performance

### Expected Performance
- **Move Calculation**: 0.5-1.5 seconds average
- **Search Depth**: 2-3 (optimized for online play)
- **Concurrent Games**: Up to 3 games simultaneously
- **Memory Usage**: ~100-200MB per game

### Optimization Tips
- Run on a fast machine for better move times
- Monitor CPU usage during peak play
- Consider reducing max concurrent games if slow

## Support

For issues or questions:
1. Check the logs for error messages
2. Verify bot account and token setup
3. Test engine locally first
4. Review Lichess bot API documentation

---

🎯 **Your V4.0 Pure Neural Power engine is ready to dominate Lichess!** 🧠⚡ 