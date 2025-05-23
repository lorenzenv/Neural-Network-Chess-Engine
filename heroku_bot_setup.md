# 🚀 Heroku Lichess Bot Deployment Guide

## Prerequisites ✅

You already have:
- ✅ Heroku app deployed with V4.0 engine
- ✅ Environment variable `LICHESS_API_TOKEN` set
- ✅ Bot account `valibot` ready

## Deployment Steps

### 1. Test Local Import (Optional)
```bash
python3 test_bot_import.py
```

### 2. Deploy Updated Code
```bash
git add .
git commit -m "🤖 Add Lichess bot with V4.0 Pure Neural Power engine"
git push heroku main
```

### 3. Scale the Bot Process
```bash
# Enable the bot dyno (this will cost additional dyno hours)
heroku ps:scale bot=1

# Check running processes
heroku ps
```

### 4. Monitor Bot Activity
```bash
# Watch bot logs in real-time
heroku logs --tail --ps bot

# Or check all logs
heroku logs --tail
```

## Heroku Configuration

### Current Setup
- **Web Process**: `web: gunicorn app:app` (your chess web app)
- **Bot Process**: `bot: python lichess_bot.py` (Lichess bot)
- **Environment**: `LICHESS_API_TOKEN` variable set

### Dyno Usage
- **Free Tier**: 550 hours/month total
- **Web + Bot**: Uses ~1,488 hours/month (62 days)
- **Recommendation**: Upgrade to Hobby dyno ($7/month) for 24/7 operation

## Bot Features on Heroku

### ⚡ **Optimized for Cloud**
- Automatic reconnection on network issues
- Graceful error handling and recovery
- Comprehensive logging to Heroku logs
- Efficient memory usage (~100-200MB)

### 🎮 **Game Management**
- Accepts challenges automatically based on criteria
- Plays up to 3 concurrent games
- Supports Rapid, Blitz, and Classical time controls
- Rating range: 800-2800

### 🧠 **V4.0 Pure Neural Power**
- Same advanced engine as your web app
- Optimized search depths (2-3) for online play
- Pure neural network evaluation
- Advanced pruning and move ordering

## Monitoring Commands

### Check Bot Status
```bash
# See if bot is running
heroku ps

# Check bot logs
heroku logs --ps bot --tail

# Check recent bot activity
heroku logs --ps bot -n 100
```

### Bot Controls
```bash
# Start bot
heroku ps:scale bot=1

# Stop bot  
heroku ps:scale bot=0

# Restart bot
heroku restart bot
```

### Environment Variables
```bash
# Check current variables
heroku config

# Update token if needed
heroku config:set LICHESS_API_TOKEN=your_new_token_here
```

## Expected Log Output

When the bot starts successfully, you'll see:
```
2025-01-23T15:30:15.000000+00:00 app[bot.1]: 🤖 Initialized Neural Chess Engine V4.0 - Pure Neural Power v4.0
2025-01-23T15:30:15.000000+00:00 app[bot.1]: ✅ Connected as valibot (Bot Account)
2025-01-23T15:30:16.000000+00:00 app[bot.1]: 🔄 Starting event stream...
```

When someone challenges the bot:
```
2025-01-23T15:35:45.000000+00:00 app[bot.1]: ⚔️  Challenge from player123 (rapid, standard, rated)
2025-01-23T15:35:45.000000+00:00 app[bot.1]: ✅ Accepted challenge from player123
2025-01-23T15:35:46.000000+00:00 app[bot.1]: 🎮 Game started: abc12345 as white vs player123
2025-01-23T15:35:47.000000+00:00 app[bot.1]: ✅ Played e2e4 in game abc12345 (0.73s)
```

## Troubleshooting

### Bot Not Starting
```bash
# Check for errors
heroku logs --ps bot -n 50

# Test imports
heroku run python test_bot_import.py
```

### Common Issues

**"Token authentication failed"**
- Verify token: `heroku config:get LICHESS_API_TOKEN`
- Ensure bot account is properly upgraded

**"Import errors"**
- Check if all dependencies are in requirements.txt
- Verify engine files deployed correctly

**"Bot not accepting challenges"**
- Check bot logs for rejection reasons
- Verify rating/time control settings match your preferences

### Memory Issues
If you get memory errors:
```bash
# Check memory usage
heroku logs --ps bot | grep "Memory"

# Consider reducing concurrent games in lichess_bot.py:
# self.max_concurrent_games = 1  # Reduce from 3 to 1
```

## Cost Optimization

### Free Tier Limitations
- **Total**: 550 dyno hours/month
- **Web Only**: ~744 hours needed for 24/7
- **Web + Bot**: ~1,488 hours needed

### Recommendations
1. **Hobby Dyno ($7/month)**: Unlimited hours, sleeps never
2. **Scheduled Bot**: Run bot only during peak hours
3. **Single Process**: Choose either web app OR bot for free tier

### Scheduling Bot (Free Tier)
```bash
# Run bot only during peak hours (example: 6 PM - 11 PM UTC)
# Stop at night to save dyno hours
heroku ps:scale bot=0  # Stop bot
heroku ps:scale bot=1  # Start bot
```

## Performance Monitoring

### Key Metrics to Watch
- **Response Time**: Bot move calculation time (<2 seconds ideal)
- **Memory Usage**: Should stay under 512MB
- **Error Rate**: Monitor for frequent disconnections
- **Game Acceptance**: Check challenge acceptance/rejection ratios

### Optimization Tips
- Monitor peak usage times
- Adjust concurrent game limits based on performance
- Consider time control preferences vs performance trade-offs

---

## Quick Start Commands

```bash
# Deploy and start
git push heroku main
heroku ps:scale bot=1

# Monitor
heroku logs --tail --ps bot

# Stop
heroku ps:scale bot=0
```

🎯 **Your V4.0 Pure Neural Power bot is ready to dominate Lichess from the cloud!** ⚡🤖 