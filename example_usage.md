# Enhanced Move Quality Test - Usage Guide

The improved `move_quality_test.py` now supports automatic position generation from multiple sources, making it much more comprehensive without manual position entry.

## Quick Start

### 1. Setup Test Databases
```bash
python move_quality_test.py --setup
```
This creates sample PGN and EPD files with famous games and tactical positions.

### 2. Run Basic Test (Default behavior)
```bash
python move_quality_test.py
```
Tests 25 positions from mixed sources (default + tactical + online).

### 3. Run Comprehensive Benchmark
```bash
python move_quality_test.py --benchmark
```
Tests 100 positions with maximum variety and saves detailed analysis.

## Advanced Usage

### Test Specific Sources

**PGN Games Analysis:**
```bash
python move_quality_test.py --pgn test_databases/famous_games.pgn --count 50
```

**EPD Test Suites:**
```bash
python move_quality_test.py --epd test_databases/tactical_positions.epd --count 30
```

**Multiple Sources:**
```bash
python move_quality_test.py --pgn games1.pgn --pgn games2.pgn --epd tactical.epd --count 75
```

### Focus on Specific Patterns

**Endgame Positions Only:**
```bash
python move_quality_test.py --pattern endgame --tactical 20 --no-default
```

**Tactical Positions:**
```bash
python move_quality_test.py --pattern pin --tactical 15
```

### Difficulty Levels

**Easy Positions:**
```bash
python move_quality_test.py --difficulty easy --count 30
```

**Mixed Difficulty (Default):**
```bash
python move_quality_test.py --difficulty mixed --count 50
```

### Save and Analyze Results

**Save Detailed Results:**
```bash
python move_quality_test.py --benchmark --save-results benchmark_results.json
```

**Legacy Single Test:**
```bash
python move_quality_test.py --test 7  # Tests only position 7
```

## Position Sources

### 1. Default Positions (9 positions)
- Carefully selected opening, middlegame, endgame positions
- Include tactical and strategic themes
- Used as baseline reference

### 2. PGN Game Extraction
- Extracts interesting positions from real games
- Filters for middlegame complexity and tactical richness
- Includes game metadata (players, result, move number)

### 3. EPD Test Suites
- Supports standard EPD format with best moves
- Compatible with popular chess test suites
- Includes position metadata and solutions

### 4. Tactical Position Generator
- **Pins**: Positions with pinned pieces
- **Forks**: Knight and other piece forks
- **Discoveries**: Discovered attacks and checks
- **Endgames**: King+Pawn, Rook endings, basic mates
- **Promotions**: Pawn promotion races and tactics

### 5. Famous Games Database
- Positions from historical games
- Kasparov vs Deep Blue, Fischer vs Spassky, etc.
- Critical decision points in chess history

## Output Analysis

### Performance Categories
- **🏆 EXCELLENT**: Optimal or near-optimal moves (≤10cp loss)
- **🥈 VERY GOOD**: Strong moves (≤30cp loss)  
- **🥉 GOOD**: Solid moves (≤60cp loss)
- **⚠️ OKAY**: Playable but weaker (≤100cp loss)
- **❌ POOR**: Significant errors (≤200cp loss)
- **💣 BLUNDER**: Very serious errors (>200cp loss)

### Enhanced Analytics
- **Source Performance**: How well engine performs on different position types
- **Pattern Performance**: Strength in specific tactical patterns
- **Average Evaluation Difference**: Centipawn loss compared to Stockfish
- **Category Distribution**: Percentage in each performance tier

### JSON Results Format
```json
{
  "test_config": {
    "count": 100,
    "difficulty": "mixed",
    "pattern": "all",
    "sources_used": ["default", "tactical_generator", "famous_games"],
    "benchmark_mode": true
  },
  "summary": {
    "total_positions": 95,
    "average_eval_diff_cp": -12,
    "excellent_ratio": 0.68,
    "good_ratio": 0.84,
    "source_performance": {...},
    "pattern_performance": {...}
  },
  "detailed_results": [...]
}
```

## Creating Custom Test Suites

### EPD Format Example
```
r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - bm Nf3; id "Italian_Game_01"; 
2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - bm Qg6; id "Tactical_Pin_01";
```

### PGN Format
Standard PGN format with game headers and moves. The extractor will automatically identify interesting positions based on:
- Move complexity and piece count
- Game phase (avoiding very early opening)
- Position evaluation changes
- Tactical richness indicators

## Performance Optimization

### Test Size Recommendations
- **Quick Test**: 10-25 positions (1-3 minutes)
- **Standard Test**: 25-50 positions (3-8 minutes)  
- **Comprehensive**: 50-100 positions (8-20 minutes)
- **Full Benchmark**: 100+ positions (20+ minutes)

### Speed vs Coverage Trade-offs
- More positions = better statistical significance
- Diverse sources = broader skill assessment
- Tactical focus = specific weakness identification
- Mixed difficulty = balanced evaluation

## Integration Examples

### Continuous Integration
```bash
# Daily engine testing
python move_quality_test.py --count 30 --save-results daily_$(date +%Y%m%d).json

# Weekly comprehensive benchmark  
python move_quality_test.py --benchmark --save-results weekly_benchmark.json
```

### A/B Testing Engine Versions
```bash
# Test version A
python move_quality_test.py --count 50 --save-results engine_v1.json

# Test version B  
python move_quality_test.py --count 50 --save-results engine_v2.json

# Compare results programmatically
```

This enhanced testing framework provides comprehensive evaluation capabilities while maintaining the simplicity of the original single-command interface. 