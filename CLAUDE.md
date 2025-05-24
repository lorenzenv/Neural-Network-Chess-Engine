# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Testing & Quality Assurance
```bash
# Run move quality tests comparing NN engine to Stockfish
python3 move_quality_test.py --count 10 --tactical 5

# Test specific positions (use FEN strings)
python3 move_quality_test.py --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Run comprehensive test suite
python3 move_quality_test.py --count 20 --tactical 10
```

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run web interface locally
python3 app.py

# Run Lichess bot
python3 lichess_bot.py
```

### Model Training (Advanced)
```bash
# Train new neural network model
python3 train.py
```

## Architecture Overview

This is a **pure neural network chess engine** that deliberately minimizes traditional chess knowledge in favor of learned patterns. The core philosophy is to let the neural network handle position evaluation rather than hard-coding chess heuristics.

### Core Components

**Chess Engine (`chess_engine.py`)**
- Hybrid search combining neural network evaluation with alpha-beta search
- Three speed modes: "fast" (3s), "balanced" (5s), "strong" (15s) 
- Configurable NN vs classical evaluation blending via `Config.NN_EVALUATION_BLEND`
- Training-informed ensemble methods for improved color balance
- Transposition tables, killer moves, and quiescence search

**Neural Network Model (`model.tflite`)**
- TensorFlow Lite comparison model trained on 2M chess positions
- Takes two 769-bit position representations and outputs comparison score (0-1)
- NOT an absolute position evaluator - designed to compare positions
- Uses ensemble methods and symmetry validation for better accuracy

**Position Encoding (`util.py`)**
- `beautifyFEN()`: Converts FEN to 65-element array (64 squares + turn)
- `bitifyFEN()`: Converts to 769-bit representation (12 piece types × 64 squares + turn)
- Critical for neural network input - any bugs here break NN evaluation

**Web Interface (`app.py`)**
- Flask web app for human vs engine play
- JSON API endpoint `/get_move` accepts FEN, returns UCI move
- Serves static chessboard UI at root `/`

**Lichess Integration (`lichess_bot.py`)**
- Automated bot for online play on Lichess
- Handles game state, time controls, and API communication
- Uses the core engine with appropriate time settings

### Key Design Principles

1. **Pure NN Philosophy**: Minimize hard-coded chess knowledge, let NN learn patterns
2. **Comparison-Based Evaluation**: NN compares positions rather than providing absolute scores
3. **Training-Informed Methods**: Ensemble techniques and reference positions based on training insights
4. **Conservative Blending**: Classical evaluation as safety net to prevent blunders
5. **Context-Aware**: Game history and reference positions improve NN accuracy

### Speed Mode Configuration

Located in `chess_engine.py` at `Config.SPEED_MODE`:
- **fast**: 3s/move, depth 6, 95% NN blend - for online play
- **balanced**: 5s/move, depth 6, 90% NN blend - default development
- **strong**: 15s/move, depth 8, 85% NN blend - for analysis

### Neural Network Evaluation Methods

The engine includes multiple NN evaluation approaches:
- `evaluate_absolute_score_final_optimized()`: Production method with conservative scaling
- `evaluate_move_comparison_training_informed()`: Advanced move ordering with strategic references  
- `compare_positions_ensemble()`: Symmetry-validated comparison for color balance

## Testing Framework

**Move Quality Tests (`move_quality_test.py`)**
- Compares engine moves to Stockfish recommendations
- Measures evaluation differences in centipawns
- Supports tactical position testing and PGN game analysis
- Tracks move quality statistics and improvement over time

**Test Databases (`test_databases/`)**
- Collection of test positions and game files
- Used for regression testing and development validation

## Important Notes

- The engine is designed as a **comparison model**, not absolute evaluator
- Recent v2.2.0 fixes corrected critical bitboard conversion bugs in `util.py`
- Neural network works optimally when comparing meaningful positions
- Avoid adding traditional chess knowledge - let the NN learn patterns
- Always test changes with move quality tests before deployment