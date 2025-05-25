# Essential Files for Neural Network Chess Engine

## Core Engine Files
- **`neural_network_inference.py`** - Neural network evaluation engine (FIXED)
- **`search_coordinator.py`** - Alpha-beta search with NN evaluation (FIXED)
- **`util.py`** - Utility functions for FEN processing
- **`model.tflite`** - Trained neural network model

## Bot Interface
- **`lichess_bot.py`** - Lichess bot interface and game management

## Testing & Evaluation
- **`move_quality_test.py`** - Move quality testing suite (KEEP as requested)

## Configuration Files
- **`requirements.txt`** - Python dependencies
- **`Procfile`** - Heroku deployment configuration
- **`runtime.txt`** - Python version specification
- **`.gitignore`** - Git ignore rules

## Documentation
- **`README.md`** - Original project documentation
- **`BUG_FIX_DOCUMENTATION.md`** - Critical bug fix documentation
- **`ESSENTIAL_FILES.md`** - This file

## Result Files (Optional)
- **`baseline_results.json`** - Baseline performance results
- **`move_quality_results.json`** - Move quality test results
- **`engine_benchmarks.json`** - Engine benchmark results
- **`benchmark_positions.json`** - Test positions

## Web Interface (Optional)
- **`static/`** - Static web assets
- **`templates/`** - HTML templates
- **`favicon.ico`** - Website icon

## Removed Files
The following debug and development files have been cleaned up:
- ❌ `debug_blunder.py` - Debug script for investigating blunders
- ❌ `debug_model_format.py` - Model format testing script
- ❌ `optimized_search.py` - Old search implementation
- ❌ `pure_nn_engine.py` - Old engine implementation
- ❌ `philosophy_guard.py` - Development philosophy checker
- ❌ `simple_benchmark.py` - Simple benchmark script
- ❌ `train.py` - Model training script
- ❌ `deepchess.pdf` - Research paper
- ❌ `__pycache__/` - Python cache directory

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Set up Lichess bot token in environment
3. Run bot: `python3 lichess_bot.py`
4. Test move quality: `python3 move_quality_test.py`

## Critical Fix Applied
✅ **Neural network evaluation logic has been fixed** - the engine no longer makes catastrophic blunders due to incorrect position comparison and scoring interpretation. 