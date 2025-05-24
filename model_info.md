# `model.tflite` Information and Usage Guide

This document describes the `model.tflite` neural network model, its expected inputs/outputs, and how to correctly use it for chess position evaluation within a search algorithm, based on insights gained from integrating it into `chess_engine.py`.

## Model Overview

*   **File:** `model.tflite`
*   **Type:** TensorFlow Lite model.

## Inputs

The model expects two primary inputs, both derived from FEN strings:

1.  **Input Tensor 0: Context Position (`fen_context`)**
    *   This is the FEN string of a chess position that provides the "perspective" for the evaluation. Typically, this should be the FEN of the root position of the current search.
    *   Converted to a 1x769 `float32` bitboard.

2.  **Input Tensor 1: Position to Evaluate (`fen_to_evaluate`)**
    *   This is the FEN string of the actual chess position you want to get an evaluation score for.
    *   Converted to a 1x769 `float32` bitboard.

**Bitboard Conversion:**
The FEN strings are converted to bitboards using utility functions, likely similar to `util.beautifyFEN` for normalization and `util.make_bitboard` for the actual conversion, as found in the project. The `make_x_cached(fen_context, fen_to_evaluate)` function in `chess_engine.py` handles this, returning two bitboards.

## Output

The model produces a single `float32` value.

*   **Meaning:** This raw output represents `P(player whose turn it was in fen_context wins | the current board state is fen_to_evaluate)`.
    *   For example, if it was White's turn in `fen_context`, the output is White's probability of winning given the `fen_to_evaluate`.
    *   If it was Black's turn in `fen_context`, the output is Black's probability of winning given the `fen_to_evaluate`.

## Interpretation for an Absolute Evaluation Score (White's POV)

To use this model effectively in a search, the raw output needs to be converted into a scaled, absolute score from White's point of view:

1.  **Obtain Raw Output:**
    *   Let `raw_nn_output` be the float value from `model(bitboard(fen_context), bitboard(fen_to_evaluate))`.

2.  **Center the Score:**
    *   `unscaled_score = float(raw_nn_output) - 0.5`
    *   This centers the score around 0, where positive values favor the player whose turn it was in `fen_context`.

3.  **Scale and Adjust for Perspective:**
    *   Determine the active player in `fen_context` (e.g., `active_player_in_context = fen_context.split()[1]`).
    *   Define a scaling factor (e.g., `SCALING_FACTOR = Config.NN_SCALING_FACTOR` which is `1000.0` in `chess_engine.py`).

    *   If `active_player_in_context == 'w'` (White):
        *   `final_score_white_pov = unscaled_score * SCALING_FACTOR`
    *   Else (if `active_player_in_context == 'b'` (Black)):
        *   The `unscaled_score` is from Black's perspective. To get White's POV, negate it.
        *   `final_score_white_pov = -unscaled_score * SCALING_FACTOR`

This `final_score_white_pov` is the absolute evaluation of `fen_to_evaluate` from White's perspective, conditioned on the game starting (or the search context being) `fen_context`.

## Recommended Usage in a Chess Engine Search Algorithm

The model should be used to provide an *absolute* evaluation of board positions, not deltas.

1.  **Set Context at Search Root:**
    *   When starting a new search for a move (e.g., in an iterative deepening framework for a root board position `R`), store its FEN: `fen_root = R.fen()`.
    *   This `fen_root` will serve as the fixed `fen_context` for **all** neural network evaluations performed during this entire search for the best move from `R`.

2.  **Evaluate Positions During Search:**
    *   Whenever the search algorithm needs to evaluate a specific board position `P_current` (e.g., at a leaf node of the main search, or for the stand-pat score in quiescence search):
        *   Call the neural network with `fen_context = fen_root` and `fen_to_evaluate = P_current.fen()`.
        *   Process the output as described in the "Interpretation" section to get `final_score_white_pov` for `P_current`.

3.  **Use in Search:**
    *   This `final_score_white_pov` is then used directly by the search algorithm (e.g., as the return value for leaf nodes in alpha-beta, or as the stand-pat score in quiescence). It is **not** a delta to be added to a parent node's score.

## Conceptual Python Example:

```python
# Assume:
# interpreter = tflite.Interpreter(model_path="model.tflite") # (loaded)
# input_details, output_details are obtained from the interpreter
# make_x_cached(fen1, fen2) -> (bitboard_for_fen1, bitboard_for_fen2) # As in chess_engine.py
# SCALING_FACTOR = 1000.0 # Example, from Config.NN_SCALING_FACTOR

def get_absolute_score_for_position(fen_to_evaluate: str, fen_context_for_pov: str) -> float:
    # NN expects (context_board, board_to_evaluate)
    x_context_np, x_evaluate_np = make_x_cached(fen_context_for_pov, fen_to_evaluate)
    
    interpreter.set_tensor(input_details[0]['index'], x_context_np) 
    interpreter.set_tensor(input_details[1]['index'], x_evaluate_np)
    interpreter.invoke()
    # raw_evaluation is P(player_to_move_in_context_fen wins | board is now fen_to_evaluate)
    raw_evaluation = interpreter.get_tensor(output_details[0]['index'])[0][0] 
    
    # Score from the perspective of the player whose turn it was in fen_context_for_pov
    unscaled_score_from_context_player_pov = float(raw_evaluation) - 0.5

    active_color_in_context = fen_context_for_pov.split()[1] # e.g., 'w' or 'b'
    
    final_score_wpov = unscaled_score_from_context_player_pov * SCALING_FACTOR
    if active_color_in_context == 'b': # If Black was to move in context_fen, raw_eval was Black's P(win)
        final_score_wpov *= -1 # Negate to get White's POV
        
    return final_score_wpov

# --- Usage in the search algorithm (conceptual) ---

# At the beginning of iterative_deepening_search for a given board:
# initial_fen_at_root = board.fen() 

# When alpha_beta search reaches a leaf node or quiescence search needs a stand-pat score:
# current_node_fen = board.fen() # (after moves to reach this node)
# evaluation_wpov = get_absolute_score_for_position(current_node_fen, initial_fen_at_root)
# ... then use evaluation_wpov in search logic (e.g., return for current player's POV) ...
```

By following this approach, the `model.tflite` can be effectively used to guide the chess engine's search. 