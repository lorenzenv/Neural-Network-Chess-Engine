#!/usr/bin/env python3
"""
PURE NEURAL NETWORK CHESS INFERENCE ENGINE

🚨 CRITICAL PHILOSOPHY: This module contains ZERO chess knowledge. 
If you find yourself adding chess rules, heuristics, or "classical evaluation" - STOP.
The neural network learns patterns from data, not from programmed chess knowledge.

The only exceptions are:
1. Basic chess move legality (to avoid crashes)
2. Terminal position detection (checkmate/stalemate detection only)
3. Minimal material counting ONLY for safety bounds (not evaluation)
"""

import chess
import numpy as np
from util import bitifyFEN, beautifyFEN
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf

# 🛡️ PHILOSOPHY GUARDS: Prevent chess knowledge creep
FORBIDDEN_TERMS = ["classical", "heuristic", "opening", "endgame", "tactics"]

def _check_philosophy_violation(code_text: str):
    """Fail-fast check to prevent chess knowledge from creeping in."""
    for term in FORBIDDEN_TERMS:
        if term.lower() in code_text.lower():
            raise ValueError(f"❌ PHILOSOPHY VIOLATION: Found '{term}' - this implies chess knowledge!")

# 🧠 PURE NN CONFIGURATION
class NeuralNetworkConfig:
    """Configuration for pure NN inference - NO CHESS KNOWLEDGE."""
    
    # NN confidence and reliability settings
    NN_CONFIDENCE_THRESHOLD = 0.9        # How confident must NN be to trust fully?
    NN_RELIABILITY_FACTOR = 0.85         # Overall NN trust factor
    
    # Safety bounds (NOT evaluation heuristics)
    MAX_SAFE_EVALUATION_BOUND = 800      # Prevent NN from giving extreme scores
    MIN_SAFE_EVALUATION_BOUND = -800     # Keep evaluations reasonable
    
    # Terminal position handling (basic chess rules only)
    CHECKMATE_SCORE = 15000              # Score for checkmate detection
    STALEMATE_SCORE = 0                  # Score for stalemate detection
    
    # Search integration settings
    NN_MOVE_ORDERING_CONFIDENCE = 0.6    # When to trust NN move ordering
    
    # 🚨 EMPHASIS ON NN: This engine relies purely on its neural network evaluations.
    MINIMAL_SAFETY_FACTOR = 0.05         # Tiny safety margin for terminal detection only

class NeuralNetworkEvaluator:
    """
    Pure Neural Network Position Evaluator
    
    🚨 ZERO CHESS KNOWLEDGE: This class only calls the neural network.
    All chess understanding comes from the trained model, not code.
    """
    
    def __init__(self, model_path: str = "model.tflite"):
        """Initialize NN inference engine - NO CHESS KNOWLEDGE."""
        try:
            self.interpreter = Interpreter(model_path=model_path)
        except:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
        
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # 💡 Basic Caching
        self.comparison_cache = {}
        self.evaluation_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 🛡️ Philosophy check
        _check_philosophy_violation(str(self.__dict__))
    
    def compare_two_positions(self, fen1: str, fen2: str) -> float:
        """
        PURE NN: Compare two positions using only the neural network.
        Returns 0.0-1.0 where >0.5 means fen1 is better for white than fen2.
        
        🚨 NO CHESS KNOWLEDGE - only NN inference.
        FIXED: Following README logic more closely.
        """
        # 💡 Check cache
        cache_key = (fen1, fen2)
        if cache_key in self.comparison_cache:
            self.cache_hits += 1
            return self.comparison_cache[cache_key]
        
        try:
            # Convert FENs to NN input format - EXACTLY like training
            beautified1 = beautifyFEN(fen1)
            beautified2 = beautifyFEN(fen2)
            bitboard1 = bitifyFEN(beautified1)
            bitboard2 = bitifyFEN(beautified2)
            
            # Prepare NN inputs as separate tensors (matching our TFLite model architecture)
            input_data1 = np.array([bitboard1]).astype(np.float32)
            input_data2 = np.array([bitboard2]).astype(np.float32)
            
            # NN inference with two separate inputs (as our TFLite model expects)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data1)
            self.interpreter.set_tensor(self.input_details[1]['index'], input_data2)
            self.interpreter.invoke()
            result = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            comparison_result = float(result[0][0])
            
            # 💡 Store in cache
            self.comparison_cache[cache_key] = comparison_result
            self.cache_misses += 1
            return comparison_result
            
        except Exception as e:
            print(f"NN inference error: {e}")
            return 0.5  # Neutral if NN fails
    
    def compare_positions_with_symmetry(self, fen1: str, fen2: str) -> float:
        """
        Enhanced NN comparison with symmetry validation for better accuracy.
        Uses multiple NN calls to reduce evaluation noise.
        
        🚨 PURE NN APPROACH - no chess logic, just multiple NN inferences.
        """
        try:
            # Primary comparison
            primary_result = self.compare_two_positions(fen1, fen2)
            
            # Symmetry check: compare in reverse order
            reverse_result = 1.0 - self.compare_two_positions(fen2, fen1)
            
            # Ensemble averaging for stability
            confidence_weight = abs(primary_result - 0.5) * 2.0
            
            if confidence_weight > 0.7:
                # High confidence - trust primary result more
                final_result = 0.8 * primary_result + 0.2 * reverse_result
            else:
                # Lower confidence - average more heavily
                final_result = 0.6 * primary_result + 0.4 * reverse_result
            
            return final_result
            
        except Exception as e:
            print(f"NN symmetry comparison error: {e}")
            return self.compare_two_positions(fen1, fen2)
    
    def evaluate_position_against_reference(self, position_fen: str, reference_fen: str = None) -> float:
        """
        Evaluate a position by comparing against a reference position.
        
        🚨 PURE NN: Uses only starting position as universal reference OR parent FEN.
        No chess knowledge about what makes a "good" reference.
        MODIFIED: Terminal checks (checkmate, stalemate) are commented out.
        """
        # 💡 Check cache (use position_fen and reference_fen as key)
        cache_key = (position_fen, reference_fen) # reference_fen can be None
        if cache_key in self.evaluation_cache:
            self.cache_hits += 1
            return self.evaluation_cache[cache_key]

        _reference_fen = reference_fen
        if _reference_fen is None:
            # Use starting position as universal reference (learned in training)
            # This will be used for the root node of the search.
            _reference_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        try:
            # MODIFIED: Terminal checks removed/commented out
            # board = chess.Board(position_fen)
            # if board.is_checkmate():
            #     return -NeuralNetworkConfig.CHECKMATE_SCORE if board.turn == chess.WHITE else NeuralNetworkConfig.CHECKMATE_SCORE
            # if board.is_stalemate() or board.is_insufficient_material():
            #     return NeuralNetworkConfig.STALEMATE_SCORE
            
            # Pure NN comparison
            comparison_score = self.compare_positions_with_symmetry(position_fen, _reference_fen)
            
            # Convert comparison to evaluation score
            # NN output: >0.5 means position_fen is better for White than _reference_fen
            # Standard chess scores: positive for White's advantage, negative for Black's advantage
            eval_score = (comparison_score - 0.5) * 2.0 * 600  # Scale to centipawn-like range
            
            # Apply safety bounds (NOT chess evaluation)
            eval_score = max(NeuralNetworkConfig.MIN_SAFE_EVALUATION_BOUND, 
                           min(NeuralNetworkConfig.MAX_SAFE_EVALUATION_BOUND, eval_score))
            
            # 💡 Store in cache
            self.evaluation_cache[cache_key] = eval_score
            self.cache_misses += 1
            return eval_score
            
        except Exception as e:
            print(f"NN position evaluation error: {e} for FEN: {position_fen} vs REF: {_reference_fen}")
            return 0.0
    
    def evaluate_moves_for_ordering(self, current_fen: str, candidate_moves: list) -> dict:
        """
        Evaluate candidate moves for ordering using pure NN inference.
        Compares the resulting position of a move (parent) against the child position.
        
        🚨 PURE NN: No chess knowledge about move types, just NN scores.
        FIXED: Compares parent to child with correct scoring interpretation.
        """
        move_scores = {} # Stores {move: score_for_current_player}
        
        if not candidate_moves:
            return move_scores
        
        try:
            board = chess.Board(current_fen)
            is_white_turn = (board.turn == chess.WHITE)
            
            for move in candidate_moves:
                try:
                    board.push(move)
                    resulting_fen = board.fen()
                    board.pop() # Backtrack
                    
                    # FIXED: Compare parent vs child (not child vs parent)
                    # This gives us "is parent better than child" from White's perspective
                    raw_nn_score = self.compare_positions_with_symmetry(current_fen, resulting_fen)
                    
                    ordering_score = 0.5
                    if is_white_turn:
                        # White wants HIGH raw scores (better for White)
                        # Lower ordering score = higher priority
                        ordering_score = 1.0 - raw_nn_score  
                    else:
                        # Black wants LOW raw scores (worse for White = better for Black)
                        # Lower ordering score = higher priority
                        ordering_score = raw_nn_score
                    
                    move_scores[move] = ordering_score

                except Exception as e:
                    print(f"Error evaluating move {move} for ordering (FEN: {current_fen}): {e}")
        
        except Exception as e:
            # This catches errors like invalid current_fen before move iteration.
            print(f"NN move ordering major error for FEN {current_fen}: {e}")
            # Fallback: give all moves neutral score if the initial FEN or board setup fails.
            for move_obj in candidate_moves: # Ensure we are iterating over the original list if board setup failed
                move_scores[move_obj] = 0.5
        
        return move_scores

# 🚨 SAFETY FUNCTIONS: Only for preventing crashes, NOT for evaluation
def minimal_material_count(fen: str) -> float:
    """
    MINIMAL material counting ONLY for safety bounds.
    This is NOT for evaluation - only to prevent illegal moves.
    
    🚨 This should NEVER influence position evaluation.
    """
    try:
        board = chess.Board(fen)
        
        # Basic piece values for safety only
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }
        
        white_material = sum(len(board.pieces(piece_type, chess.WHITE)) * value 
                           for piece_type, value in piece_values.items())
        black_material = sum(len(board.pieces(piece_type, chess.BLACK)) * value 
                           for piece_type, value in piece_values.items())
        
        return (white_material - black_material) * 100  # Convert to centipawn scale
        
    except Exception:
        return 0.0

def detect_terminal_position(fen: str) -> tuple:
    """
    Detect terminal positions using basic chess rules.
    Returns (is_terminal, score) - ONLY for game end detection.
    
    🚨 This is NOT evaluation - only prevents search from continuing illegally.
    MODIFIED: This function now always returns (False, 0.0) as per user request
    to remove terminal condition checks.
    """
    try:
        # board = chess.Board(fen) # Original
        
        # if board.is_checkmate(): # Original
        #     score = -NeuralNetworkConfig.CHECKMATE_SCORE if board.turn == chess.WHITE else NeuralNetworkConfig.CHECKMATE_SCORE # Original
        #     return True, score # Original
        
        # if board.is_stalemate() or board.is_insufficient_material(): # Original
        #     return True, NeuralNetworkConfig.STALEMATE_SCORE # Original
        
        return False, 0.0 # Always return False, 0.0
        
    except Exception: # Original
        return False, 0.0 # Original

# 🛡️ FINAL PHILOSOPHY CHECK
def validate_pure_nn_philosophy():
    """
    Validate that this module maintains pure NN philosophy.
    Call this during initialization to catch philosophy violations.
    """
    import inspect
    
    # Get all functions and classes in this module
    current_module = inspect.getmembers(inspect.getmodule(inspect.currentframe()))
    
    for name, obj in current_module:
        if inspect.isfunction(obj) or inspect.isclass(obj):
            source = inspect.getsource(obj) if hasattr(obj, '__code__') else str(obj)
            _check_philosophy_violation(source)
    
    print("✅ Pure NN philosophy validation passed - no chess knowledge detected!")

if __name__ == "__main__":
    # Test NN inference
    validate_pure_nn_philosophy()
    evaluator = NeuralNetworkEvaluator()
    
    # Test basic comparison
    starting_pos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    
    comparison = evaluator.compare_two_positions(after_e4, starting_pos)
    evaluation = evaluator.evaluate_position_against_reference(after_e4)
    
    print(f"NN Comparison (e4 vs start): {comparison:.3f}")
    print(f"NN Evaluation (e4): {evaluation:.1f}cp")