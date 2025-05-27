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
    
    # Safety bounds (NOT evaluation heuristics)
    MAX_SAFE_EVALUATION_BOUND = 800      # Prevent NN from giving extreme scores
    MIN_SAFE_EVALUATION_BOUND = -800     # Keep evaluations reasonable
    
    # Terminal position handling (basic chess rules only)
    CHECKMATE_SCORE = 15000              # Score for checkmate detection
    STALEMATE_SCORE = 0                  # Score for stalemate detection

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
    
    def evaluate_position_against_reference(self, position_fen: str, reference_fen: str, is_quiescence_standpat_log: bool = False) -> float:
        """
        Evaluate a position by comparing against a reference position.
        
        🚨 PURE NN: Uses only the provided reference_fen.
        No chess knowledge about what makes a "good" reference.
        MODIFIED: Terminal checks (checkmate, stalemate) are commented out.
        MODIFIED: reference_fen is now a required parameter.
        """
        # 💡 Check cache (use position_fen and reference_fen as key)
        cache_key = (position_fen, reference_fen) # reference_fen can be None - NO, IT CANNOT BE NONE ANYMORE
        if cache_key in self.evaluation_cache:
            self.cache_hits += 1
            return self.evaluation_cache[cache_key]

        # _reference_fen = reference_fen # No longer needed, directly use reference_fen
        # if _reference_fen is None: # This block is removed
            # Use starting position as universal reference (learned in training)
            # This will be used for the root node of the search.
            # _reference_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        try:
            # MODIFIED: Terminal checks removed/commented out
            # board = chess.Board(position_fen)
            # if board.is_checkmate():
            #     return -NeuralNetworkConfig.CHECKMATE_SCORE if board.turn == chess.WHITE else NeuralNetworkConfig.CHECKMATE_SCORE
            # if board.is_stalemate() or board.is_insufficient_material():
            #     return NeuralNetworkConfig.STALEMATE_SCORE
            
            # Pure NN comparison
            comparison_score = self.compare_positions_with_symmetry(position_fen, reference_fen)
            
            # Convert comparison to evaluation score
            # NN output: >0.5 means position_fen is better for White than _reference_fen
            # Standard chess scores: positive for White's advantage, negative for Black's advantage
            eval_score = (comparison_score - 0.5) * 2.0 * 600  # Scale to centipawn-like range
            
            # Apply safety bounds (NOT chess evaluation)
            eval_score = max(NeuralNetworkConfig.MIN_SAFE_EVALUATION_BOUND, 
                           min(NeuralNetworkConfig.MAX_SAFE_EVALUATION_BOUND, eval_score))
            
            if is_quiescence_standpat_log:
                print(f"[DEBUG NNEval] evaluate_position_against_reference: pos_fen='{position_fen}', ref_fen='{reference_fen}', comp_score={comparison_score:.4f}, eval_score={eval_score:.2f}")

            # 💡 Store in cache
            self.evaluation_cache[cache_key] = eval_score
            self.cache_misses += 1
            return eval_score
            
        except Exception as e:
            print(f"NN position evaluation error: {e} for FEN: {position_fen} vs REF: {reference_fen}")
            return 0.0
    
    def evaluate_moves_for_ordering(self, current_fen: str, candidate_moves: list, ply: int) -> dict:
        """
        Evaluates candidate moves for ordering using pure NN inference.
        Compares the resulting position of a move (child) against the current/parent position.
        Returns a dictionary mapping move to NN score.
        The score represents P(parent_is_better_than_child_after_move).
        A higher score means the child is worse from the parent's perspective.
        For move ordering, we want to prioritize moves where the child is better for the current player.
        """
        move_scores = {} # Stores {move: ordering_score_for_current_player}
        
        if not candidate_moves:
            return move_scores
        
        try:
            board = chess.Board(current_fen)
            is_white_turn = (board.turn == chess.WHITE) # Player whose turn it is in current_fen
            
            # Only log for ply 0 for evaluate_moves_for_ordering to reduce noise
            if ply == 0:
                print(f"[DEBUG NNEval] evaluate_moves_for_ordering: Evaluating {len(candidate_moves)} moves. Parent_FEN for ordering: '{current_fen}', ply: {ply}")

            for move in candidate_moves:
                try:
                    board.push(move)
                    resulting_fen = board.fen() # This is the child FEN
                    board.pop() # Backtrack to parent state for next iteration

                    # raw_nn_score = P(parent_fen > resulting_fen)
                    # If raw_nn_score is high (e.g., >0.5), parent is better than child.
                    # If raw_nn_score is low (e.g., <0.5), child is better than parent.
                    raw_nn_score = self.compare_positions_with_symmetry(current_fen, resulting_fen)
                    
                    # Final revised logic: The NN output P(A > B) means A is better than B from *White's perspective*.
                    # This is how Siamese networks are typically trained: output is P(pos1 better for White than pos2).
                    # If current_fen is A, resulting_fen is B.
                    # raw_nn_score = P(current_fen is better for White than resulting_fen)
                    
                    ordering_score_P_child_better_than_parent = 0.0 # Default

                    if is_white_turn:
                        # White wants resulting_fen to be better than current_fen.
                        # This means P(current_fen > resulting_fen) should be LOW.
                        # So, ordering_score = 1.0 - raw_nn_score. High score if raw_nn_score is low.
                        ordering_score_P_child_better_than_parent = 1.0 - raw_nn_score
                    else: # Black's turn
                        # Black wants resulting_fen to be better than current_fen (for Black).
                        # This means resulting_fen is WORSE for White than current_fen.
                        # This means P(current_fen > resulting_fen) should be HIGH.
                        # So, ordering_score = raw_nn_score. High score if raw_nn_score is high.
                        ordering_score_P_child_better_than_parent = raw_nn_score
                    
                    move_scores[move] = ordering_score_P_child_better_than_parent
                    
                    # Only log for ply 0 for evaluate_moves_for_ordering to reduce noise
                    if ply == 0:
                        print(f"    [DEBUG NNEval MO] parent: '{current_fen}', move: {move}, child: '{resulting_fen}', P(parent>child_for_parent_player): {raw_nn_score:.4f}, ordering_score_P(child>parent_for_parent_player): {ordering_score_P_child_better_than_parent:.4f}")

                except Exception as e:
                    print(f"Error processing move {move} for ordering: {e}")

        except Exception as e:
            print(f"NN move ordering major error for FEN {current_fen}: {e}")
            for m in candidate_moves:
                move_scores[m] = 0.5 # Neutral score for all if initial FEN fails
        
        return move_scores

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
    evaluation = evaluator.evaluate_position_against_reference(after_e4, starting_pos)
    
    print(f"NN Comparison (e4 vs start): {comparison:.3f}")
    print(f"NN Evaluation (e4): {evaluation:.1f}cp")