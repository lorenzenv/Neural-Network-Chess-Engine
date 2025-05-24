import chess
import numpy as np
import functools
import time
import threading
from util import * # Changed from specific imports to wildcard

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

from util import *  # or your specific imports

# global model loading with thread safety
interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Thread lock for interpreter access
interpreter_lock = threading.Lock()

# ------- Engine Metadata -------
ENGINE_NAME = "valibot"
ENGINE_VERSION = "2.3.0"
ENGINE_FEATURES = [
    "Enhanced Neural Network Evaluation", 
    "Advanced Move Ordering", 
    "Aspiration Windows",
    "Adaptive Late Move Reduction",
    "Enhanced Transposition Table",
    "Intelligent Time Management",
    "Pure NN Philosophy"
]

# ------- Configuration -------
class Config:
    # Speed Mode: "fast", "balanced", "strong"
    SPEED_MODE = "balanced"  # Upgraded from fast for better NN utilization
    
    NN_SCALING_FACTOR = 1000.0
    MAX_PLY_FOR_KILLERS = 30
    QUIESCENCE_MAX_DEPTH_RELATIVE = 6  # Increased for better tactical vision
    LMR_MIN_MOVES_TRIED = 4 # Slightly more conservative LMR for NN
    LMR_REDUCTION = 1
    NMP_REDUCTION = 3
    TT_SIZE_POWER = 20
    FEN_CACHE_SIZE = 4096  # Increased for better NN caching
    
    # Enhanced settings for pure NN approach
    if SPEED_MODE == "fast":
        MAX_SEARCH_DEPTH_ID = 8
        ITERATIVE_DEEPENING_TIME_LIMIT_PER_MOVE = 5.0  # Increased for NN
        NN_EVALUATION_BLEND = 0.6  # Conservative blending
        SEARCH_TIME_EXTENSION_FACTOR = 2.0
        NN_MOVE_ORDERING_MAX_PLY = 4  # Expand NN usage
        NN_MOVE_ORDERING_MAX_MOVES = 30
    elif SPEED_MODE == "balanced":
        MAX_SEARCH_DEPTH_ID = 10  # Increased for better NN utilization
        ITERATIVE_DEEPENING_TIME_LIMIT_PER_MOVE = 8.0
        NN_EVALUATION_BLEND = 0.55  # Conservative blending
        SEARCH_TIME_EXTENSION_FACTOR = 2.5
        NN_MOVE_ORDERING_MAX_PLY = 6  # More NN usage
        NN_MOVE_ORDERING_MAX_MOVES = 35
    else:  # "strong"
        MAX_SEARCH_DEPTH_ID = 12
        ITERATIVE_DEEPENING_TIME_LIMIT_PER_MOVE = 15.0
        NN_EVALUATION_BLEND = 0.5  # Conservative blending
        SEARCH_TIME_EXTENSION_FACTOR = 3.0
        NN_MOVE_ORDERING_MAX_PLY = 8  # Deep NN usage
        NN_MOVE_ORDERING_MAX_MOVES = 40

# ------- Piece values for MVV-LVA -------
PIECE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 320,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000
}

# ------- Zobrist Hashing -------
class ZobristHash:
    def __init__(self):
        # np.random.seed(42) # For deterministic hashes during testing
        self.piece_keys = np.random.randint(0, np.iinfo(np.uint64).max, size=(12, 64), dtype=np.uint64) # 12 piece types, 64 squares
        self.black_to_move_key = np.random.randint(0, np.iinfo(np.uint64).max, dtype=np.uint64)
        self.castling_keys = np.random.randint(0, np.iinfo(np.uint64).max, size=16, dtype=np.uint64) # For 2^4 castling rights combinations
        self.ep_keys = np.random.randint(0, np.iinfo(np.uint64).max, size=64, dtype=np.uint64) # For EP square

    def get_piece_id(self, piece: chess.Piece) -> int:
        # P=0, N=1, B=2, R=3, Q=4, K=5 (for white)
        # P=6, N=7, B=8, R=9, Q=10, K=11 (for black)
        piece_id = piece.piece_type - 1 # 0-5
        if piece.color == chess.BLACK:
            piece_id += 6
        return piece_id

    def hash(self, board: chess.Board) -> np.uint64:
        h = np.uint64(0)
        for sq_idx in range(64):
            piece = board.piece_at(sq_idx)
            if piece:
                h ^= self.piece_keys[self.get_piece_id(piece), sq_idx]
        
        if board.turn == chess.BLACK:
            h ^= self.black_to_move_key
        
        h ^= self.castling_keys[board.castling_rights & 15] # Use lower 4 bits for castling rights

        if board.ep_square is not None:
            h ^= self.ep_keys[board.ep_square]
            
        return h

# ------- Enhanced Transposition Table -------
class EnhancedTT:
    def __init__(self):
        size = 1 << Config.TT_SIZE_POWER
        self.mask = np.uint64(size - 1)
        self.keys = np.zeros(size, dtype=np.uint64)
        self.entries = [None] * size
        self.generation = 0  # For age-based replacement
        self.hits = 0
        self.misses = 0
    
    def index(self, key: np.uint64) -> int:
        return int(key & self.mask)

    def store(self, key: np.uint64, entry: tuple): # depth, score_wpov, best_move_uci, flag, generation
        i = self.index(key)
        
        # Enhanced replacement strategy: prefer deeper searches and newer generations
        should_replace = True
        if self.entries[i] is not None and self.keys[i] != key:
            existing_depth = self.entries[i][0]
            existing_generation = self.entries[i][4] if len(self.entries[i]) > 4 else 0
            new_depth = entry[0]
            
            # Keep deeper searches, but also prefer newer generations
            generation_bonus = (self.generation - existing_generation) * 2
            if existing_depth > new_depth + generation_bonus:
                should_replace = False
        
        if should_replace:
            # Add generation to entry
            enhanced_entry = entry + (self.generation,)
            self.keys[i] = key 
            self.entries[i] = enhanced_entry

    def lookup(self, key: np.uint64):
        i = self.index(key)
        if self.keys[i] == key:
            self.hits += 1
            return self.entries[i]
        self.misses += 1
        return None
    
    def new_search(self):
        """Call at the start of each new search to age entries"""
        self.generation += 1
        
    def get_stats(self) -> dict:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'generation': self.generation
        }

# Maintain backward compatibility
FixedTT = EnhancedTT

# ------- Cached FEN to Bitboard Conversion -------
@functools.lru_cache(maxsize=Config.FEN_CACHE_SIZE)
def make_x_cached(fen_1: str, fen_2: str): # Changed param names for clarity
    b1 = bitifyFEN(beautifyFEN(fen_1))
    b2 = bitifyFEN(beautifyFEN(fen_2))
    return (
        np.array(b1, dtype=np.float32).reshape(1, 769),
        np.array(b2, dtype=np.float32).reshape(1, 769)
    )

# ------- NN Evaluator -------
class NNEvaluator:
    def __init__(self):
        # Interpreter is now global, no need to initialize here
        pass

    def compare_positions(self, fen1: str, fen2: str) -> float:
        """Compare two positions. Returns value between 0 and 1.
        1.0 means fen1 is more winning for white than fen2
        0.0 means fen2 is more winning for white than fen1
        """
        try:
            x1_np, x2_np = make_x_cached(fen1, fen2)
            x1_copy = np.copy(x1_np)
            x2_copy = np.copy(x2_np)
            
            with interpreter_lock:
                interpreter.set_tensor(input_details[0]['index'], x1_copy) 
                interpreter.set_tensor(input_details[1]['index'], x2_copy)
                interpreter.invoke()
                raw_comparison = float(interpreter.get_tensor(output_details[0]['index'])[0][0])
            
            return raw_comparison
            
        except Exception as e:
            print(f"NN Comparison error: {e}")
            return 0.5  # Neutral if NN fails

    def compare_positions_ensemble(self, fen1: str, fen2: str) -> float:
        """
        TRAINING-INFORMED: Enhanced comparison using symmetry validation and ensemble methods.
        Based on training analysis: model was trained with random position swapping.
        """
        try:
            # Get comparisons in both directions
            forward = self.compare_positions(fen1, fen2)
            backward = self.compare_positions(fen2, fen1)
            
            # Training insight: these should be symmetric due to random swapping during training
            expected_backward = 1.0 - forward
            symmetry_error = abs(backward - expected_backward)
            
            # Training-based confidence assessment
            base_confidence = 1.0 - 2.0 * abs(forward - 0.5)  # Distance from neutral
            symmetry_confidence = 1.0 - min(symmetry_error * 2.0, 1.0)  # Symmetry quality
            
            # Combined confidence score
            overall_confidence = (base_confidence + symmetry_confidence) / 2.0
            
            if overall_confidence > 0.7:
                # High confidence: use forward comparison
                ensemble_result = forward
            elif overall_confidence > 0.4:
                # Medium confidence: blend with symmetry-corrected version
                symmetry_corrected = (forward + (1.0 - backward)) / 2.0
                ensemble_result = 0.3 * forward + 0.7 * symmetry_corrected
            else:
                # Low confidence: conservative symmetric average
                ensemble_result = (forward + (1.0 - backward)) / 2.0
            
            return ensemble_result
            
        except Exception as e:
            print(f"Training-informed ensemble comparison error: {e}")
            return 0.5

    def evaluate_move_comparison(self, current_fen: str, move_fens: list, player_is_white: bool) -> list:
        """
        Compare multiple move positions using the neural network.
        Returns a list of comparison scores for each move FEN relative to current position.
        Higher scores mean better for the current player.
        IMPROVED: Uses ensemble method to reduce color bias.
        """
        if not move_fens:
            return []
        
        scores = []
        for move_fen in move_fens:
            try:
                # Use ensemble comparison for better stability
                comparison = self.compare_positions_ensemble(current_fen, move_fen)
                
                # More symmetric interpretation for both colors
                if player_is_white:
                    # For White: if move_result > current, that's good (high comparison value)
                    # We want high scores for good moves, so don't invert
                    raw_score = comparison
                else:
                    # For Black: if move_result < current for White perspective, that's good for Black
                    # So we want low comparison values to translate to high scores
                    raw_score = 1.0 - comparison
                
                # Apply confidence scaling - moves near 0.5 are uncertain
                confidence = abs(comparison - 0.5) * 2.0  # 0 to 1 scale
                
                # Enhanced confidence adjustment for color balance
                if confidence < 0.2:
                    # Very uncertain comparison - use neutral score
                    score = 0.5
                elif confidence < 0.4:
                    # Somewhat uncertain - blend with neutral
                    score = 0.5 * 0.3 + raw_score * 0.7
                else:
                    # Confident comparison - use raw score
                    score = raw_score
                
                scores.append(score)
                
            except Exception as e:
                print(f"NN Move comparison error: {e}")
                scores.append(0.5)  # Neutral if NN fails
        
        return scores

    def evaluate_move_comparison_training_informed(self, current_fen: str, move_fens: list, player_is_white: bool) -> list:
        """
        TRAINING-INFORMED: Advanced move comparison that leverages training insights.
        Uses strategic reference positions and confidence-weighted scoring.
        """
        if not move_fens:
            return []
        
        scores = []
        
        # Strategic reference positions based on training understanding
        reference_positions = []
        
        # Primary reference: current position (most relevant)
        reference_positions.append(current_fen)
        
        # Training insight: starting position provides good baseline comparison
        starting_position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        if current_fen != starting_position:
            reference_positions.append(starting_position)
        
        # Add turn-specific reference (train data had both sides)
        try:
            board = chess.Board(current_fen)
            if board.turn == chess.WHITE:
                # For white to move, add a position where black just moved
                alt_position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
            else:
                # For black to move, add a position where white just moved  
                alt_position = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
            if alt_position not in reference_positions:
                reference_positions.append(alt_position)
        except:
            pass
        
        for move_fen in move_fens:
            try:
                evaluations = []
                confidences = []
                
                # Compare against each reference position
                for ref_fen in reference_positions:
                    if ref_fen != move_fen:
                        # Use training-informed ensemble comparison
                        comparison = self.compare_positions_ensemble(move_fen, ref_fen)
                        confidence = 1.0 - 2.0 * abs(comparison - 0.5)  # Distance from neutral
                        
                        # Player perspective adjustment (training had both perspectives)
                        if player_is_white:
                            # White wants higher comparison values (move better than reference)
                            adjusted_score = comparison
                        else:
                            # Black wants lower comparison values (move worse for white than reference)
                            adjusted_score = 1.0 - comparison
                        
                        evaluations.append(adjusted_score)
                        confidences.append(confidence)
                
                if evaluations:
                    # Training-informed confidence weighting
                    if len(confidences) > 0 and sum(confidences) > 0:
                        total_weight = sum(confidences)
                        weighted_score = sum(score * conf for score, conf in zip(evaluations, confidences)) / total_weight
                    else:
                        weighted_score = sum(evaluations) / len(evaluations)
                    
                    # Training insight: apply confidence boosting for clear evaluations
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
                    
                    if avg_confidence > 0.8:
                        # Very confident: use full evaluation
                        final_score = weighted_score
                    elif avg_confidence > 0.5:
                        # Moderately confident: slight regression to mean
                        final_score = 0.15 * 0.5 + 0.85 * weighted_score
                    else:
                        # Low confidence: stronger regression to mean
                        final_score = 0.4 * 0.5 + 0.6 * weighted_score
                    
                    scores.append(final_score)
                else:
                    scores.append(0.5)  # Neutral if no comparisons possible
                    
            except Exception as e:
                print(f"Training-informed move comparison error: {e}")
                scores.append(0.5)
        
        return scores

    def evaluate_absolute_score_white_pov(self, fen_to_evaluate: str, fen_context_for_pov: str) -> float:
        """
        Blunder-safe evaluation method. Uses classical evaluation as primary with NN adjustments.
        IMPROVED: Uses ensemble NN comparison for better color balance.
        """
        try:
            # Always start with classical evaluation as the foundation
            classic_cp = classical_material_eval(fen_to_evaluate)
            
            # Check for obviously bad positions (large material disadvantage)
            board = chess.Board(fen_to_evaluate)
            if board.is_checkmate():
                return -10000.0 if board.turn == chess.WHITE else 10000.0
            
            if board.is_stalemate() or board.is_insufficient_material():
                return 0.0
            
            # Use NN comparison conservatively only if we have meaningful context
            if fen_context_for_pov and fen_context_for_pov != fen_to_evaluate:
                # Use ensemble comparison for better color balance
                comparison = self.compare_positions_ensemble(fen_to_evaluate, fen_context_for_pov)
                
                # Only apply NN adjustment if the comparison is confident (not close to 0.5)
                confidence = abs(comparison - 0.5) * 2.0
                if confidence > 0.2:  # Only use NN if reasonably confident
                    # Convert comparison to small adjustment factor (-100 to +100 cp)
                    nn_adjustment = (comparison - 0.5) * 200 * confidence
                    
                    # Apply conservative adjustment - never more than 20% of classical eval
                    max_adjustment = abs(classic_cp) * 0.2 if classic_cp != 0 else 50
                    nn_adjustment = max(-max_adjustment, min(max_adjustment, nn_adjustment))
                    
                    final_score = classic_cp + nn_adjustment
                else:
                    final_score = classic_cp
            else:
                final_score = classic_cp
            
            # Stricter bounds to prevent extreme evaluations
            return max(-1500, min(1500, final_score))
            
        except Exception as e:
            print(f"NN Evaluation error: {e}")
            # Fallback to classical evaluation
            return classical_material_eval(fen_to_evaluate)

    def evaluate_absolute_score_white_pov_pure_nn(self, fen_to_evaluate: str, reference_fens: list) -> float:
        """
        PURE NN: Evaluation using only neural network comparisons with multiple references.
        No classical evaluation fallback.
        """
        try:
            if not reference_fens:
                return 0.0  # Neutral if no references
            
            board = chess.Board(fen_to_evaluate)
            
            # Handle terminal positions
            if board.is_checkmate():
                return -20000.0 if board.turn == chess.WHITE else 20000.0
            if board.is_stalemate() or board.is_insufficient_material():
                return 0.0
            
            evaluations = []
            confidences = []
            
            # Compare against multiple reference positions
            for ref_fen in reference_fens:
                if ref_fen != fen_to_evaluate:
                    comparison = self.compare_positions_ensemble(fen_to_evaluate, ref_fen)
                    confidence = abs(comparison - 0.5) * 2.0
                    
                    # Convert comparison to centipawn-like score
                    # comparison > 0.5 means fen_to_evaluate is better for white
                    if comparison > 0.5:
                        eval_score = (comparison - 0.5) * 2000 * confidence  # Scale to reasonable range
                    else:
                        eval_score = (comparison - 0.5) * 2000 * confidence  # Negative if worse for white
                    
                    evaluations.append(eval_score)
                    confidences.append(confidence)
            
            if evaluations:
                # Confidence-weighted average
                total_weight = sum(confidences) if sum(confidences) > 0 else 1.0
                weighted_eval = sum(eval_val * conf for eval_val, conf in zip(evaluations, confidences)) / total_weight
                
                # Scale based on overall confidence
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
                if avg_confidence < 0.3:
                    weighted_eval *= 0.5  # Reduce extreme evaluations for low confidence
                
                return max(-2000, min(2000, weighted_eval))  # Reasonable bounds
            else:
                return 0.0
                
        except Exception as e:
            print(f"Pure NN Evaluation error: {e}")
            return 0.0

    def evaluate_absolute_score_training_informed(self, fen_to_evaluate: str, game_context_fens: list = None) -> float:
        """
        TRAINING-INFORMED: Pure NN evaluation leveraging training insights about comparison model.
        Uses strategically selected reference positions based on training methodology.
        """
        try:
            board = chess.Board(fen_to_evaluate)
            
            # Handle terminal positions
            if board.is_checkmate():
                return -25000.0 if board.turn == chess.WHITE else 25000.0
            if board.is_stalemate() or board.is_insufficient_material():
                return 0.0
            
            # Strategic reference positions based on training insights
            reference_positions = []
            
            # Training insight 1: Starting position as universal reference
            starting_pos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            reference_positions.append(starting_pos)
            
            # Training insight 2: Add move 2 positions (common in training data)
            common_positions = [
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",  # 1.e4
                "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",  # 1.e4 e5
                "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",  # 1.e4 e5 2.Nf3
            ]
            
            for pos in common_positions:
                if pos != fen_to_evaluate:
                    reference_positions.append(pos)
            
            # Training insight 3: Add game context if available
            if game_context_fens:
                for context_fen in game_context_fens[-3:]:  # Last 3 positions
                    if context_fen != fen_to_evaluate and context_fen not in reference_positions:
                        reference_positions.append(context_fen)
            
            # Training insight 4: Add turn-specific reference
            if board.turn == chess.BLACK:
                # For black to move positions, add white-to-move reference
                reference_positions.append("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
            
            evaluations = []
            weights = []
            
            # Compare against each reference with training-informed weighting
            for i, ref_fen in enumerate(reference_positions):
                if ref_fen != fen_to_evaluate:
                    try:
                        # Use training-informed ensemble comparison
                        comparison = self.compare_positions_ensemble(fen_to_evaluate, ref_fen)
                        confidence = 1.0 - 2.0 * abs(comparison - 0.5)
                        
                        # Training insight: convert comparison to evaluation
                        # comparison > 0.5 means position is better for white than reference
                        if comparison >= 0.5:
                            eval_contribution = (comparison - 0.5) * 2.0 * 1500  # Scale to reasonable range
                        else:
                            eval_contribution = (comparison - 0.5) * 2.0 * 1500  # Negative for worse positions
                        
                        # Weight by reference quality (starting position most important)
                        if ref_fen == starting_pos:
                            reference_weight = 2.0  # Starting position is key reference
                        elif i < 4:  # Common opening positions
                            reference_weight = 1.5
                        else:  # Game context
                            reference_weight = 1.0
                        
                        # Combine confidence and reference weight
                        total_weight = confidence * reference_weight
                        
                        evaluations.append(eval_contribution)
                        weights.append(total_weight)
                        
                    except Exception as e:
                        continue
            
            if evaluations and weights:
                # Weighted average of evaluations
                total_weight = sum(weights)
                if total_weight > 0:
                    weighted_eval = sum(eval_val * weight for eval_val, weight in zip(evaluations, weights)) / total_weight
                else:
                    weighted_eval = sum(evaluations) / len(evaluations)
                
                # Training insight: add confidence scaling
                avg_weight = sum(weights) / len(weights) if weights else 0.5
                if avg_weight < 0.3:
                    # Low confidence: conservative evaluation
                    weighted_eval *= 0.6
                elif avg_weight > 0.8:
                    # High confidence: trust the evaluation more
                    weighted_eval *= 1.1
                
                # Bound to reasonable range
                final_eval = max(-2500, min(2500, weighted_eval))
                return final_eval
            else:
                # Fallback: minimal material evaluation
                return classical_material_eval(fen_to_evaluate)
                
        except Exception as e:
            print(f"Training-informed evaluation error: {e}")
            return classical_material_eval(fen_to_evaluate)

    def evaluate_absolute_score_final_optimized(self, fen_to_evaluate: str, game_context_fens: list = None) -> float:
        """
        FINAL OPTIMIZED: Balanced approach leveraging training insights with conservative scaling.
        Combines the best of training-informed methods with stability safeguards.
        """
        try:
            board = chess.Board(fen_to_evaluate)
            
            # Handle terminal positions
            if board.is_checkmate():
                return -20000.0 if board.turn == chess.WHITE else 20000.0
            if board.is_stalemate() or board.is_insufficient_material():
                return 0.0
            
            # Strategic reference positions (fewer, more carefully chosen)
            reference_positions = []
            
            # Core reference: starting position (universal baseline)
            starting_pos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            reference_positions.append(starting_pos)
            
            # Add one common middlegame position for context
            middlegame_ref = "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4"
            if fen_to_evaluate != middlegame_ref:
                reference_positions.append(middlegame_ref)
            
            # Include recent game context (only if available and reasonable)
            if game_context_fens and len(game_context_fens) > 1:
                # Use only the most recent context position
                recent_context = game_context_fens[-2] if len(game_context_fens) >= 2 else game_context_fens[-1]
                if recent_context != fen_to_evaluate and recent_context not in reference_positions:
                    reference_positions.append(recent_context)
            
            evaluations = []
            confidences = []
            
            # Compare against each reference with conservative weighting
            for i, ref_fen in enumerate(reference_positions):
                if ref_fen != fen_to_evaluate:
                    try:
                        # Use ensemble comparison
                        comparison = self.compare_positions_ensemble(fen_to_evaluate, ref_fen)
                        base_confidence = 1.0 - 2.0 * abs(comparison - 0.5)
                        
                        # Very conservative evaluation scaling
                        if comparison >= 0.5:
                            # Position better for white than reference
                            eval_contribution = (comparison - 0.5) * 600  # Much more conservative
                        else:
                            # Position worse for white than reference  
                            eval_contribution = (comparison - 0.5) * 600
                        
                        # Reference weighting (starting position most important)
                        if ref_fen == starting_pos:
                            reference_weight = 1.5  # Reduced from 2.0
                        else:
                            reference_weight = 1.0
                        
                        # Conservative confidence scaling
                        adjusted_confidence = base_confidence * 0.7  # More conservative
                        total_weight = adjusted_confidence * reference_weight
                        
                        evaluations.append(eval_contribution)
                        confidences.append(total_weight)
                        
                    except Exception:
                        continue
            
            if evaluations and confidences:
                # Weighted average with conservative bounds
                total_weight = sum(confidences)
                if total_weight > 0:
                    weighted_eval = sum(eval_val * conf for eval_val, conf in zip(evaluations, confidences)) / total_weight
                else:
                    weighted_eval = sum(evaluations) / len(evaluations)
                
                # Additional stability: blend with classical evaluation for safety
                classical_eval = classical_material_eval(fen_to_evaluate)
                
                # Confidence-based blending
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.3
                if avg_confidence > 0.6:
                    # High confidence: trust NN more but still blend
                    final_eval = 0.8 * weighted_eval + 0.2 * classical_eval
                elif avg_confidence > 0.3:
                    # Medium confidence: balanced blend
                    final_eval = 0.6 * weighted_eval + 0.4 * classical_eval
                else:
                    # Low confidence: trust classical more
                    final_eval = 0.3 * weighted_eval + 0.7 * classical_eval
                
                # Very conservative bounds
                final_eval = max(-800, min(800, final_eval))
                return final_eval
            else:
                # Fallback to classical evaluation
                return classical_material_eval(fen_to_evaluate)
                
        except Exception as e:
            print(f"Final optimized evaluation error: {e}")
            return classical_material_eval(fen_to_evaluate)

# ------- Engine Implementation -------
class Engine:
    def __init__(self, fen: str, model_path: str = "model.tflite"):
        self.board = chess.Board(fen)
        self.zobrist_hasher = ZobristHash()
        self.tt = FixedTT()
        self.nn_evaluator = NNEvaluator() # Changed: No longer passes model_path

        self.start_time_for_move = None
        self.time_limit_for_move = Config.ITERATIVE_DEEPENING_TIME_LIMIT_PER_MOVE

        self.nodes_searched = 0; self.q_nodes_searched = 0; self.tt_hits = 0
        self.beta_cutoffs = 0; self.nmp_cutoffs = 0; self.lmr_activations = 0
        # REMOVED: self.killer_moves for pure NN approach
        
        # Training-informed: Track game context for better NN evaluation
        self.game_context_fens = [fen]  # Track position history for context
        self.reference_positions = [fen]  # Start with initial position

    def get_version_info(self) -> dict:
        return {"name": ENGINE_NAME, "version": ENGINE_VERSION, "features": ENGINE_FEATURES}

    def time_is_up(self) -> bool:
        if self.start_time_for_move is None: return False
        elapsed = time.time() - self.start_time_for_move
        
        # Enhanced time management
        time_limit = self.time_limit_for_move
        
        # Allow extra time for critical positions
        if self.board.is_check():
            time_limit *= Config.SEARCH_TIME_EXTENSION_FACTOR
        elif len(list(self.board.legal_moves)) <= 5:
            time_limit *= 1.5  # Limited options need deeper thinking
        elif hasattr(self, 'nodes_searched') and self.nodes_searched > 0:
            # Early termination if search is not productive
            node_rate = self.nodes_searched / elapsed if elapsed > 0 else 0
            if node_rate < 1000 and elapsed > time_limit * 0.5:  # Very slow search
                return True
        
        # Check if we're in endgame (fewer pieces = more time for precision)
        piece_count = len(self.board.piece_map())
        if piece_count <= 12:  # Endgame
            time_limit *= 1.3
        elif piece_count <= 6:  # Late endgame  
            time_limit *= 1.6
        
        return elapsed >= time_limit

    def get_move(self) -> str:
        try:
            best_move_uci, _ = self.iterative_deepening_search()
            
            # Validate that the returned move is actually legal
            if best_move_uci and best_move_uci not in ["checkmate", "draw", "no_legal_moves_or_error"]:
                try:
                    move_obj = chess.Move.from_uci(best_move_uci)
                    if move_obj not in self.board.legal_moves:
                        print(f"Warning: Illegal move {best_move_uci} returned from search, using fallback")
                        best_move_uci = None
                except ValueError:
                    print(f"Warning: Invalid UCI format {best_move_uci}, using fallback")
                    best_move_uci = None
            
            if best_move_uci is None:
                if self.board.is_checkmate(): 
                    return "checkmate"
                if self.board.is_stalemate() or self.board.is_insufficient_material() or \
                   self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition():
                    return "draw"
                legal_moves = list(self.board.legal_moves)
                if legal_moves: 
                    return legal_moves[0].uci()
                return "no_legal_moves_or_error"
            
            return best_move_uci
        except Exception as e:
            print(f"Error in get_move: {e}")
            # Emergency fallback
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                return legal_moves[0].uci()
            return "no_legal_moves_or_error"

    def order_moves(self, legal_moves, ply_count, tt_best_move_uci):
        move_scores = []
        tt_move_obj = None
        if tt_best_move_uci:
            try:
                tt_move_obj = chess.Move.from_uci(tt_best_move_uci)
                if tt_move_obj not in legal_moves: tt_move_obj = None
            except ValueError: tt_move_obj = None

        # Enhanced NN approach: Use advanced NN evaluation for move ordering
        use_nn_ordering = (ply_count <= Config.NN_MOVE_ORDERING_MAX_PLY and 
                          len(legal_moves) <= Config.NN_MOVE_ORDERING_MAX_MOVES and 
                          not self.time_is_up())
        nn_scores = {}  # Use dict to map moves to scores
        
        if use_nn_ordering:
            try:
                current_fen = self.board.fen()
                move_fens = []
                moves_for_nn = []
                
                # Generate FENs for each legal move with safety checks
                for move in legal_moves:
                    try:
                        # Verify the move is actually legal before processing
                        if move in self.board.legal_moves:
                            self.board.push(move)
                            move_fen = self.board.fen()
                            move_fens.append(move_fen)
                            moves_for_nn.append(move)
                            self.board.pop()
                    except Exception as e:
                        print(f"Error processing move {move}: {e}")
                        continue
                
                if move_fens and len(move_fens) == len(moves_for_nn):
                    # Use ADVANCED NN comparison for better move evaluation
                    nn_score_list = self.nn_evaluator.evaluate_move_comparison_training_informed(
                        current_fen, move_fens, self.board.turn == chess.WHITE
                    )
                    
                    # Map scores to moves
                    if len(nn_score_list) == len(moves_for_nn):
                        for move, score in zip(moves_for_nn, nn_score_list):
                            nn_scores[move] = score
                    else:
                        use_nn_ordering = False
                else:
                    use_nn_ordering = False
                    
            except Exception as e:
                print(f"NN ordering error: {e}")
                use_nn_ordering = False

        # Enhanced move scoring with better prioritization
        for move in legal_moves:
            score = 0
            
            # Highest priority: TT move (search knowledge)
            if move == tt_move_obj: 
                score = 10000000
            # NN-guided move scoring (primary method)
            elif use_nn_ordering and move in nn_scores:
                nn_score = nn_scores[move]
                # Conservative NN score scaling 
                confidence = abs(nn_score - 0.5) * 2.0
                if confidence > 0.7:
                    # High confidence moves get modest boost
                    score = 5500000 + int(nn_score * 1000000)
                elif confidence > 0.5:
                    # Medium confidence moves get smaller boost
                    score = 5000000 + int(nn_score * 800000)
                elif confidence > 0.3:
                    # Low confidence moves
                    score = 4700000 + int(nn_score * 600000)
                else:
                    # Very low confidence - minimal boost
                    score = 4500000 + int(nn_score * 400000)
            # Enhanced capture scoring without chess knowledge
            elif self.board.is_capture(move):
                score = 3000000  # Increased base capture score
                victim_piece_obj = self.board.piece_at(move.to_square)
                if victim_piece_obj:
                    # Simple material-based ordering without piece knowledge
                    victim_type = victim_piece_obj.piece_type
                    if victim_type == chess.QUEEN:
                        score += 900000  # Highest value target
                    elif victim_type == chess.ROOK:
                        score += 500000
                    elif victim_type in [chess.BISHOP, chess.KNIGHT]:
                        score += 320000
                    elif victim_type == chess.PAWN:
                        score += 100000
                    
                # Promotion captures get extra boost
                if move.promotion:
                    score += 800000
                    
            elif move.promotion == chess.QUEEN: 
                score = 2500000  # Increased promotion value
            elif move.promotion: 
                score = 2000000  # Other promotions still valuable
            elif self.board.gives_check(move): 
                score = 1500000  # Increased check value - can be tactically powerful
            elif self.board.is_castling(move): 
                score = 1200000  # Castling often important
            else:
                # Base score for quiet moves - let NN handle all evaluation
                score = 1000000
                
                # Small positional hints without chess knowledge
                # These are search efficiency improvements, not chess knowledge
                to_square = move.to_square
                
                # Prefer central squares (search efficiency)
                center_distance = abs(chess.square_file(to_square) - 3.5) + abs(chess.square_rank(to_square) - 3.5)
                centrality_bonus = int((7 - center_distance) * 5000)
                score += centrality_bonus
                
                # Prefer forward moves for search efficiency (both colors)
                if self.board.turn == chess.WHITE:
                    rank_bonus = chess.square_rank(to_square) * 2000
                else:
                    rank_bonus = (7 - chess.square_rank(to_square)) * 2000
                score += rank_bonus
                
            move_scores.append((score, move))
        
        move_scores.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in move_scores]

    def iterative_deepening_search(self):
        self.start_time_for_move = time.time()
        self.nodes_searched = 0; self.q_nodes_searched = 0; self.tt_hits = 0
        self.beta_cutoffs = 0; self.nmp_cutoffs = 0; self.lmr_activations = 0
        
        # Enhanced TT management
        self.tt.new_search()  # Age TT entries
        if len(self.tt.entries) > (1 << (Config.TT_SIZE_POWER - 2)):
             self.tt = FixedTT()

        best_move_overall_uci = None; best_score_overall_white_pov = -float('inf')
        initial_fen_at_root = self.board.fen()
        
        # Debug: Verify initial board state
        initial_legal_moves = list(self.board.legal_moves)
        print(f"Debug: Starting position FEN: {initial_fen_at_root}")
        print(f"Debug: Initial legal moves: {[m.uci() for m in initial_legal_moves[:5]]}...")

        # Aspiration windows for improved search efficiency
        aspiration_window_size = 50  # Start with 50cp window
        prev_score = 0  # Previous iteration score
        
        for depth in range(1, Config.MAX_SEARCH_DEPTH_ID + 1):
            # Debug: Verify board state before each depth
            current_fen = self.board.fen()
            if current_fen != initial_fen_at_root:
                print(f"ERROR: Board state corrupted before depth {depth}!")
                print(f"Expected: {initial_fen_at_root}")
                print(f"Actual:   {current_fen}")
                # Reset board state
                self.board = chess.Board(initial_fen_at_root)
            
            if self.time_is_up() and depth > 1 :
                print(f"Time up before Depth {depth}. Using Depth {depth-1} result.")
                break
            
            search_start_time_this_depth = time.time()
            
            # Aspiration windows (for depth >= 3)
            if depth >= 3 and abs(prev_score) < 1000:  # Don't use aspiration in mate situations
                alpha = prev_score - aspiration_window_size
                beta = prev_score + aspiration_window_size
                
                # Try search with aspiration window
                try:
                    score_from_root_player_pov = self.alpha_beta(depth, alpha, beta,
                                                                initial_fen_at_root, True, 0)
                    
                    # If we failed high or low, research with full window
                    if score_from_root_player_pov <= alpha:
                        print(f"Depth {depth}: Failed low, researching...")
                        score_from_root_player_pov = self.alpha_beta(depth, -float('inf'), beta,
                                                                    initial_fen_at_root, True, 0)
                    elif score_from_root_player_pov >= beta:
                        print(f"Depth {depth}: Failed high, researching...")
                        score_from_root_player_pov = self.alpha_beta(depth, alpha, float('inf'),
                                                                    initial_fen_at_root, True, 0)
                                                                    
                except TimeoutError:
                    print(f"Search for Depth {depth} timed out during aspiration.")
                    break
                    
            else:
                # Full window search for early depths or mate situations
                try:
                    score_from_root_player_pov = self.alpha_beta(depth, -float('inf'), float('inf'),
                                                                    initial_fen_at_root, True, 0)
                except TimeoutError:
                    print(f"Search for Depth {depth} timed out.")
                    break
                    
            prev_score = score_from_root_player_pov
            
            # Adaptive aspiration window sizing
            if depth >= 3:
                if abs(score_from_root_player_pov - prev_score) > aspiration_window_size * 2:
                    aspiration_window_size = min(aspiration_window_size * 2, 200)  # Widen if we're failing
                else:
                    aspiration_window_size = max(aspiration_window_size * 0.8, 25)  # Narrow if stable

            # Debug: Verify board state after search
            post_search_fen = self.board.fen()
            if post_search_fen != initial_fen_at_root:
                print(f"ERROR: Board state corrupted after depth {depth} search!")
                print(f"Expected: {initial_fen_at_root}")
                print(f"Actual:   {post_search_fen}")
                # Reset board state
                self.board = chess.Board(initial_fen_at_root)

            search_duration_this_depth = time.time() - search_start_time_this_depth
            
            tt_entry_root = self.tt.lookup(self.zobrist_hasher.hash(self.board))
            current_best_move_uci_from_tt = None
            if tt_entry_root and len(tt_entry_root) >= 3 and tt_entry_root[2]:
                current_best_move_uci_from_tt = tt_entry_root[2]
                
                # CRITICAL FIX: Validate TT move is actually legal
                if current_best_move_uci_from_tt:
                    try:
                        move_obj = chess.Move.from_uci(current_best_move_uci_from_tt)
                        current_legal_moves = list(self.board.legal_moves)
                        if move_obj not in current_legal_moves:
                            print(f"Debug: TT move {current_best_move_uci_from_tt} not in legal moves")
                            print(f"Debug: Current legal moves: {[m.uci() for m in current_legal_moves[:5]]}...")
                            print(f"Debug: Current board FEN: {self.board.fen()}")
                            current_best_move_uci_from_tt = None
                    except (ValueError, chess.InvalidMoveError):
                        print(f"Warning: TT move {current_best_move_uci_from_tt} is invalid UCI, ignoring")
                        current_best_move_uci_from_tt = None
            
            score_white_pov = score_from_root_player_pov if self.board.turn == chess.WHITE else -score_from_root_player_pov

            if current_best_move_uci_from_tt:
                best_move_overall_uci = current_best_move_uci_from_tt
                best_score_overall_white_pov = score_white_pov
                
                # Enhanced search statistics
                tt_stats = self.tt.get_stats()
                nodes_per_sec = int(self.nodes_searched / search_duration_this_depth) if search_duration_this_depth > 0 else 0
                
                print(f"Depth {depth}: PV={best_move_overall_uci} Score(WPOV)={score_white_pov:.2f} Nodes={self.nodes_searched} "
                      f"(Q:{self.q_nodes_searched}) NPS={nodes_per_sec} TTHits={self.tt_hits} ({tt_stats['hit_rate']:.1f}%) "
                      f"BetaCuts={self.beta_cutoffs} NMPCuts={self.nmp_cutoffs} LMR_Acts={self.lmr_activations} "
                      f"Time={search_duration_this_depth:.2f}s AspWin={aspiration_window_size}")
            else: # Should be rare if search completes
                nodes_per_sec = int(self.nodes_searched / search_duration_this_depth) if search_duration_this_depth > 0 else 0
                print(f"Depth {depth}: No PV from TT for root. Score(WPOV)={score_white_pov:.2f} Nodes={self.nodes_searched} "
                      f"NPS={nodes_per_sec} Time={search_duration_this_depth:.2f}s")

            if abs(score_white_pov) > 90000: print("Mate score detected..."); break # Check WPOV score
        
        # Final validation before returning
        if best_move_overall_uci is not None:
            try:
                move_obj = chess.Move.from_uci(best_move_overall_uci)
                final_legal_moves = list(self.board.legal_moves)
                if move_obj not in final_legal_moves:
                    print(f"Debug: Final validation failed for move {best_move_overall_uci}")
                    print(f"Debug: Final legal moves: {[m.uci() for m in final_legal_moves[:5]]}...")
                    print(f"Debug: Final board FEN: {self.board.fen()}")
                    best_move_overall_uci = None
                else:
                    # Training-informed: Update game context with new position
                    try:
                        self.board.push(move_obj)
                        new_fen = self.board.fen()
                        self.game_context_fens.append(new_fen)
                        # Keep only last 10 positions for context
                        if len(self.game_context_fens) > 10:
                            self.game_context_fens = self.game_context_fens[-10:]
                        self.board.pop()  # Restore original position
                    except:
                        pass  # Don't break if context update fails
            except (ValueError, chess.InvalidMoveError):
                print(f"Warning: Final move {best_move_overall_uci} is invalid, using fallback")
                best_move_overall_uci = None
        
        if best_move_overall_uci is None:
             legal_moves = list(self.board.legal_moves)
             if legal_moves:
                 best_move_overall_uci = legal_moves[0].uci()
                 print("Warning: Fallback to first legal move at end of ID.")
        
        return best_move_overall_uci, best_score_overall_white_pov

    def alpha_beta(self, depth_remaining: int, alpha: float, beta: float, 
                     root_fen_for_nn_context: str, is_pv_node: bool, ply_count: int) -> float: # Returns score from current player's POV
        
        if self.time_is_up() and ply_count > 0 : raise TimeoutError("Search time limit exceeded")

        self.nodes_searched += 1
        alpha_original = alpha
        current_node_zobrist_key = self.zobrist_hasher.hash(self.board)

        if ply_count > 0 and (self.board.can_claim_threefold_repetition() or self.board.is_fifty_moves()):
            return 0.0 

        # Re-enable transposition table
        tt_entry = self.tt.lookup(current_node_zobrist_key)
        tt_best_move_uci = None
        if tt_entry:
            # Handle both old (4-element) and new (5-element) TT entries
            if len(tt_entry) >= 4:
                tt_depth, tt_score_white_pov, tt_best_move_uci, tt_flag = tt_entry[:4]
            else:
                tt_depth, tt_score_white_pov, tt_best_move_uci, tt_flag = None, None, None, None
            
            # Validate that the TT move is actually legal in this position
            if tt_best_move_uci:
                try:
                    tt_move_obj = chess.Move.from_uci(tt_best_move_uci)
                    if tt_move_obj not in self.board.legal_moves:
                        # TT move is not legal in this position, ignore it
                        tt_best_move_uci = None
                except (ValueError, chess.InvalidMoveError):
                    # Invalid move format, ignore it
                    tt_best_move_uci = None
            
            if tt_depth >= depth_remaining and tt_best_move_uci is not None:
                self.tt_hits += 1
                tt_score_current_player_pov = tt_score_white_pov if self.board.turn == chess.WHITE else -tt_score_white_pov
                if tt_flag == "EXACT": return tt_score_current_player_pov
                if tt_flag == "LOWERBOUND": alpha = max(alpha, tt_score_current_player_pov)
                elif tt_flag == "UPPERBOUND": beta = min(beta, tt_score_current_player_pov)
                if alpha >= beta: return tt_score_current_player_pov

        if self.board.is_checkmate(): return -100000.0 + ply_count 
        if self.board.is_stalemate() or self.board.is_insufficient_material(): return 0.0

        if depth_remaining <= 0:
            return self.quiescence(alpha, beta, root_fen_for_nn_context, ply_count)

        # Simplified NMP - let NN handle tactical complexity
        can_do_nmp = (depth_remaining >= (Config.NMP_REDUCTION + 1) and 
                     not self.board.is_check() and 
                     ply_count > 0)
        
        if can_do_nmp:
            non_pawn_material = sum(len(self.board.pieces(pt, self.board.turn)) for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
            if non_pawn_material < 2 : can_do_nmp = False
        
        if can_do_nmp:
            null_move_is_pushed = False
            try:
                self.board.push(chess.Move.null())
                null_move_is_pushed = True
                score = -self.alpha_beta(depth_remaining - 1 - Config.NMP_REDUCTION, -beta, -alpha, 
                                         root_fen_for_nn_context, False, ply_count + 1) # NMP is not a PV node
            finally:
                if null_move_is_pushed:
                    self.board.pop()
                    
            if score >= beta: 
                self.nmp_cutoffs +=1
                score_to_store_white_pov = score if self.board.turn == chess.WHITE else -score
                self.tt.store(current_node_zobrist_key, (depth_remaining, score_to_store_white_pov, None, "LOWERBOUND"))
                return beta
        
        legal_moves = list(self.board.legal_moves)
        if not legal_moves: return 0.0 
        ordered_moves = self.order_moves(legal_moves, ply_count, tt_best_move_uci)
        
        best_move_found_uci_for_tt = None
        value_from_current_player_pov = -float('inf') 

        for i, move in enumerate(ordered_moves):
            # Double-check that the move is still legal (paranoid check)
            if move not in self.board.legal_moves:
                print(f"Warning: Move {move.uci()} from ordering is not in legal moves, skipping")
                continue
                
            # Important: Save board state before push
            move_is_pushed = False
            try:
                self.board.push(move)
                move_is_pushed = True
                
                is_giving_check = self.board.is_check()
                is_capture = self.board.is_capture(move)
                is_promotion = move.promotion is not None
                
                # NN-guided search extensions
                extension = 0
                if is_giving_check and depth_remaining >= 2:
                    extension = 1  # Extend checks
                
                # NN-guided critical position extension
                if depth_remaining >= 3 and ply_count <= 4 and not self.time_is_up():
                    try:
                        # Use NN to detect if this is a critical position
                        current_position_fen = self.board.fen()
                        comparison_with_parent = self.nn_evaluator.compare_positions_ensemble(
                            current_position_fen, root_fen_for_nn_context
                        )
                        # Extend if position evaluation changes significantly
                        position_criticality = abs(comparison_with_parent - 0.5) * 2.0
                        if position_criticality > 0.8:  # High criticality
                            extension = max(extension, 1)
                    except:
                        pass  # Don't extend if NN evaluation fails
                    
                child_search_depth = depth_remaining - 1 + extension
                
                current_move_score_from_child_pov = 0 

                if i == 0 : # First move, always full window search, PV status depends on parent
                    current_move_score_from_child_pov = -self.alpha_beta(child_search_depth, -beta, -alpha,
                                                                    root_fen_for_nn_context, 
                                                                    is_pv_node, # Child is PV only if parent is PV and it's the first move
                                                                    ply_count + 1)
                else: # Subsequent moves, PVS / Enhanced LMR logic
                    # Enhanced LMR with more conditions
                    base_reduction = Config.LMR_REDUCTION
                    can_reduce = (
                        child_search_depth >= base_reduction + 1 and 
                        i >= Config.LMR_MIN_MOVES_TRIED and 
                        not extension and 
                        not is_capture and
                        not is_promotion and
                        not is_giving_check and
                        ply_count >= 3 and  # Don't reduce near root
                        not is_pv_node  # Reduce less in PV nodes
                    )
                    
                    # Adaptive reduction based on move number and depth
                    if can_reduce:
                        # Increase reduction for later moves and higher depths
                        extra_reduction = 0
                        if i >= 8:  # Late moves
                            extra_reduction += 1
                        if depth_remaining >= 6:  # Deep searches
                            extra_reduction += 1
                        if ply_count >= 8:  # Far from root
                            extra_reduction += 1
                        
                        reduction = min(base_reduction + extra_reduction, child_search_depth - 1)
                        
                        self.lmr_activations += 1
                        current_move_score_from_child_pov = -self.alpha_beta(child_search_depth - reduction, 
                                                                        -alpha -1, -alpha, 
                                                                        root_fen_for_nn_context, False, ply_count + 1)
                        
                        # If LMR failed high, research with less reduction
                        if current_move_score_from_child_pov > alpha and reduction > 1:
                            current_move_score_from_child_pov = -self.alpha_beta(child_search_depth - 1, 
                                                                            -alpha -1, -alpha, 
                                                                            root_fen_for_nn_context, False, ply_count + 1)
                    else: 
                        # Standard null-window search
                        current_move_score_from_child_pov = -self.alpha_beta(child_search_depth, 
                                                                        -alpha -1, -alpha, 
                                                                        root_fen_for_nn_context, False, ply_count + 1)

                    # If null-window search failed high, re-search with full window (PVS)
                    if current_move_score_from_child_pov > alpha and current_move_score_from_child_pov < beta : 
                        current_move_score_from_child_pov = -self.alpha_beta(child_search_depth, -beta, -alpha,
                                                                        root_fen_for_nn_context, is_pv_node, ply_count + 1)
                
            finally:
                # CRITICAL: Always pop the move, even if an exception occurred
                if move_is_pushed:
                    self.board.pop()

            if current_move_score_from_child_pov > value_from_current_player_pov:
                value_from_current_player_pov = current_move_score_from_child_pov
                # Only store the move if it's still legal (extra safety)
                best_move_found_uci_for_tt = move.uci()
            
            alpha = max(alpha, value_from_current_player_pov)
            if alpha >= beta:
                self.beta_cutoffs +=1
                break
        
        final_score_white_pov = value_from_current_player_pov if self.board.turn == chess.WHITE else -value_from_current_player_pov
        tt_flag = "EXACT"
        if value_from_current_player_pov <= alpha_original: tt_flag = "UPPERBOUND"
        elif value_from_current_player_pov >= beta: tt_flag = "LOWERBOUND"
        
        self.tt.store(current_node_zobrist_key, (depth_remaining, final_score_white_pov, best_move_found_uci_for_tt, tt_flag))
        return value_from_current_player_pov

    def quiescence(self, alpha: float, beta: float, 
                    root_fen_for_nn_context: str, ply_count: int) -> float: # Returns score from current player's POV
        self.q_nodes_searched += 1
        
        if self.time_is_up() and ply_count > 0: raise TimeoutError("Search time limit exceeded")

        if self.board.can_claim_threefold_repetition() or self.board.is_fifty_moves(): return 0.0
        
        # Use TRAINING-INFORMED NN evaluation for stand-pat
        current_fen = self.board.fen()
        
        # Use final optimized evaluation with game context
        stand_pat_score_white_pov = self.nn_evaluator.evaluate_absolute_score_final_optimized(
            current_fen, self.game_context_fens
        )
        
        stand_pat_current_player_pov = stand_pat_score_white_pov if self.board.turn == chess.WHITE else -stand_pat_score_white_pov

        if stand_pat_current_player_pov >= beta: return beta
        alpha = max(alpha, stand_pat_current_player_pov)

        q_moves_with_scores = []
        for move in self.board.legal_moves:
            score = 0
            if self.board.is_capture(move):
                score = 2000000 
                victim_type = chess.PAWN
                if self.board.is_en_passant(move): victim_type = chess.PAWN
                else:
                    victim_piece_obj = self.board.piece_at(move.to_square)
                    if victim_piece_obj: victim_type = victim_piece_obj.piece_type
                # Simplified capture ordering - just by victim value
                if victim_type == chess.QUEEN: score += 900000
                elif victim_type == chess.ROOK: score += 500000
                elif victim_type in [chess.BISHOP, chess.KNIGHT]: score += 300000
                else: score += 100000  # Pawn
            elif move.promotion:
                score = 1500000 + (900000 if move.promotion == chess.QUEEN else 300000)
            
            if score > 0 : q_moves_with_scores.append((score, move))
        
        q_moves_with_scores.sort(key=lambda item: item[0], reverse=True)
        ordered_q_moves = [item[1] for item in q_moves_with_scores]

        if not ordered_q_moves or ply_count >= Config.MAX_PLY_FOR_KILLERS + Config.QUIESCENCE_MAX_DEPTH_RELATIVE :
            return stand_pat_current_player_pov

        best_val_current_player_pov = stand_pat_current_player_pov

        for move in ordered_q_moves:
            move_is_pushed = False
            try:
                self.board.push(move)
                move_is_pushed = True
                score = -self.quiescence(-beta, -alpha, 
                                         root_fen_for_nn_context, 
                                         ply_count + 1)
            finally:
                if move_is_pushed:
                    self.board.pop()
            
            if score > best_val_current_player_pov:
                best_val_current_player_pov = score
            
            alpha = max(alpha, best_val_current_player_pov)
            if alpha >= beta: break 
        
        return best_val_current_player_pov

def classical_material_eval(fen: str) -> int:
    """Minimal classical evaluation - just basic material count as safety net for NN"""
    board = chess.Board(fen)
    MATERIAL_VALUES = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
                       chess.ROOK: 500, chess.QUEEN: 900}
    score = 0
    
    # Basic material count only
    for square, piece in board.piece_map().items():
        value = MATERIAL_VALUES.get(piece.piece_type, 0)
        score += value if piece.color == chess.WHITE else -value
    
    return score