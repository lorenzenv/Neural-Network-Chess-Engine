#!/usr/bin/env python3
"""
PURE NEURAL NETWORK CHESS ENGINE - MAIN INTERFACE

🚨 ZERO CHESS KNOWLEDGE: This is the main entry point for the pure NN engine.
All chess evaluation comes from the neural network, not programmed knowledge.

This engine maintains the same interface as the old engine but with pure NN philosophy.
"""

from search_coordinator import PureNeuralNetworkEngine

# Engine identification
ENGINE_VERSION = "3.0.0-PureNN"
ENGINE_NAME = "Pure Neural Network Chess Engine"
ENGINE_FEATURES = [
    "🧠 100% Neural Network Evaluation",
    "🚫 ZERO Chess Knowledge", 
    "🔍 Advanced Alpha-Beta Search",
    "⚡ Enhanced Transposition Table",
    "🎯 NN-Guided Move Ordering",
    "🕐 Intelligent Time Management",
    "🌟 Pure Machine Learning Approach",
    "🛡️ Philosophy Guard Protection"
]

# Main engine class (for compatibility)
class Engine:
    """
    Main engine interface - Pure NN only.
    
    🚨 This is a compatibility wrapper around PureNeuralNetworkEngine.
    Maintains the same interface but enforces NN purity.
    """
    
    def __init__(self, fen: str = None):
        self.engine = PureNeuralNetworkEngine(fen)
        print("🧠 Pure NN Chess Engine loaded - ZERO chess knowledge!")
    
    def get_move(self, time_limit: float = None) -> str:
        """Get best move using pure neural network evaluation."""
        return self.engine.get_move(time_limit)
    
    def set_position(self, fen: str):
        """Set board position."""
        self.engine.set_position(fen)

# Validate philosophy on import
if __name__ == "__main__":
    from neural_network_inference import validate_pure_nn_philosophy
    
    print("🛡️  Validating Pure NN Philosophy...")
    validate_pure_nn_philosophy()
    
    # Test engine
    print("\n🧠 Testing Pure NN Engine...")
    engine = Engine()
    move = engine.get_move(2.0)
    print(f"First move: {move}")
    
    print("\n✅ Pure NN Engine test complete!")