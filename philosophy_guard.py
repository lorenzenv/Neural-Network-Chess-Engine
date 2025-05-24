#!/usr/bin/env python3
"""
PURE NN PHILOSOPHY GUARD SYSTEM

🚨 This module prevents chess knowledge from creeping into the pure NN engine.
Run this as a pre-commit hook or CI check to catch philosophy violations.
"""

import os
import re
import sys
from pathlib import Path

# 🚫 FORBIDDEN TERMS: These indicate chess knowledge creep
FORBIDDEN_PATTERNS = [
    # Classical evaluation terms
    r'classical.*eval',
    r'material.*eval',
    r'positional.*eval',
    r'piece.*value.*table',
    r'pawn.*structure',
    r'king.*safety',
    
    # Opening/endgame knowledge
    r'opening.*book',
    r'endgame.*table',
    r'opening.*principle',
    r'endgame.*rule',
    
    # Tactical patterns (hard-coded)
    r'pin.*detection',
    r'fork.*detection',
    r'skewer.*pattern',
    r'discovered.*attack',
    
    # Chess-specific heuristics
    r'center.*control',
    r'piece.*activity',
    r'tempo.*eval',
    r'development.*bonus',
    
    # Evaluation blending (except minimal safety)
    r'classical.*blend',
    r'heuristic.*weight',
    r'eval.*mix',
]

# ⚠️  WARNING TERMS: These need manual review
WARNING_PATTERNS = [
    r'piece.*value',
    r'material.*count',
    r'chess.*knowledge',
    r'traditional.*eval',
    r'rule.*based',
]

# ✅ ALLOWED TERMS: These are fine even if they sound chess-related
ALLOWED_EXCEPTIONS = [
    'move_ordering',  # Algorithm optimization
    'alpha_beta',     # Search algorithm
    'transposition',  # Search optimization
    'quiescence',     # Search technique
    'killer_move',    # Search heuristic
    'null_move',      # Search pruning
    'late_move_reduction',  # Search optimization
    'aspiration_window',    # Search technique
    'iterative_deepening',  # Search strategy
    'zobrist_hash',   # Position identification
    'time_management', # Search control
    'terminal_position',    # Game end detection
    'checkmate',      # Basic game rules
    'stalemate',      # Basic game rules
    'illegal_move',   # Move validation
    'neural_network', # Our core technology
    'inference',      # NN evaluation
    'comparison',     # NN method
    'evaluation',     # Generic term (context matters)
    'position',       # Generic term
    'move',           # Generic term
    'search',         # Algorithm term
    'score',          # Generic term
]

class PhilosophyViolation(Exception):
    """Raised when chess knowledge is detected."""
    pass

class PhilosophyWarning(Warning):
    """Raised when suspicious patterns are detected."""
    pass

def check_file_for_violations(file_path: Path) -> tuple:
    """
    Check a single file for philosophy violations.
    
    Returns:
        (violations, warnings) - lists of detected issues
    """
    violations = []
    warnings = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line_lower = line.lower()
            
            # Skip comments that are explanatory
            if line.strip().startswith('#') and 'philosophy' in line_lower:
                continue
            
            # Check for forbidden patterns
            for pattern in FORBIDDEN_PATTERNS:
                if re.search(pattern, line_lower):
                    # Check if it's an allowed exception
                    is_exception = any(exception in line_lower for exception in ALLOWED_EXCEPTIONS)
                    if not is_exception:
                        violations.append({
                            'file': file_path,
                            'line': line_num,
                            'content': line.strip(),
                            'pattern': pattern,
                            'severity': 'VIOLATION'
                        })
            
            # Check for warning patterns
            for pattern in WARNING_PATTERNS:
                if re.search(pattern, line_lower):
                    warnings.append({
                        'file': file_path,
                        'line': line_num,
                        'content': line.strip(),
                        'pattern': pattern,
                        'severity': 'WARNING'
                    })
    
    except Exception as e:
        print(f"Error checking {file_path}: {e}")
    
    return violations, warnings

def check_codebase_philosophy(root_dir: str = ".") -> bool:
    """
    Check entire codebase for philosophy violations.
    
    Returns:
        True if philosophy is maintained, False if violations found
    """
    root_path = Path(root_dir)
    all_violations = []
    all_warnings = []
    
    # Files to check
    python_files = list(root_path.glob("*.py"))
    
    # Skip certain files
    skip_files = {'philosophy_guard.py', 'util.py', 'train.py'}
    python_files = [f for f in python_files if f.name not in skip_files]
    
    print("🛡️  Checking Pure NN Philosophy...")
    print(f"   Scanning {len(python_files)} Python files...")
    
    for file_path in python_files:
        violations, warnings = check_file_for_violations(file_path)
        all_violations.extend(violations)
        all_warnings.extend(warnings)
    
    # Report results
    if all_violations:
        print(f"\n❌ PHILOSOPHY VIOLATIONS DETECTED ({len(all_violations)}):")
        for v in all_violations:
            print(f"   {v['file'].name}:{v['line']} - {v['content']}")
            print(f"      Pattern: {v['pattern']}")
        print("\n🚨 FIX REQUIRED: Remove chess knowledge from code!")
        return False
    
    if all_warnings:
        print(f"\n⚠️  PHILOSOPHY WARNINGS ({len(all_warnings)}):")
        for w in all_warnings:
            print(f"   {w['file'].name}:{w['line']} - {w['content']}")
            print(f"      Pattern: {w['pattern']}")
        print("\n📝 REVIEW REQUIRED: Check if these add chess knowledge")
    
    if not all_violations:
        print("✅ Pure NN Philosophy maintained - no chess knowledge detected!")
        return True

def philosophy_guard_pre_commit():
    """Pre-commit hook to prevent chess knowledge commits."""
    print("🛡️  Pre-commit Philosophy Guard")
    
    if not check_codebase_philosophy():
        print("\n💥 COMMIT BLOCKED: Philosophy violations detected!")
        print("   Remove chess knowledge before committing.")
        sys.exit(1)
    else:
        print("✅ Commit approved - Pure NN philosophy maintained")

def create_philosophy_reminder():
    """Create a reminder file about the philosophy."""
    reminder_content = """
🧠 PURE NEURAL NETWORK CHESS ENGINE PHILOSOPHY

⚠️  CRITICAL REMINDER: This engine contains ZERO chess knowledge.

❌ DO NOT ADD:
- Classical evaluation (material, position, etc.)
- Opening books or endgame tables  
- Hard-coded tactical patterns
- Chess-specific heuristics
- Rule-based evaluations

✅ ONLY ALLOWED:
- Neural network inference
- Search algorithms (alpha-beta, etc.)
- Basic game rules (move legality, checkmate detection)
- Algorithm optimizations (transposition tables, etc.)

🎯 THE GOAL: Let the neural network learn chess patterns from data,
not from programmed knowledge.

If you want to add chess knowledge, you're working on the wrong engine!
"""
    
    with open("PURE_NN_PHILOSOPHY.md", "w") as f:
        f.write(reminder_content)
    
    print("📝 Created PURE_NN_PHILOSOPHY.md reminder")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pure NN Philosophy Guard")
    parser.add_argument("--check", action="store_true", help="Check codebase for violations")
    parser.add_argument("--pre-commit", action="store_true", help="Run as pre-commit hook")
    parser.add_argument("--create-reminder", action="store_true", help="Create philosophy reminder")
    
    args = parser.parse_args()
    
    if args.pre_commit:
        philosophy_guard_pre_commit()
    elif args.create_reminder:
        create_philosophy_reminder()
    else:
        check_codebase_philosophy()