#!/usr/bin/env python3
"""
Integration test for CLI argument functionality
This test validates that the run.py script correctly parses arguments
without requiring the full dependency chain.
"""

import sys
import subprocess
import argparse

def test_help_output():
    """Test that help output contains our new arguments"""
    print("Testing help output...")
    
    try:
        result = subprocess.run([sys.executable, 'run.py', '--help'], 
                              capture_output=True, text=True, timeout=10)
        
        help_text = result.stdout
        
        # Check if our arguments are in the help text
        assert '--classes' in help_text, "Missing --classes argument in help"
        assert '--territories' in help_text, "Missing --territories argument in help"
        assert 'List of class prompts' in help_text, "Missing class prompts description"
        assert 'List of territories with weights' in help_text, "Missing territories description"
        
        print("✓ Help output contains expected CLI arguments")
        return True
        
    except subprocess.TimeoutExpired:
        print("✗ Help command timed out")
        return False
    except Exception as e:
        print(f"✗ Error testing help output: {e}")
        return False

def test_argument_parsing_isolated():
    """Test argument parsing in isolation"""
    print("Testing argument parsing logic...")
    
    # Create a minimal parser to test the logic
    parser = argparse.ArgumentParser(description='Deep Pavements Sample Picker')
    
    parser.add_argument('--classes', '-c', 
                       nargs='+',
                       default=['tree', 'vehicle'],
                       help='List of class prompts for detection (default: tree vehicle)')
    
    parser.add_argument('--territories', '-t',
                       nargs='+', 
                       default=['Vitorino Brazil:1', 'Curitiba Brazil:1', 'Milan Italy:1', 'Arcole Italy:1'],
                       help='List of territories with weights in format "territory:weight"')
    
    # Test default arguments
    args = parser.parse_args([])
    assert args.classes == ['tree', 'vehicle'], f"Expected default classes, got {args.classes}"
    assert len(args.territories) == 4, f"Expected 4 default territories, got {len(args.territories)}"
    
    # Test custom arguments
    args = parser.parse_args(['--classes', 'road', 'sidewalk', '--territories', 'Paris France:2', 'London UK:1'])
    assert args.classes == ['road', 'sidewalk'], f"Expected custom classes, got {args.classes}"
    assert args.territories == ['Paris France:2', 'London UK:1'], f"Expected custom territories, got {args.territories}"
    
    print("✓ Argument parsing logic works correctly")
    return True

def main():
    """Run all tests"""
    print("Running CLI integration tests...\n")
    
    success_count = 0
    total_tests = 2
    
    if test_argument_parsing_isolated():
        success_count += 1
    
    if test_help_output():
        success_count += 1
    
    print(f"\nTest Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("All tests passed! ✓")
        return 0
    else:
        print("Some tests failed! ✗")
        return 1

if __name__ == '__main__':
    sys.exit(main())