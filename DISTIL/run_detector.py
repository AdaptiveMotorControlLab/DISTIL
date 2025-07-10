#!/usr/bin/env python3
"""
Simple entry point script to run the backdoor target detector
"""

import sys
import os

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from DISTIL.scripts.detect_backdoor_targets import main

if __name__ == "__main__":
    main() 