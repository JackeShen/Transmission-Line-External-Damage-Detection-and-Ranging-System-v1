#!/usr/bin/env python
"""Convenience launcher for lineUI.
Usage: conda run -n goal python new_approach/lineUI/run.py
"""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
from new_approach.lineUI.app import main
main()
