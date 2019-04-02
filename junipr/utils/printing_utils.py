"""Printing utilities for JUNIPR."""
from __future__ import absolute_import

__all__ = ['print_progress']

def print_progress(step_i, n_steps, n_print_outs=10):
    if step_i != 0 and step_i%(n_steps//n_print_outs) ==0:
        print("-- Step", step_i, "of", n_steps, flush=True)