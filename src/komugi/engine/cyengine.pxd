# cython: language_level=3
"""
Declaration file for cyengine — move application + terminal detection.

Allows other .pyx modules to ``cimport`` the public helpers.
"""

from komugi.engine.cyboard cimport Board

cpdef bint is_repetition_draw(int count)
cpdef bint is_terminal_nonmate(Board b, int repetition_count)
