# cython: language_level=3
"""
Declaration file for cymovegen — Cython move generation for Gungi.

Allows other .pyx modules to ``cimport`` the public helpers if needed.
"""

from komugi.engine.cyboard cimport Board

cpdef void fill_legal_mask(Board b, unsigned char[::1] mask)
