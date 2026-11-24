# cython: binding=True
# cython: embedsignature=True
# cython: embedsignature.format=python
"""Trivial extension module proving the Cython toolchain is wired up.

The directives above are the project-wide mandate, not decoration: `binding`
makes cpdef functions introspectable (so mypy.stubtest can compare them
against the stub) and the embedded python-format signatures are what stubtest
reads.
"""


cpdef int probe_add(int a, int b):
    """Return a + b, computed in compiled C."""
    return a + b
