"""The v1 side of kom-023 §D1's cross-engine protocol (KOM-023b).

``komugi.eval.v1bridge`` is the harness's end of the wire; this package is the
**other** end -- the program that runs inside ``nix develop .#v1``, holds v1's
own engine and network, and answers ``hello``/``reset``/``observe``/``pick``/
``probe``/``bye``. Run it as::

    nix develop .#v1 -c python -m komugi.eval.v1runner --check-imports

Three properties are structural rather than incidental, and every one of them
is what makes a module in this package possible to test at all:

* **stdlib only.** Nothing here imports numpy, torch, or any other komugi
  module. ``torch`` and ``numpy`` are reached only through
  ``runner._optional_module``, and only to *report* their versions in ``hello``'s
  device block. So this package type-checks and unit-tests in the default shell,
  under Komugi's python 3.12, while running for real under ``.#v1``'s 3.11.
* **The wire is vendored, not imported** -- see :mod:`komugi.eval.v1runner.protocol`
  for why, and for the drift test that keeps the copy honest.
* **v1 is injected, never imported at module scope.**
  :mod:`komugi.eval.v1runner.reference` resolves v1's ``Game``, net and search
  from ``--reference-root`` and three ``MODULE:ATTR`` specs, so the serve loop
  can be driven end to end against fakes with no v1 present.

The module layout, in dependency order:

* :mod:`~komugi.eval.v1runner.protocol` -- §D1's framing, §D2.3's record, P1.
* :mod:`~komugi.eval.v1runner.counter` -- §D3.5's counted evaluator and refusal.
* :mod:`~komugi.eval.v1runner.reference` -- loading v1, and refusing to.
* :mod:`~komugi.eval.v1runner.runner` -- the serve loop and the entry point.
"""

from __future__ import annotations

from komugi.eval.v1runner.counter import (
    BudgetRefused,
    CountedNet,
    EvalCounter,
    positions_in,
)
from komugi.eval.v1runner.protocol import (
    EVAL_CEILING,
    PROTOCOL_VERSION,
    RULES_DIVERGENCE,
    ProtocolError,
    decode_message,
    encode_message,
    p1_breach_reason,
    record_from_wire,
    record_key,
    record_to_wire,
)
from komugi.eval.v1runner.reference import (
    LoadedReference,
    ReferenceUnavailable,
    default_reference_root,
    load_reference,
    observation_shape,
)
from komugi.eval.v1runner.runner import (
    ArgumentError,
    HelpRequested,
    Runner,
    RunnerConfig,
    main,
    parse_args,
)

__all__ = [
    "EVAL_CEILING",
    "PROTOCOL_VERSION",
    "RULES_DIVERGENCE",
    "ArgumentError",
    "BudgetRefused",
    "CountedNet",
    "EvalCounter",
    "HelpRequested",
    "LoadedReference",
    "ProtocolError",
    "ReferenceUnavailable",
    "Runner",
    "RunnerConfig",
    "decode_message",
    "default_reference_root",
    "encode_message",
    "load_reference",
    "main",
    "observation_shape",
    "p1_breach_reason",
    "parse_args",
    "positions_in",
    "record_from_wire",
    "record_key",
    "record_to_wire",
]
