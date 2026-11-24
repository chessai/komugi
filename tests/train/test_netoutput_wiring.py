"""Wiring between :class:`KomugiNet`'s outputs and the loss assembler.

This file began as a strict-xfail tripwire (the wave-8 adversarial review's
F1): the assembler's soft-policy term read the ``policy_logits`` key while
``NetOutput`` carried a dedicated ``soft_policy_logits`` head, so enabling soft
policy trained the *main* policy head on the sharpened target -- fighting the
primary policy loss -- while ``SoftPolicyHead`` received no gradient at all,
silently and at full forward cost.

The rewire landed with KOM-030 (``losses.SOFT_POLICY_LOGITS`` plus
:func:`komugi.train.wiring.outputs_from_net`), the xfail XPASSed as designed,
and the marker is gone. What remains is the permanent regression test its
docstring promised: the soft-policy gradient lands on ``SoftPolicyHead`` and
nowhere near ``PolicyHead``, and the key mapping that makes that true is
pinned against a real forward pass rather than restated.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Final

import pytest
import torch

from komugi.modes import MODE_FAST, MODE_FULL
from komugi.net import HeadsConfig, KomugiNet, NetConfig, NetOutput, TrunkConfig
from komugi.train import losses
from komugi.train.wiring import fast_row_weights, outputs_from_net

_CHILD_TIMEOUT_S: Final[float] = 120.0
"""Wall clock a fresh-interpreter import case gets. Generous: the child pays
the full ``import torch`` cost, which is most of it."""


def _tiny_net() -> KomugiNet:
    """A one-block, 81-action net: big enough to wire, small enough to run."""
    trunk = TrunkConfig(in_planes=4, channels=8, blocks=1, gpool_every=3, seed=7)
    heads = HeadsConfig(channels=8, num_actions=81, policy_planes=1, seed=7)
    return KomugiNet(NetConfig(trunk=trunk, heads=heads))


def test_soft_policy_loss_reaches_the_soft_policy_head() -> None:
    """The soft-policy gradient must land on ``SoftPolicyHead``, not ``PolicyHead``.

    The promoted tripwire. Soft policy is the only enabled head, so every
    gradient in the net traces back to its term alone: any on ``PolicyHead``
    would mean the sharpened target had reached the main head again.
    """
    net = _tiny_net()
    out = net(torch.randn(2, 4, 9, 9))
    outputs = outputs_from_net(out)
    targets = {losses.POLICY_TARGET: torch.full((2, 81), 1.0 / 81.0)}
    assembler = losses.LossAssembler(
        losses.LossConfig(
            policy_enabled=False,
            policy_weight=0.0,
            soft_policy_enabled=True,
            soft_policy_weight=1.0,
            wdl_enabled=False,
            wdl_weight=0.0,
        )
    )

    total, _ = assembler(outputs, targets)
    total.backward()  # type: ignore[no-untyped-call]

    soft_grads = [p.grad for p in net.soft_policy.parameters()]
    assert soft_grads, "SoftPolicyHead has parameters"
    assert all(
        g is not None and bool((g.abs().sum() > 0).item()) for g in soft_grads
    ), "the soft-policy loss term produced no gradient on SoftPolicyHead"
    # ... and the sharpened target must not leak into the main policy head.
    assert all(
        g is None or bool((g.abs().sum() == 0).item())
        for g in (p.grad for p in net.policy.parameters())
    ), "the soft-policy loss term backpropagated into PolicyHead"


def test_the_main_policy_loss_still_reaches_only_the_policy_head() -> None:
    """The mirror image, so the rewire cannot have merely swapped the bug over."""
    net = _tiny_net()
    out = net(torch.randn(2, 4, 9, 9))
    targets = {losses.POLICY_TARGET: torch.full((2, 81), 1.0 / 81.0)}
    assembler = losses.LossAssembler(
        losses.LossConfig(
            policy_enabled=True,
            policy_weight=1.0,
            soft_policy_enabled=False,
            wdl_enabled=False,
        )
    )

    total, _ = assembler(outputs_from_net(out), targets)
    total.backward()  # type: ignore[no-untyped-call]

    assert all(
        p.grad is not None and bool((p.grad.abs().sum() > 0).item())
        for p in net.policy.parameters()
    )
    assert all(
        p.grad is None or bool((p.grad.abs().sum() == 0).item())
        for p in net.soft_policy.parameters()
    )


def test_outputs_from_net_maps_every_staged_head_to_the_key_the_loss_reads() -> None:
    """The key golden, against a real forward pass -- now all eight heads.

    Identity of the *tensors*, not equality of their values: the mapping is
    supposed to be a rename and nothing else, so an entry that happened to
    compare equal (the two policy heads on a degenerate input, say) would not
    prove the right one was picked. The staged heads make that stricter rather
    than looser: ``short_term_value`` and ``draft_outcome_logits`` are both
    ``(N, 3)`` float32, so a mix-up between them is shape-clean and equality
    would not see it.

    The key set is asserted **exactly**, in both directions. A head missing
    from ``outputs_from_net`` is a head that can never be trained, however
    correctly its loss branch is written; an extra key is a name a future typo
    could find a plausible tensor under.
    """
    net = _tiny_net()
    out = net(torch.randn(2, 4, 9, 9))
    assert isinstance(out, NetOutput)
    mapped = outputs_from_net(out)

    assert set(mapped) == {
        losses.POLICY_LOGITS,
        losses.SOFT_POLICY_LOGITS,
        losses.WDL_LOGITS,
        losses.MLH_PRED,
        losses.UNCERTAINTY_PRED,
        losses.OWNERSHIP_LOGITS,
        losses.SHORT_TERM_VALUE_PRED,
        losses.DRAFT_OUTCOME_LOGITS,
    }
    assert mapped[losses.POLICY_LOGITS] is out.policy_logits
    assert mapped[losses.SOFT_POLICY_LOGITS] is out.soft_policy_logits
    assert mapped[losses.WDL_LOGITS] is out.wdl_logits
    # The rename that used to reach nothing at all: NetOutput.moves_left is
    # spelled "mlh" on the loss side.
    assert mapped[losses.MLH_PRED] is out.moves_left
    assert losses.MLH_PRED == "mlh"
    assert mapped[losses.UNCERTAINTY_PRED] is out.uncertainty
    assert mapped[losses.OWNERSHIP_LOGITS] is out.ownership_logits
    assert mapped[losses.SHORT_TERM_VALUE_PRED] is out.short_term_value
    assert mapped[losses.DRAFT_OUTCOME_LOGITS] is out.draft_outcome_logits

    assert mapped[losses.POLICY_LOGITS].shape == (2, 81)
    assert mapped[losses.SOFT_POLICY_LOGITS].shape == (2, 81)
    assert mapped[losses.WDL_LOGITS].shape == (2, 3)
    assert mapped[losses.MLH_PRED].shape == (2,)
    assert mapped[losses.UNCERTAINTY_PRED].shape == (2,)
    assert mapped[losses.OWNERSHIP_LOGITS].shape == (2, 81, 3)
    assert mapped[losses.SHORT_TERM_VALUE_PRED].shape == (2, 3)
    assert mapped[losses.DRAFT_OUTCOME_LOGITS].shape == (2, 3)

    # Four of the four staged keys are spelled exactly as their NetOutput
    # fields, so `moves_left` stays the only rename this function performs.
    for key in (
        losses.UNCERTAINTY_PRED,
        losses.OWNERSHIP_LOGITS,
        losses.SHORT_TERM_VALUE_PRED,
        losses.DRAFT_OUTCOME_LOGITS,
    ):
        assert mapped[key] is getattr(out, key)


def test_a_soft_enabled_config_now_demands_the_soft_key() -> None:
    """The loud break the rewire was allowed to cause, pinned as behaviour.

    A caller still handing the assembler only ``policy_logits`` for a
    soft-enabled config gets a ``KeyError`` naming the key it needs -- the
    failure mode the old wiring had instead of a silent, wrong gradient.
    """
    outputs = {losses.POLICY_LOGITS: torch.zeros((1, 4))}
    targets = {losses.POLICY_TARGET: torch.full((1, 4), 0.25)}
    assembler = losses.LossAssembler(
        losses.LossConfig(
            policy_enabled=False,
            soft_policy_enabled=True,
            soft_policy_weight=1.0,
            wdl_enabled=False,
        )
    )
    with pytest.raises(KeyError, match=losses.SOFT_POLICY_LOGITS):
        assembler(outputs, targets)


# --------------------------------------------------------------------------
# fast_row_weights: the `search_mode` -> multiplier translation (KOM-036 §1.2)
# --------------------------------------------------------------------------


def test_fast_row_weights_are_one_for_full_and_w_for_fast() -> None:
    """Acceptance #6: the mapping, over a tensor carrying both modes.

    ``MODE_FULL`` / ``MODE_FAST`` are imported from :mod:`komugi.modes` -- the
    root leaf -- and **not** restated as ``0``/``1`` here, and not imported
    from ``komugi.selfplay.worker`` (an import cycle) nor from a
    ``komugi.selfplay.modes`` (the same cycle at package granularity). A
    mapping keyed on the wrong mode is what this node is red against.
    """
    modes = torch.tensor([MODE_FULL, MODE_FAST, MODE_FAST, MODE_FULL])
    weights = fast_row_weights(modes, 0.125)

    assert tuple(weights.shape) == (4,)
    assert weights.tolist() == [1.0, 0.125, 0.125, 1.0]
    # Arm B, and arm C: the same code path, one different float.
    assert fast_row_weights(modes, 0.0).tolist() == [1.0, 0.0, 0.0, 1.0]
    assert fast_row_weights(modes, 1.0).tolist() == [1.0, 1.0, 1.0, 1.0]


def test_an_unknown_search_mode_is_refused() -> None:
    """Acceptance #7: a third mode raises rather than reading as a full row.

    Silently taking the full-row weight would train an unknown stratum at a
    weight nobody chose -- and would do it invisibly, since the arm's whole
    observable effect is a number in a loss curve.
    """
    with pytest.raises(ValueError, match="search_mode"):
        fast_row_weights(torch.tensor([MODE_FULL, 2]), 0.125)
    with pytest.raises(ValueError, match="search_mode"):
        fast_row_weights(torch.tensor([-1, MODE_FAST]), 0.125)
    with pytest.raises(ValueError, match="1-D"):
        fast_row_weights(torch.zeros((2, 2), dtype=torch.int64), 0.125)


_IMPORT_CASES: Final[dict[str, str]] = {
    "selfplay-first": (
        "import komugi.selfplay\n"
        "import torch\n"
        "from komugi.train.wiring import fast_row_weights\n"
        "assert fast_row_weights(torch.tensor([0, 1]), 0.5).tolist() == [1.0, 0.5]\n"
    ),
    "train-first": (
        "import komugi.train\n"
        "import torch\n"
        "from komugi.train.wiring import fast_row_weights\n"
        "assert fast_row_weights(torch.tensor([0, 1]), 0.5).tolist() == [1.0, 0.5]\n"
    ),
    "leaf-at-package-granularity": (
        "import sys\n"
        "import komugi.train\n"
        "assert 'komugi.selfplay' not in sys.modules, sorted(\n"
        "    name for name in sys.modules if name.startswith('komugi')\n"
        ")\n"
    ),
}
"""The three fresh-interpreter cases of acceptance #6a.

Each **must** run in a child interpreter: in-process pytest ordering hides
every one of these bugs, because by the time this module is collected some
other test has already imported both packages in some order.
"""


@pytest.mark.parametrize("case", sorted(_IMPORT_CASES))
def test_fast_row_weights_imports_cleanly_from_either_direction(case: str) -> None:
    """Acceptance #6a: the leaf is a leaf, in a fresh interpreter, both ways.

    * **selfplay-first** is red against rev 1's design, which imported the mode
      codes from ``komugi.selfplay.worker``: entering through
      ``import komugi.selfplay`` freezes ``records`` at its
      ``from komugi.train.shards import ...`` line, and ``worker.py`` then asks
      that half-built module for ``MLH_SCALE`` -- ``ImportError``, at
      collection, across most of ``tests/selfplay/**``.
    * **train-first** is the direction that catches the *package*-level edge. A
      ``komugi/selfplay/modes.py`` placement passes it only because
      ``train/__init__`` happens to load ``shards`` before ``wiring``, and
      breaks on any reorder of those two lines.
    * **leaf-at-package-granularity** is the strictly stronger property, and is
      available only under the ``src/komugi/modes.py`` placement: importing the
      trainer must not drag in the self-play package at all -- no
      ``shared_memory``, no ``resource_tracker``, no ``orchestrate``. It pins
      the property the language enforces rather than a property of one file's
      source text.
    """
    # The child gets this interpreter's own import path, spelled explicitly:
    # nothing here is pip-installed, so a bare `python -c` would not find the
    # package the parent is testing.
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        path for path in sys.path if path
    )
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_CASES[case]],
        capture_output=True,
        text=True,
        timeout=_CHILD_TIMEOUT_S,
        env=environment,
    )
    assert result.returncode == 0, (
        f"the {case} import case failed:\n{result.stdout}\n{result.stderr}"
    )
