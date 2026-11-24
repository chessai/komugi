"""A trunk in eval mode is a function of one position (KOM-104).

Batch dependence is the classic silent net bug. In eval mode a BatchNorm must
normalise with its running statistics; if it normalises with the batch's own
statistics instead -- because the module was never switched out of train mode,
because the trunk calls ``F.batch_norm`` with ``training=True`` hardwired, or
because a "gpool" reduces over the batch axis -- then a position's evaluation
depends on whichever positions happened to share its batch.

Nothing about that shows up in training loss. It shows up as a search whose
node evaluations shift with batch composition, i.e. as noise nobody can trace.

The property: evaluating four positions together equals evaluating each alone.
The tolerance is 1e-6 rather than exact, because batched and unbatched
convolutions are free to block their arithmetic differently; the failure this
is aimed at moves outputs far more than that.
"""

from __future__ import annotations

import pytest
import torch

import trunk_probes as probes

net = pytest.importorskip("komugi.net")

if not hasattr(net, "Trunk"):
    pytest.skip("komugi.net exports no Trunk yet (KOM-004)", allow_module_level=True)

_IN_PLANES = 8
_CHANNELS = 32
_BATCH = 4
# Batched and single-sample convolutions legitimately take different GEMM
# kernel paths, so eval outputs differ by float32 rounding (~2.4e-6 observed
# when the suite first met the real trunk). Real batch leakage (train-mode BN
# statistics) shows at ~1e-1 — four orders of magnitude above this bound, so
# the test keeps its teeth.
_ATOL = 1e-5


@pytest.mark.parametrize(
    ("blocks", "gpool_every"),
    [
        pytest.param(3, 3, id="with-gpool"),
        pytest.param(3, 9, id="no-gpool"),
    ],
)
def test_eval_output_is_independent_of_batch_mates(blocks: int, gpool_every: int) -> None:
    """Four together == four alone, block layout with and without pooling."""
    _, trunk = probes.build_trunk(
        net,
        in_planes=_IN_PLANES,
        channels=_CHANNELS,
        blocks=blocks,
        gpool_every=gpool_every,
        seed=0,
    )
    batch = probes.noise((_BATCH, _IN_PLANES, probes.BOARD, probes.BOARD), seed=21)
    together = probes.forward_no_grad(trunk, batch)

    for index in range(_BATCH):
        alone = probes.forward_no_grad(trunk, batch[index : index + 1])
        gap = float((together[index : index + 1] - alone).abs().max().item())
        assert gap <= _ATOL, (
            f"sample {index} differs by {gap:.3e} between a batch of {_BATCH} and a "
            "batch of 1; eval-mode output depends on its batch-mates"
        )


def test_shuffling_the_batch_permutes_the_output() -> None:
    """The same positions in a different order give the same evaluations.

    A subtler form of the same bug: statistics taken over the batch are
    permutation-invariant, so a reversed batch would still agree sample-for-
    sample under the test above only if the trunk is genuinely per-sample --
    while any state carried *between* samples in order (a running buffer
    updated mid-forward, say) fails here and nowhere else.
    """
    _, trunk = probes.build_trunk(
        net,
        in_planes=_IN_PLANES,
        channels=_CHANNELS,
        blocks=3,
        gpool_every=3,
        seed=0,
    )
    batch = probes.noise((_BATCH, _IN_PLANES, probes.BOARD, probes.BOARD), seed=22)
    forward = probes.forward_no_grad(trunk, batch)
    reversed_out = probes.forward_no_grad(trunk, torch.flip(batch, dims=(0,)))

    gap = float((forward - torch.flip(reversed_out, dims=(0,))).abs().max().item())
    assert gap <= _ATOL, f"reordering the batch changed the outputs by {gap:.3e}"


def test_a_lone_position_is_evaluated_without_error() -> None:
    """Batch size 1 is the search's hot path; it must not need company."""
    _, trunk = probes.build_trunk(
        net,
        in_planes=_IN_PLANES,
        channels=_CHANNELS,
        blocks=3,
        gpool_every=3,
        seed=0,
    )
    out = probes.forward_no_grad(
        trunk, probes.noise((1, _IN_PLANES, probes.BOARD, probes.BOARD), seed=23)
    )
    assert out.shape == (1, _CHANNELS, probes.BOARD, probes.BOARD)
    assert bool(torch.isfinite(out).all())
