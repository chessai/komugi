"""The liveness digest: a hash that moves if and only if *learning* moved the net.

KOM-022 §3a. Three digests of a net are plausible and two of them are traps,
so this module states which one is the liveness digest and why the other two
are not.

**Not the checkpoint filename digest.** That covers ``meta_json``
(:mod:`komugi.train.checkpoint`), and the meta always carries ``iteration``
(:func:`komugi.train.step.training_iteration`), so a completely frozen net
publishes a *different* filename digest every iteration. Using it as the
liveness hash makes the v1 frozen-loop gate
(:mod:`komugi.telemetry.liveness`) a permanent pass -- the exact failure that
module exists to catch.

**Not the whole state dict, and not its floating-point half either.** The net
is BatchNorm throughout, and :meth:`komugi.train.step.TrainStep.run` runs it in
``train()`` mode: the *forward pass* advances every module's
``num_batches_tracked`` (int64) **and** its ``running_mean`` / ``running_var``
(float32). Both move ``steps_per_iteration`` times an iteration on a net that
is not learning at all -- a collapsed learning rate, detached gradients, an
optimiser step that no-ops -- so a digest containing either of them cannot
fail, whichever dtype predicate is used. The trainer states this as its own
contract: *"BatchNorm is the one piece of state a step mutates besides the
parameters"*.

**So: the optimiser-touched parameters, and nothing else.**
:func:`canonical_digest` walks exactly the keys of ``net.named_parameters()``
-- the set the optimiser is constructed over and the only set a training step
writes to. A net whose parameters are bit-identical across two iterations has
not learned, whatever its buffers did, and that is the statement the hash gate
must be able to make.

The split is **by name**, not by dtype and not by a name list, so it applies
to a ``Module`` and to a ``model_state`` loaded back from a checkpoint alike
(the anchor check and the ``self_play`` fingerprint both digest a *file*).
Deriving the key set from the live net rather than from a literal is what
makes a future module's trainable tensors digested automatically and its
buffers excluded automatically.

BatchNorm drift is not discarded -- it is a real signal, and a *stalled*
running statistic while parameters advance is one no other gate expresses.
:func:`bn_digest` gives it its own name, its own series and, deliberately, no
gate: it says "data flowed through the net in ``train()`` mode", which is a
different question from "did learning change the net" and must never be
allowed to answer it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Final, NamedTuple

import torch

_DIGEST_SEPARATOR: Final[bytes] = b"\0"

_LOW_BITS: Final[int] = 52
"""How many bits of a digest survive the trip through ``extras``.

:class:`~komugi.telemetry.records.IterationMetrics` extras are floats, and a
float64 carries 53 bits of integer mantissa exactly. 52 bits is one below
that, so ``float(value)`` is lossless and the recorded number is comparable
across iterations rather than approximately comparable.
"""

_LOW_MASK: Final[int] = (1 << _LOW_BITS) - 1


class Digest(NamedTuple):
    """A canonical hash over a named tensor set, plus how many scalars it covered.

    ``scalars`` is the shape check
    :class:`~komugi.telemetry.records.NetFingerprint` asks for: a hash collision
    is not a real concern, but hashing the *wrong* tensors is, and a parameter
    count that suddenly includes 28 BatchNorm running statistics is exactly the
    drift this ticket exists to prevent. It is *not* named ``count``: a
    :class:`~typing.NamedTuple` field of that name shadows ``tuple.count``,
    which strict mypy refuses outright.
    """

    sha256: str
    scalars: int


def parameter_keys(module: torch.nn.Module) -> tuple[str, ...]:
    """Return the sorted names of `module`'s optimiser-touched parameters.

    Sorted because dict ordering is not a contract and the digest below walks
    the keys in this order; derived from ``named_parameters()`` because that
    is precisely the set ``torch.optim`` is constructed over. Callers derive
    this **once per process** from the campaign's own net factory, so the same
    split applies to a live module and to a state dict read back from a file.
    """
    return tuple(sorted(name for name, _ in module.named_parameters()))


def _walk(
    feed: Callable[[bytes], None],
    state: Mapping[str, torch.Tensor],
    keys: Iterable[str],
) -> int:
    """Feed `keys` of `state` to `feed`, returning the scalars covered.

    Name, dtype, shape and raw bytes, each separated by a NUL, so two tensors
    that differ only in shape (a reshape that preserves the buffer) hash
    differently. The bytes are taken from a CPU, contiguous copy: a digest
    that depended on device placement or on stride would report a move that
    never happened.
    """
    covered = 0
    for key in keys:
        tensor = state[key]
        feed(key.encode("utf-8"))
        feed(_DIGEST_SEPARATOR)
        feed(str(tensor.dtype).encode("utf-8"))
        feed(_DIGEST_SEPARATOR)
        feed(repr(tuple(tensor.shape)).encode("utf-8"))
        feed(_DIGEST_SEPARATOR)
        feed(tensor.detach().to("cpu").contiguous().numpy().tobytes())
        covered += int(tensor.numel())
    return covered


def canonical_digest(
    state: Mapping[str, torch.Tensor], param_keys: Sequence[str]
) -> Digest:
    """Return the liveness digest of `state`: its parameters, and nothing else.

    `param_keys` is the key set from :func:`parameter_keys`. Every one of them
    must be present: a state dict missing a parameter is **not this net**, and
    digesting the subset that happens to be there would silently narrow the
    gate to whatever survived -- the failure mode where a gate keeps returning
    a verdict about less and less.

    Buffers are excluded by *being buffers*, i.e. by not appearing in
    `param_keys`. Not by dtype: the dtype predicate asks "is this a float?",
    which a running statistic answers yes to, while the parameter/buffer split
    asks "does the optimiser write to this?", which is the actual question.

    Raises:
        KeyError: If `state` is missing any key in `param_keys`, naming the
            missing ones.
    """
    missing = [key for key in param_keys if key not in state]
    if missing:
        raise KeyError(
            f"state dict is missing {len(missing)} parameter(s) the digest "
            f"covers ({missing[:4]}{'...' if len(missing) > 4 else ''}): this "
            f"is not the net the key set was derived from, and digesting the "
            f"subset that is present would silently narrow the liveness gate"
        )
    digest = hashlib.sha256()
    covered = _walk(digest.update, state, param_keys)
    return Digest(sha256=digest.hexdigest(), scalars=covered)


def bn_digest(
    state: Mapping[str, torch.Tensor], param_keys: Sequence[str]
) -> Digest:
    """Return the BatchNorm series' digest: the float buffers, never the parameters.

    The 28 ``running_mean`` / ``running_var`` tensors at the default b10c128
    shape -- everything floating-point in `state` that is *not* a parameter.
    Integer buffers (``num_batches_tracked``) are left out for the reason an
    exponentially-averaged counter is not a counter: their advance is
    ``steps_per_iteration`` by construction and carries no information this
    series does not already have.

    Its advance means **data flowed through the net in ``train()`` mode**. It
    says nothing whatever about learning, it is recorded and warned on, and it
    is never passed to :func:`~komugi.telemetry.liveness.self_play_net_advanced`
    -- being passed there is the defect this split exists to fix.

    An empty result is legitimate (a net with no BatchNorm), and the count is
    then zero; callers record it rather than treating it as a missing signal.
    """
    excluded = set(param_keys)
    keys = sorted(
        key
        for key, tensor in state.items()
        if key not in excluded and tensor.is_floating_point()
    )
    digest = hashlib.sha256()
    covered = _walk(digest.update, state, keys)
    return Digest(sha256=digest.hexdigest(), scalars=covered)


def digest_lo(sha256_hex: str) -> float:
    """Return a digest's low 52 bits as an exactly-representable float.

    The trick ``probe_sha256_lo`` uses: ``extras`` values must be finite
    floats, and 52 bits is the widest slice of a hash that survives the trip
    into one without rounding. Enough to notice a series that stopped moving;
    not a cryptographic claim, and never used as one.
    """
    return float(int(sha256_hex[-13:], 16) & _LOW_MASK)
