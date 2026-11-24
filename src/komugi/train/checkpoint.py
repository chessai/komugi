"""Checkpoints that are either wholly there or not there at all.

A checkpoint is a model state dict, an optimiser state dict, a step number and
a JSON-serialisable metadata mapping. The interesting part is not what is
stored but the guarantee about *when* it becomes visible.

The commit is the rename, and the name carries the checksum
-------------------------------------------------------------
:func:`save_checkpoint` serialises to memory, writes the bytes to a
dot-prefixed temporary file, ``fsync``-s it, and only then
:func:`os.replace`-s it to::

    checkpoint-<step:09d>-<digest16>.pt

where ``digest16`` is the first 16 hex characters (64 bits) of ``sha256`` over
the exact file bytes. The directory is ``fsync``-ed afterwards so the new name
itself is durable.

Putting the digest in the *name* rather than inside the file makes the
integrity check independent of the serialiser: verification is "hash the bytes
you found, compare with the name you found them under". A file truncated by a
full disk after the rename, a byte flipped by bad memory, a half-copied
transfer -- all three change the bytes without changing the name, and all
three are caught before a single tensor is unpickled. A file that never got
renamed is invisible to the loader entirely, because the temporary name is
dot-prefixed and outside the loader's glob.

The plane map is stamped by the codec, not by the caller
--------------------------------------------------------
:func:`save_checkpoint` fills ``in_planes`` and ``plane_map_version`` into
`meta` whenever the caller has not (:func:`plane_meta`), so that the guard at
the load sites -- :func:`require_matching_plane_count` -- is a statement about
*old files* rather than about which writer remembered. It is the writers'
invariant because there are many of them and one reader: the trainer, the EMA
publisher self-play actually serves, an anchor pin, and every test rig. A value
the caller *did* state is never overwritten; that is what keeps the refusal
falsifiable.

Crash-safety invariants
-----------------------
1. **No half-written checkpoint is ever loadable.** A reader either sees the
   final name (bytes complete and fsync'd before the rename) or does not see
   the file at all.
2. **A damaged checkpoint costs that checkpoint, not the run.**
   :func:`load_checkpoint` walks candidates newest-first and skips any whose
   digest, structure or deserialisation fails, emitting a
   :class:`CheckpointWarning` per skip, and returns the newest one that is
   whole. A torn newest checkpoint therefore costs one save interval, not the
   training run.
3. **Saving never destroys a good checkpoint before a better one exists.**
   Superseding files for the same step are unlinked only *after* the new one
   has been published and the directory fsync'd, so the window in which the
   step has no valid file is empty.
4. **Nothing here executes what it reads.** ``torch.load`` runs with
   ``weights_only=True``, and metadata travels as a JSON string rather than a
   pickled object, so a hostile or corrupt checkpoint is a parse failure
   rather than a code path. That refusal is itself survivable: the
   ``UnpicklingError`` it raises is in :data:`_LOAD_ERRORS`, so a checkpoint
   carrying an object the allowlist rejects is skipped like any other
   unusable file rather than taking the run down with it.

What is *not* claimed: that 64 bits of digest detects an adversarially crafted
collision. It is an integrity check against truncation and bit rot, not a
signature. And the serialised bytes are held in memory once, in full, during
save and load -- fine at the model sizes in §2.1, and the price of hashing
the bytes before they are published.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch

from komugi.engine.observation import PLANE_MAP_VERSION, observation_size

FORMAT_VERSION: Final[int] = 1
"""Stamped inside every checkpoint; a mismatch is a skip, not a crash."""

IN_PLANES_KEY: Final[str] = "in_planes"
"""The meta key naming the plane count a checkpoint's weights were written for."""

PLANE_MAP_VERSION_KEY: Final[str] = "plane_map_version"
"""The meta key naming the encoder's plane map (KOM-002 §5.3)."""

CHECKPOINT_GLOB: Final[str] = "checkpoint-*.pt"
_CHECKPOINT_RE: Final[re.Pattern[str]] = re.compile(
    r"checkpoint-(\d{9,})-([0-9a-f]{16})\.pt"
)
_DIGEST_CHARS: Final[int] = 16
_CHECKPOINT_MODE: Final[int] = 0o644

_LOAD_ERRORS: Final[tuple[type[Exception], ...]] = (
    OSError,
    ValueError,
    KeyError,
    TypeError,
    RuntimeError,
    EOFError,
    pickle.UnpicklingError,
)
"""What a damaged checkpoint throws on the way back in.

``RuntimeError`` is torch's own deserialisation failure, ``ValueError`` covers
the digest and JSON checks, ``TypeError`` a payload whose members are not the
shapes we expect. ``pickle.UnpicklingError`` is what ``weights_only=True``
raises when the stream asks for a global it will not build -- the *expected*
outcome for a checkpoint whose optimiser state picked up an object outside
torch's allowlist, and one that must cost that checkpoint rather than the run
(it is not a subclass of any of the others, so without this entry it would
escape :func:`load_checkpoint` as a crash). Enumerated rather than a bare
``Exception`` so a bug in this module still surfaces as a traceback.
"""


class CheckpointWarning(UserWarning):
    """A checkpoint file was skipped at load time.

    Its own category so a trainer can escalate it -- "resume from the newest
    or stop" is a legitimate policy -- without catching unrelated warnings.
    """


@dataclass(frozen=True, slots=True)
class CheckpointRef:
    """A checkpoint file on disk: its path, its step and its embedded digest."""

    path: Path
    step: int
    digest: str


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """A loaded checkpoint.

    Frozen guards the *bindings*, not the dicts: ``model_state`` and
    ``optim_state`` are ordinary mutable mappings, because that is what
    ``load_state_dict`` wants and copying them defensively would double peak
    memory at exactly the moment it is highest.
    """

    step: int
    model_state: dict[str, torch.Tensor]
    optim_state: dict[str, Any]
    meta: dict[str, Any]
    path: Path


def _fsync_directory(directory: Path) -> None:
    """``fsync`` a directory so a rename into it is durable, not just atomic."""
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _digest_of(data: bytes) -> str:
    """Return the first :data:`_DIGEST_CHARS` hex characters of ``sha256(data)``."""
    return hashlib.sha256(data).hexdigest()[:_DIGEST_CHARS]


def list_checkpoints(directory: str | os.PathLike[str]) -> tuple[CheckpointRef, ...]:
    """Return the checkpoint files in `directory`, newest first.

    Ordering is by step descending; ties (the same step saved twice, which
    :func:`save_checkpoint` normally prevents by superseding) break on the
    digest string, so the order is total and deterministic rather than
    filesystem-dependent. Files whose names do not parse are not checkpoints
    and are ignored silently -- a stray ``checkpoint-notes.txt`` is not
    corruption. A missing directory reads as empty.
    """
    path = Path(directory)
    refs: list[CheckpointRef] = []
    if not path.is_dir():
        return ()
    for candidate in path.glob(CHECKPOINT_GLOB):
        match = _CHECKPOINT_RE.fullmatch(candidate.name)
        if match is None:
            continue
        refs.append(
            CheckpointRef(
                path=candidate, step=int(match.group(1)), digest=match.group(2)
            )
        )
    return tuple(sorted(refs, key=lambda ref: (ref.step, ref.digest), reverse=True))


_STEM_WEIGHT_SUFFIX: Final[str] = "trunk.stem.weight"
"""The parameter whose second axis IS the plane count of a saved net.

The trunk's stem is ``nn.Conv2d(in_planes, channels, 3, ...)``, so its weight is
``(channels, in_planes, 3, 3)``. Matched on the *suffix* because a state dict
may arrive prefixed -- ``_orig_mod.`` under ``torch.compile``, ``module.``
under a data-parallel wrapper -- and a prefix does not change which tensor this
is.
"""


def stamped_in_planes(model_state: Mapping[str, Any] | None = None) -> int:
    """Return the plane count to stamp for `model_state`'s weights.

    Read from the **saved net's own stem** when the state dict carries one, and
    from the running encoder otherwise. The distinction is the whole point
    (review F5): ``in_planes`` in a checkpoint's meta is a claim about *those
    weights*, and :func:`require_matching_plane_count` refuses on it before
    ``load_state_dict`` runs. Stamping the process's number onto an 8-plane
    scaling net's weights makes that claim false, the guard then passes, and the
    load dies in torch with the ``size mismatch for stem.weight`` traceback the
    guard exists to replace.

    The encoder fallback covers state dicts that have no trunk at all -- a bare
    ``nn.Linear`` in a codec test, an optimiser-only artefact -- where there is
    no better answer than "the map this process encodes under", which is what
    the key meant before.
    """
    if model_state is not None:
        for key, value in model_state.items():
            if not key.endswith(_STEM_WEIGHT_SUFFIX):
                continue
            if isinstance(value, torch.Tensor) and value.ndim == 4:
                return int(value.shape[1])
    return int(observation_size()[0])


def plane_meta(model_state: Mapping[str, Any] | None = None) -> dict[str, int]:
    """Return the plane-map facts every checkpoint written today must carry.

    ``in_planes`` describes the WEIGHTS (:func:`stamped_in_planes`);
    ``plane_map_version`` describes the **running encoder**, because a plane
    map is a property of the process that produced the observations those
    weights were trained on and no tensor shape can carry it. That asymmetry is
    deliberate: the version is exactly the fact a shape cannot see (KOM-002 §4),
    which is why :func:`require_matching_plane_count` checks both.
    """
    return {
        IN_PLANES_KEY: stamped_in_planes(model_state),
        PLANE_MAP_VERSION_KEY: PLANE_MAP_VERSION,
    }


def _stamped(
    meta: Mapping[str, Any], model_state: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return `meta` with the plane-map keys filled in where they are absent.

    ``setdefault``, never overwrite: a caller that states a plane count -- a
    test forging a 118-plane file, a tool re-publishing an older checkpoint's
    metadata -- is making a claim about *those* weights, and silently replacing
    it with this process's would turn the refusal in
    :func:`require_matching_plane_count` into a rubber stamp.
    """
    stamped = dict(meta)
    for key, value in plane_meta(model_state).items():
        stamped.setdefault(key, value)
    return stamped


def _nonfinite_tensor_path(value: object, path: str) -> str | None:
    """Return a description of the first non-finite floating tensor in `value`.

    Walks the nested containers an optimiser state dict is actually made of --
    ``{'state': {0: {'exp_avg': tensor, ...}}, 'param_groups': [...]}`` -- so a
    NaN in a moment estimate is found, not only one in a parameter. ``None``
    when everything under `value` is finite (or is not a floating tensor: step
    counters, integer buffers and plain scalars have nothing to check).
    """
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            return path
        return None
    if isinstance(value, Mapping):
        for key, item in value.items():
            found = _nonfinite_tensor_path(item, f"{path}[{key!r}]")
            if found is not None:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _nonfinite_tensor_path(item, f"{path}[{index}]")
            if found is not None:
                return found
    return None


def _refuse_nonfinite_state(
    model_state: Mapping[str, Any], optim_state: Mapping[str, Any]
) -> None:
    """Refuse a checkpoint whose weights or optimiser moments are non-finite.

    Enforced on **both** ends of the codec, which is the point (review TS-01).
    On the way out, a poisoned checkpoint is never published: a training step
    that manages to write NaN into the net -- the refusals in
    :mod:`komugi.train.step` are the first line, this is the second -- does not
    get to leave a file behind that the next resume, the EMA publisher or the
    campaign's gate would read as this iteration's weights. On the way back in,
    such a file is treated exactly as a digest failure is: skipped with a
    :class:`CheckpointWarning`, with the newest-valid walk-back continuing past
    it, so a poisoned file that reached the directory some other way (an older
    build, a hand-written artefact, a half-corrupt copy) can never be the
    checkpoint a run selects.

    Raises:
        ValueError: Naming the state dict and the key path of the offender.
    """
    labelled: tuple[tuple[str, Mapping[str, Any]], ...] = (
        ("model state", model_state),
        ("optimiser state", optim_state),
    )
    for label, state in labelled:
        found = _nonfinite_tensor_path(state, label)
        if found is not None:
            raise ValueError(
                f"non-finite value in {found}: a checkpoint carrying NaN or inf "
                f"weights is not a checkpoint, it is the wreckage of one -- "
                f"resuming from it trains nothing and publishes nothing "
                f"measurable (review TS-01)"
            )


def save_checkpoint(
    directory: str | os.PathLike[str],
    step: int,
    model_state: Mapping[str, torch.Tensor],
    optim_state: Mapping[str, Any],
    meta: Mapping[str, Any],
) -> Path:
    """Write a checkpoint atomically and return the path it was published under.

    `meta` must be JSON-serialisable; it is stored as canonical JSON so that
    reading it back needs no unpickler and the bytes are reproducible.
    A non-serialisable value raises ``TypeError`` *before* anything is written,
    so a bad metadata field cannot leave a temporary file behind.

    **``in_planes`` and ``plane_map_version`` are stamped here** when `meta`
    does not already carry them (:func:`plane_meta`) -- ``in_planes`` read off
    `model_state`'s own stem, so the stamp describes the weights in this file
    rather than the process that wrote it -- which makes every writer
    -- the trainer, the EMA publisher, an anchor pin, a test rig -- produce a
    checkpoint that :func:`require_matching_plane_count` accepts, without each
    of them having to remember. The alternative, asking every call site to pass
    the two keys, is the arrangement that shipped an EMA series no load site
    would read back: the guard fired on files this repository had just written,
    inside a spawned inference server, as an exit code. A writer that forgets is
    not a survivable failure mode, so the writer is not asked to remember.

    Any previously published checkpoint for the same `step` is removed after
    the new one is durable -- last write wins, with no interval in which the
    step has no valid file.

    Raises:
        ValueError: If `step` is negative, or if any floating tensor in
            `model_state` or `optim_state` is non-finite
            (:func:`_refuse_nonfinite_state`). Checked before the temporary
            file exists, so a refused save leaves the directory untouched.
        TypeError: If `meta` is not JSON-serialisable.
    """
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    # Before the temporary file, for the reason the metadata check is before it:
    # a refusal must not leave anything on disk to be found later.
    _refuse_nonfinite_state(model_state, optim_state)
    meta_json = json.dumps(
        _stamped(meta, model_state), sort_keys=True, separators=(",", ":")
    )

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "step": step,
        "meta_json": meta_json,
        "model": dict(model_state),
        "optim": dict(optim_state),
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    data = buffer.getvalue()
    digest = _digest_of(data)

    final_path = target / f"checkpoint-{step:09d}-{digest}.pt"
    tmp_path = target / f".{final_path.name}.tmp-{os.getpid()}"

    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _CHECKPOINT_MODE)
    try:
        written = 0
        while written < len(data):
            written += os.write(fd, data[written:])
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp_path, final_path)
    _fsync_directory(target)

    superseded = [
        ref
        for ref in list_checkpoints(target)
        if ref.step == step and ref.path != final_path
    ]
    if superseded:
        for ref in superseded:
            ref.path.unlink(missing_ok=True)
        _fsync_directory(target)
    return final_path


def _decode(ref: CheckpointRef) -> Checkpoint:
    """Verify and deserialise one checkpoint file, or raise.

    Digest first: the bytes are checked against the name before torch is asked
    to interpret them, so corruption is reported as corruption rather than as
    whatever the unpickler happens to trip over. Numerical integrity last, once
    there are tensors to look at: a file whose weights or optimiser moments are
    non-finite is INVALID, in exactly the sense a bad digest is, and fails with
    the ``ValueError`` that makes :func:`load_checkpoint` skip it and keep
    walking (review TS-01).
    """
    data = ref.path.read_bytes()
    actual = _digest_of(data)
    if actual != ref.digest:
        raise ValueError(
            f"digest mismatch: file hashes to {actual}, name claims {ref.digest} "
            f"({len(data)} bytes -- a truncated or damaged write)"
        )
    raw: object = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise TypeError(f"checkpoint payload is a {type(raw).__name__}, expected a dict")

    version = raw.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"checkpoint format version {version!r}, expected {FORMAT_VERSION}"
        )
    step = raw.get("step")
    if not isinstance(step, int) or isinstance(step, bool):
        raise TypeError(f"checkpoint step is {step!r}, expected an int")
    if step != ref.step:
        raise ValueError(f"checkpoint says step {step}, filename says {ref.step}")

    meta_json = raw.get("meta_json")
    if not isinstance(meta_json, str):
        raise TypeError(f"checkpoint meta_json is a {type(meta_json).__name__}")
    meta: object = json.loads(meta_json)
    if not isinstance(meta, dict):
        raise TypeError("checkpoint meta is not a JSON object")

    model = raw.get("model")
    if not isinstance(model, dict):
        raise TypeError("checkpoint model state is not a dict")
    model_state: dict[str, torch.Tensor] = {}
    for key, value in model.items():
        if not isinstance(key, str):
            raise TypeError(f"model state has a non-string key {key!r}")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"model state entry {key!r} is not a tensor")
        model_state[key] = value

    optim = raw.get("optim")
    if not isinstance(optim, dict):
        raise TypeError("checkpoint optimiser state is not a dict")
    optim_state: dict[str, Any] = {}
    for key, value in optim.items():
        if not isinstance(key, str):
            raise TypeError(f"optimiser state has a non-string key {key!r}")
        optim_state[key] = value

    # Same refusal as the writer's, on the way back in: a file whose weights are
    # NaN is unusable for exactly the reason a truncated one is, so it fails
    # here as a ValueError and is skipped by the same walk-back (review TS-01).
    _refuse_nonfinite_state(model_state, optim_state)

    return Checkpoint(
        step=step,
        model_state=model_state,
        optim_state=optim_state,
        meta=dict(meta),
        path=ref.path,
    )


def load_checkpoint(
    directory: str | os.PathLike[str],
) -> Checkpoint | None:
    """Load the newest checkpoint in `directory` that survives verification.

    Walks candidates newest-first; the first one whose digest matches, whose
    payload decodes and whose tensors are finite is returned. Each rejection
    emits a :class:`CheckpointWarning` naming the file and the reason -- a torn
    write must be *loud* even though it is survivable, because the silent
    version of this is a run that quietly resumes an hour behind where it
    thought it was.

    Returns ``None`` when the directory holds no checkpoint at all, or none
    that verifies. ``None`` is "start from scratch"; distinguishing it from
    "everything was corrupt" is what :func:`list_checkpoints` is for.
    """
    for ref in list_checkpoints(directory):
        try:
            return _decode(ref)
        except _LOAD_ERRORS as exc:
            warnings.warn(
                f"skipping unusable checkpoint {ref.path.name}: {exc}",
                CheckpointWarning,
                stacklevel=2,
            )
    return None


def load_checkpoint_file(path: str | os.PathLike[str]) -> Checkpoint | None:
    """Load exactly the checkpoint file `path` if it verifies.

    The companion to :func:`load_checkpoint` for the caller who has *pinned* a
    checkpoint -- a rollback, a comparison against an older iteration -- where
    silently substituting a newer sibling from the same directory would defeat
    the pin. Verification is the same digest walk; only the selection differs.

    Returns ``None`` when `path` is not a listed checkpoint in its directory
    or does not verify (with the same :class:`CheckpointWarning` a directory
    walk would emit).
    """
    target = Path(path)
    for ref in list_checkpoints(target.parent):
        if ref.path != target:
            continue
        try:
            return _decode(ref)
        except _LOAD_ERRORS as exc:
            warnings.warn(
                f"skipping unusable checkpoint {ref.path.name}: {exc}",
                CheckpointWarning,
                stacklevel=2,
            )
        return None
    return None


def require_matching_plane_count(checkpoint: Checkpoint, in_planes: int) -> None:
    """Refuse, BY NAME, a checkpoint whose weights are shaped for other planes.

    The trunk's stem is ``nn.Conv2d(in_planes, channels, 3, ...)``, so its weight
    is ``(channels, in_planes, 3, 3)``. Loading a 118-plane checkpoint into a
    128-plane net therefore cannot work, and there is no honest migration: the
    missing input channels have no learned weights, and zero-filling them would
    ship a net whose stem was calibrated at a different fan-in.

    So this does not rescue anything. It replaces a raw torch ``size mismatch
    for stem.weight`` traceback with a message that names both plane counts and
    the ticket that broke them apart.

    **Call this at the load SITE, on the object a loader returned -- never from
    inside :func:`load_checkpoint`, :func:`load_checkpoint_file` or
    :func:`_decode`.** Every member of :data:`_LOAD_ERRORS` raised in there is
    caught, downgraded to a :class:`CheckpointWarning` and skipped past; a
    refusal placed there would turn into "warn, try the next candidate, return
    ``None``", and every caller reads ``None`` as "start from scratch". The
    campaign would look like it resumed, with an untrained net.

    **Both keys are checked, and the second one is the one a shape cannot see.**
    The plane map reserves planes 121-127 (KOM-002 §2), so a future ticket that
    fills plane 121 leaves ``in_planes`` at 128 and changes what every plane
    *means*: weights trained under map version 2 would load without complaint
    and read a different board. ``plane_map_version`` is the eight bytes that
    close that gap on the checkpoint side, exactly as
    :func:`komugi.selfplay.records._check_plane_map_version` closes it on the
    shard side -- and it is only worth stamping if it is read.

    A checkpoint written before KOM-002 carries neither key. That is refused
    with the same error: absence must not read as agreement. The refusal stays
    sharp precisely because :func:`save_checkpoint` stamps both keys on the way
    out (:func:`plane_meta`): a file that reaches here carrying neither was
    written by a binary that predates the plane map, not by a call site that
    forgot to pass them.

    Args:
        checkpoint: What :func:`load_checkpoint` / :func:`load_checkpoint_file`
            returned. A ``None`` return means "no checkpoint at all" and keeps
            its existing meaning -- do not pass it here.
        in_planes: The plane count of the net that is about to receive the
            weights.

    Raises:
        ValueError: If the checkpoint's recorded ``in_planes`` differs, if it
            records none, or if its ``plane_map_version`` is not the running
            encoder's (absent included).
    """
    recorded = checkpoint.meta.get(IN_PLANES_KEY)
    if recorded is None:
        raise ValueError(
            f"checkpoint {checkpoint.path.name} records no 'in_planes' in its "
            "meta, so it predates KOM-002 (input planes v2) and was written "
            f"for a plane map this net does not have ({in_planes} planes). "
            "Old checkpoints are deliberately not loadable across a plane-map "
            "change: the stem's weight is (channels, in_planes, 3, 3) and the "
            "added input channels have no learned values. Start from a fresh "
            "root."
        )
    if int(recorded) != in_planes:
        raise ValueError(
            f"checkpoint {checkpoint.path.name} was written for "
            f"{int(recorded)} input planes but this net has {in_planes} "
            "(KOM-002, input planes v2). The stem's weight is "
            "(channels, in_planes, 3, 3), so these weights cannot be loaded "
            "and there is no migration that would not silently ship a "
            "mis-calibrated stem. Start from a fresh root."
        )

    version = checkpoint.meta.get(PLANE_MAP_VERSION_KEY)
    if version is None or int(version) != PLANE_MAP_VERSION:
        stated = "no plane_map_version" if version is None else f"version {int(version)}"
        raise ValueError(
            f"checkpoint {checkpoint.path.name} records {stated} but this "
            f"encoder writes plane map version {PLANE_MAP_VERSION} "
            "(KOM-002, input planes v2). The plane COUNT agrees "
            f"({in_planes}), which is exactly why the count cannot see this: "
            "the reserved planes 121-127 let a later map change what a plane "
            "MEANS without changing how many there are, so weights trained "
            "under the older map would load silently and read a different "
            "board. Start from a fresh root."
        )
