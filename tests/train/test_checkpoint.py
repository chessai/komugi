"""A checkpoint is whole or absent, and a torn one costs one save interval.

Three properties:

* round-trip -- model tensors, optimiser state and metadata come back exactly,
  and the returned state dicts are what ``load_state_dict`` wants;
* atomicity -- the committed name carries a digest of the committed bytes, and
  no in-progress file is ever visible under a checkpoint name;
* torn-write recovery -- truncating the newest checkpoint makes the loader warn
  and fall back to the previous one, rather than raising or, worse, resuming
  from half a tensor.
"""

from __future__ import annotations

import datetime
import hashlib
import io
import json
import pickle
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import pytest
import torch
from torch import nn

from komugi.campaign import factory
from komugi.engine.observation import PLANE_MAP_VERSION, observation_size
from komugi.net import HeadsConfig, KomugiNet, NetConfig, TrunkConfig
from komugi.modes import MODE_FULL
from komugi.terminations import TERMINATION_DECISIVE
from komugi.train import checkpoint as checkpoint_module
from komugi.train.checkpoint import (
    CHECKPOINT_GLOB,
    FORMAT_VERSION,
    Checkpoint,
    CheckpointWarning,
    list_checkpoints,
    load_checkpoint,
    plane_meta,
    require_matching_plane_count,
    save_checkpoint,
)
from komugi.train.ema import ema_meta, publish_ema
from komugi.train.losses import LossAssembler
from komugi.train.step import StepConfig, TrainStep, training_iteration
from komugi.train.wiring import P0A_RECIPE


def _tiny_model(seed: int) -> nn.Linear:
    """Return a 3->2 linear layer with deterministic weights."""
    torch.manual_seed(seed)
    return nn.Linear(3, 2)


def _trained_optimiser(model: nn.Module) -> torch.optim.Optimizer:
    """Return an SGD-with-momentum whose state dict is non-trivial.

    A freshly constructed optimiser has empty per-parameter state, which would
    make the round-trip assertion vacuous; one step populates the momentum
    buffers so there are real tensors to lose.
    """
    optimiser = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    output = model(torch.ones(1, 3))
    loss = output.sum()
    loss.backward()
    optimiser.step()
    return optimiser


def _save(directory: Path, step: int, seed: int) -> tuple[Path, nn.Linear]:
    """Save a checkpoint at `step` from a model seeded with `seed`."""
    model = _tiny_model(seed)
    optimiser = _trained_optimiser(model)
    path = save_checkpoint(
        directory,
        step,
        model.state_dict(),
        optimiser.state_dict(),
        {"lr": 0.1, "iteration": step, "note": "unit test"},
    )
    return path, model


def test_round_trip_is_exact(tmp_path: Path) -> None:
    """Every tensor, every optimiser buffer and every metadata field survives."""
    model = _tiny_model(0)
    optimiser = _trained_optimiser(model)
    meta = {"lr": 0.1, "iteration": 3, "tags": ["a", "b"], "ok": True}
    save_checkpoint(tmp_path, 3, model.state_dict(), optimiser.state_dict(), meta)

    loaded = load_checkpoint(tmp_path)
    assert isinstance(loaded, Checkpoint)
    assert loaded.step == 3
    # Every field the caller gave, verbatim, plus the plane-map keys the codec
    # stamps on every writer's behalf (KOM-002 §5.3) -- and nothing else.
    assert loaded.meta == {**meta, **plane_meta()}

    assert set(loaded.model_state) == set(model.state_dict())
    for name, tensor in model.state_dict().items():
        assert torch.equal(loaded.model_state[name], tensor), name

    # The loaded state dicts are usable, not merely comparable.
    revived = nn.Linear(3, 2)
    revived.load_state_dict(loaded.model_state)
    revived_optimiser = torch.optim.SGD(revived.parameters(), lr=0.5, momentum=0.9)
    revived_optimiser.load_state_dict(loaded.optim_state)
    assert revived_optimiser.param_groups[0]["lr"] == pytest.approx(0.1)
    assert torch.equal(revived(torch.ones(1, 3)), model(torch.ones(1, 3)))


def test_the_committed_name_carries_a_digest_of_the_committed_bytes(
    tmp_path: Path,
) -> None:
    """Integrity is checkable without opening the file: hash it, read the name."""
    path, _ = _save(tmp_path, 7, seed=1)
    assert path.name.startswith("checkpoint-000000007-")
    digest = path.name.rsplit("-", 1)[1].removesuffix(".pt")
    assert len(digest) == 16
    assert hashlib.sha256(path.read_bytes()).hexdigest()[:16] == digest


def test_no_temporary_file_is_visible_under_a_checkpoint_name(tmp_path: Path) -> None:
    """Only the published name matches the loader's glob."""
    _save(tmp_path, 1, seed=1)
    _save(tmp_path, 2, seed=2)
    visible = sorted(path.name for path in tmp_path.glob(CHECKPOINT_GLOB))
    assert len(visible) == 2
    assert all(not name.startswith(".") for name in visible)
    assert all(".tmp" not in name for name in visible)
    # And nothing else was left behind at all.
    assert sorted(path.name for path in tmp_path.iterdir()) == visible


def test_a_torn_newest_checkpoint_falls_back_to_the_previous_one(
    tmp_path: Path,
) -> None:
    """The invariant the trainer depends on: a bad newest costs one interval.

    Truncation is the honest simulation of a write that died mid-flight after
    the rename -- a full disk, a killed copy, a bad block. The file's bytes no
    longer hash to the digest in its own name, so it is rejected before torch
    is asked to interpret them.
    """
    _, previous = _save(tmp_path, 1, seed=11)
    newest, _ = _save(tmp_path, 2, seed=22)
    data = newest.read_bytes()
    newest.write_bytes(data[: len(data) // 2])

    with pytest.warns(CheckpointWarning, match="digest mismatch"):
        loaded = load_checkpoint(tmp_path)

    assert loaded is not None
    assert loaded.step == 1
    assert torch.equal(
        loaded.model_state["weight"], previous.state_dict()["weight"]
    )


def test_a_single_flipped_byte_is_caught(tmp_path: Path) -> None:
    """Bit rot, not just truncation: any byte change breaks the name's digest."""
    _save(tmp_path, 1, seed=11)
    newest, _ = _save(tmp_path, 2, seed=22)
    data = bytearray(newest.read_bytes())
    data[len(data) // 2] ^= 0xFF
    newest.write_bytes(bytes(data))

    with pytest.warns(CheckpointWarning, match="digest mismatch"):
        loaded = load_checkpoint(tmp_path)
    assert loaded is not None
    assert loaded.step == 1


def test_every_checkpoint_being_damaged_reads_as_no_checkpoint(tmp_path: Path) -> None:
    """None means "start from scratch"; list_checkpoints says whether that is a loss."""
    for step, seed in ((1, 1), (2, 2)):
        path, _ = _save(tmp_path, step, seed)
        path.write_bytes(b"")

    with pytest.warns(CheckpointWarning):
        loaded = load_checkpoint(tmp_path)
    assert loaded is None
    assert len(list_checkpoints(tmp_path)) == 2


# --------------------------------------------------------------------------
# Numerical integrity (review TS-01, 2026-09-12)
# --------------------------------------------------------------------------


def _accept_any_state(
    model_state: Mapping[str, Any], optim_state: Mapping[str, Any]
) -> None:
    """The pre-TS-01 writer: publishes whatever it is handed."""


def _save_poisoned(
    monkeypatch: pytest.MonkeyPatch, directory: Path, step: int, *, seed: int
) -> Path:
    """Publish a checkpoint whose first weight is NaN, past the writer's guard.

    The guard in :func:`save_checkpoint` is the thing under test on the write
    side, so the *read* side's test has to get a poisoned file onto disk some
    other way. Disabling the validator for the duration of this one call is the
    honest simulation of the file a pre-TS-01 build wrote: a real checkpoint,
    correct digest, correct structure, NaN weights.
    """
    model = _tiny_model(seed)
    optimiser = _trained_optimiser(model)
    state = dict(model.state_dict())
    state["weight"] = torch.full_like(state["weight"], float("nan"))
    monkeypatch.setattr(
        checkpoint_module, "_refuse_nonfinite_state", _accept_any_state
    )
    path = save_checkpoint(
        directory, step, state, optimiser.state_dict(), {"note": "planted"}
    )
    monkeypatch.undo()
    return path


def test_save_checkpoint_refuses_nonfinite_state(tmp_path: Path) -> None:
    """A NaN in the weights or in a moment estimate is refused, and writes nothing.

    Both halves, because the optimiser state is the half that is easy to forget:
    resuming with NaN moments makes SGD's very first update NaN no matter how
    healthy the parameters it was handed are. And nothing may reach the disk --
    a refusal that left a temporary file behind would just move the problem.
    """
    model = _tiny_model(3)
    optimiser = _trained_optimiser(model)

    bad_model = dict(model.state_dict())
    bad_model["weight"] = torch.full_like(bad_model["weight"], float("nan"))
    with pytest.raises(ValueError, match="non-finite value in model state"):
        save_checkpoint(tmp_path, 1, bad_model, optimiser.state_dict(), {})

    bad_optim = optimiser.state_dict()
    first = next(iter(bad_optim["state"].values()))
    first["momentum_buffer"] = torch.full_like(
        first["momentum_buffer"], float("inf")
    )
    with pytest.raises(ValueError, match="non-finite value in optimiser state"):
        save_checkpoint(tmp_path, 1, model.state_dict(), bad_optim, {})

    assert list(tmp_path.iterdir()) == [], "a refused save left a file behind"


def test_a_checkpoint_with_a_nan_parameter_is_never_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A poisoned NEWEST checkpoint costs one interval, exactly as a torn one does.

    This is the second half of the witness in review TS-01: the trainer
    published a NaN checkpoint and ``load_checkpoint`` handed it straight back,
    so a resume trained from nothing. Non-finite is now INVALID in the same
    sense a digest mismatch is -- warned about by name, skipped, and walked
    past to the newest file that is actually usable.
    """
    _, healthy = _save(tmp_path, 1, 11)
    poisoned = _save_poisoned(monkeypatch, tmp_path, 2, seed=22)
    assert poisoned.exists(), "the plant itself must reach the disk"

    with pytest.warns(CheckpointWarning, match="non-finite value"):
        loaded = load_checkpoint(tmp_path)

    assert loaded is not None
    assert loaded.step == 1, "the poisoned newest checkpoint was selected"
    assert torch.equal(loaded.model_state["weight"], healthy.state_dict()["weight"])
    assert bool(torch.isfinite(loaded.model_state["weight"]).all())


def test_an_empty_or_missing_directory_loads_as_none(tmp_path: Path) -> None:
    """Nothing to resume from is not an error."""
    assert load_checkpoint(tmp_path) is None
    assert list_checkpoints(tmp_path) == ()
    assert load_checkpoint(tmp_path / "never-created") is None
    assert list_checkpoints(tmp_path / "never-created") == ()


def test_the_newest_step_wins(tmp_path: Path) -> None:
    """Ordering is by step, not by name order or directory order."""
    for step in (9, 100, 11):
        _save(tmp_path, step, seed=step)
    refs = list_checkpoints(tmp_path)
    assert [ref.step for ref in refs] == [100, 11, 9]
    loaded = load_checkpoint(tmp_path)
    assert loaded is not None and loaded.step == 100


def test_resaving_a_step_supersedes_it(tmp_path: Path) -> None:
    """Last write wins, with no interval in which the step has no valid file."""
    first, _ = _save(tmp_path, 5, seed=1)
    second, latest = _save(tmp_path, 5, seed=2)
    assert first != second
    assert not first.exists()
    assert [ref.path for ref in list_checkpoints(tmp_path)] == [second]

    loaded = load_checkpoint(tmp_path)
    assert loaded is not None
    assert torch.equal(loaded.model_state["weight"], latest.state_dict()["weight"])


def test_unparseable_names_are_not_checkpoints(tmp_path: Path) -> None:
    """A stray file matching the glob but not the format is ignored, not skipped."""
    _save(tmp_path, 1, seed=1)
    (tmp_path / "checkpoint-notes.pt").write_bytes(b"not a checkpoint")
    refs = list_checkpoints(tmp_path)
    assert [ref.step for ref in refs] == [1]
    loaded = load_checkpoint(tmp_path)
    assert loaded is not None and loaded.step == 1


def test_a_forged_payload_with_a_matching_name_still_fails_structure_checks(
    tmp_path: Path,
) -> None:
    """The digest is not the only gate: the payload must be the right shape."""
    payload = {"format_version": 99, "step": 4, "meta_json": "{}", "model": {}}
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    data = buffer.getvalue()
    digest = hashlib.sha256(data).hexdigest()[:16]
    (tmp_path / f"checkpoint-000000004-{digest}.pt").write_bytes(data)

    with pytest.warns(CheckpointWarning, match="format version"):
        assert load_checkpoint(tmp_path) is None


def test_non_serialisable_metadata_fails_before_anything_is_written(
    tmp_path: Path,
) -> None:
    """A bad metadata field cannot leave a temporary file behind."""
    model = _tiny_model(0)
    with pytest.raises(TypeError):
        save_checkpoint(
            tmp_path, 1, model.state_dict(), {}, {"tensor": torch.zeros(2)}
        )
    assert list(tmp_path.iterdir()) == []


def test_metadata_is_stored_as_plain_json(tmp_path: Path) -> None:
    """Reading metadata back needs no unpickler -- it is text in the payload."""
    model = _tiny_model(0)
    meta = {"b": 2, "a": 1}
    stored = {**meta, **plane_meta()}
    save_checkpoint(tmp_path, 1, model.state_dict(), {}, meta)
    loaded = load_checkpoint(tmp_path)
    assert loaded is not None
    assert loaded.meta == stored
    assert json.dumps(loaded.meta, sort_keys=True) == json.dumps(
        stored, sort_keys=True
    )


def test_a_negative_step_is_rejected(tmp_path: Path) -> None:
    """Steps are counts; a negative one would sort ahead of everything."""
    with pytest.raises(ValueError, match="step"):
        save_checkpoint(tmp_path, -1, {}, {}, {})


def test_a_checkpoint_the_allowlist_rejects_costs_one_interval(tmp_path: Path) -> None:
    """``weights_only=True`` refusing a payload is a skip, not a crash.

    The file is perfectly intact -- ``torch.save`` wrote it, its bytes hash to
    the digest in its own name -- but its optimiser state carries an object
    outside torch's allowlist (an optimiser wrapper stashing a
    ``datetime.date``, say), so the safe unpickler refuses to build it and
    raises ``pickle.UnpicklingError``. That is not a subclass of any other
    entry in ``_LOAD_ERRORS``, so before it was listed this took the whole
    resume down instead of falling back one checkpoint.
    """
    _, previous = _save(tmp_path, 1, seed=11)
    model = _tiny_model(22)
    poisoned = save_checkpoint(
        tmp_path,
        2,
        model.state_dict(),
        {"stamped_at": datetime.date(2026, 8, 22)},
        {"iteration": 2},
    )
    # The file itself is whole: this is a deserialisation policy, not damage.
    assert hashlib.sha256(poisoned.read_bytes()).hexdigest()[:16] in poisoned.name
    with pytest.raises(pickle.UnpicklingError):
        torch.load(poisoned, map_location="cpu", weights_only=True)

    with pytest.warns(CheckpointWarning, match="skipping unusable checkpoint"):
        loaded = load_checkpoint(tmp_path)

    assert loaded is not None
    assert loaded.step == 1
    assert torch.equal(loaded.model_state["weight"], previous.state_dict()["weight"])


# ---------------------------------------------------------------------------
# KOM-002 §7 E -- old checkpoints do not load, and say so by name
# ---------------------------------------------------------------------------

PMV1_PLANES: Final[int] = 118
"""The plane count KOM-002 broke: 118, from KOM-027's map."""


def _campaign_net(monkeypatch: pytest.MonkeyPatch) -> KomugiNet:
    """The campaign's own net, at the small shape, built through the factory.

    ``SMALL_CONFIG`` shrinks the trunk and nothing else: ``in_planes`` stays at
    the production value, because that is the seam the checkpoint guard is
    about. Built through :func:`~komugi.campaign.factory.new_net` so this test
    cannot drift from the shape the load site actually receives.
    """
    monkeypatch.setenv(factory.NET_ENV, factory.SMALL_NET)
    net = factory.new_net()
    assert net.config.trunk.in_planes == factory.IN_PLANES == 128
    return net


def _save_stale_checkpoint(
    directory: Path, net: torch.nn.Module, meta: Mapping[str, object]
) -> Path:
    """Save `net`'s OWN weights under `meta`, at step 1.

    The weights are the net's own on purpose: ``load_state_dict`` would
    succeed if it were ever reached, so a refusal proves the guard fired
    *before* it rather than a size mismatch surfacing underneath.
    """
    optimiser = torch.optim.SGD(net.parameters(), lr=0.1)
    return save_checkpoint(
        directory, 1, net.state_dict(), optimiser.state_dict(), meta
    )


def _write_pre_kom002_checkpoint(
    directory: Path, net: torch.nn.Module, meta: Mapping[str, object]
) -> Path:
    """Publish a step-1 checkpoint the way the codec BEFORE KOM-002 published one.

    :func:`~komugi.train.checkpoint.save_checkpoint` now stamps ``in_planes``
    and ``plane_map_version`` into every file it writes, because a writer that
    has to *remember* to pass them is a writer that will forget -- and did, in
    the EMA publisher, the anchor pin and every test rig, which is what this
    stamping fixes. It therefore cannot produce the artefact this test is
    about any more.

    The artefact is still real: it is sitting in every root started before
    KOM-002. So it is forged here exactly as the old codec published it -- the
    same payload keys, the same canonical JSON, the same digest-carrying name
    -- with the two keys simply absent. Deliberately *not* routed through
    ``save_checkpoint`` with a bypass flag: an escape hatch that writes an
    unloadable checkpoint would be reachable from production code, and the one
    thing this whole guard exists to prevent is exactly that file being made.
    """
    optimiser = torch.optim.SGD(net.parameters(), lr=0.1)
    payload = {
        "format_version": FORMAT_VERSION,
        "step": 1,
        "meta_json": json.dumps(dict(meta), sort_keys=True, separators=(",", ":")),
        "model": dict(net.state_dict()),
        "optim": dict(optimiser.state_dict()),
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    data = buffer.getvalue()
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(data).hexdigest()[:16]
    path = directory / f"checkpoint-000000001-{digest}.pt"
    path.write_bytes(data)
    return path


def test_a_118_plane_checkpoint_is_refused_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#21: both plane counts and the ticket, before ``load_state_dict``.

    A 118-plane checkpoint into a 128-plane trunk cannot work -- the stem's
    weight is ``(channels, in_planes, 3, 3)`` and the added input channels have
    no learned values -- so the migration is a NAMED refusal, not a rescue.
    What this pins is that the name is there: a bare torch ``size mismatch for
    stem.weight`` traceback tells an operator nothing about which ticket
    invalidated their run.
    """
    net = _campaign_net(monkeypatch)
    _save_stale_checkpoint(
        tmp_path, net, {"in_planes": PMV1_PLANES, "plane_map_version": 1}
    )

    loaded = load_checkpoint(tmp_path)
    assert loaded is not None, "the checkpoint itself is intact and verifies"

    with pytest.raises(ValueError) as raised:
        require_matching_plane_count(loaded, net.config.trunk.in_planes)
    message = str(raised.value)
    assert "118" in message
    assert "128" in message
    assert "KOM-002" in message

    # And at a real call site, on the object the loader returned.
    with pytest.raises(ValueError, match="KOM-002"):
        factory._load_model_state(net, str(tmp_path))


def test_a_version_2_checkpoint_is_refused_after_the_cap_bump(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KOM-165: a campaign 2-4 checkpoint cannot be resumed under plane map 3.

    The version-1 case above is the 118-plane era, which the plane COUNT would
    also catch in most call sites. Version 2 is the case with no other tell at
    all: 128 planes, the same field names, the same dtypes, the same shapes, the
    same schema fingerprint -- and planes 115 and 120 carrying values normalised
    by a 256-ply cap instead of this process's. A net resumed across that reads
    every long game's clock at half speed, silently, for the rest of the run.

    So this pins the refusal at the version that actually exists on disk today,
    and pins BOTH numbers into the message: an operator who sees it has to be
    able to tell which side of KOM-165 the file came from without opening it.
    """
    net = _campaign_net(monkeypatch)
    _save_stale_checkpoint(
        tmp_path, net, {"in_planes": factory.IN_PLANES, "plane_map_version": 2}
    )

    loaded = load_checkpoint(tmp_path)
    assert loaded is not None
    # The premise again: only the version distinguishes this file.
    assert loaded.meta["in_planes"] == net.config.trunk.in_planes
    assert loaded.meta["plane_map_version"] == 2 != PLANE_MAP_VERSION

    with pytest.raises(ValueError) as raised:
        require_matching_plane_count(loaded, net.config.trunk.in_planes)
    message = str(raised.value)
    assert "plane map version" in message
    assert "version 2" in message
    assert str(PLANE_MAP_VERSION) in message

    with pytest.raises(ValueError, match="KOM-002"):
        factory._load_model_state(net, str(tmp_path))


def test_a_checkpoint_without_in_planes_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#22: absence is not agreement.

    A checkpoint written before KOM-002 carries neither ``in_planes`` nor
    ``plane_map_version``. Treating "no claim" as "the right claim" is exactly
    how a stale checkpoint would slip through the one guard designed to stop
    it, so the missing key takes the same named refusal.

    The file is forged rather than saved (see
    :func:`_write_pre_kom002_checkpoint`): today's codec stamps both keys, so
    the only way to hold a checkpoint that carries neither is to write the
    bytes the old one wrote. The refusal under test is unchanged -- what
    changed is that no *current* writer can reach it.
    """
    net = _campaign_net(monkeypatch)
    _write_pre_kom002_checkpoint(tmp_path, net, {"iteration": 1, "lr": 0.1})

    loaded = load_checkpoint(tmp_path)
    assert loaded is not None
    assert "in_planes" not in loaded.meta

    with pytest.raises(ValueError) as raised:
        require_matching_plane_count(loaded, net.config.trunk.in_planes)
    message = str(raised.value)
    assert "in_planes" in message
    assert "KOM-002" in message

    with pytest.raises(ValueError, match="KOM-002"):
        factory._load_model_state(net, str(tmp_path))


def test_the_plane_count_refusal_is_not_a_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#32: the refusal RAISES; it does not warn and fall through to ``None``.

    ``load_checkpoint`` catches every member of ``_LOAD_ERRORS`` -- ``ValueError``
    included -- emits a :class:`CheckpointWarning` and walks on to the next
    candidate, ultimately returning ``None``, which every caller reads as
    "start from scratch". A plane-count check placed inside the loader (or
    inside ``_decode``) would therefore turn a refusal into a campaign that
    LOOKS like it resumed, with an untrained net and nothing in the logs but a
    warning.

    Asserted both ways: the exception escapes, and no ``CheckpointWarning`` was
    emitted on the way -- the signature the skip path would leave behind.
    """
    net = _campaign_net(monkeypatch)
    _save_stale_checkpoint(
        tmp_path, net, {"in_planes": PMV1_PLANES, "plane_map_version": 1}
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="KOM-002"):
            factory._load_model_state(net, str(tmp_path))

    skips = [entry for entry in caught if issubclass(entry.category, CheckpointWarning)]
    assert not skips, (
        "the stale checkpoint was skipped rather than refused: "
        f"{[str(entry.message) for entry in skips]}"
    )


def test_a_writer_that_states_no_plane_meta_still_publishes_a_loadable_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KOM-002a: the codec stamps the plane map, so no writer can forget it.

    The regression this closes is not hypothetical. Every writer that was not
    the trainer -- the EMA publisher self-play actually *serves*
    (:func:`komugi.train.ema.publish_ema`), the anchor pin, every campaign test
    rig -- passed a ``meta`` of its own with no plane keys in it, and the guard
    above then refused files this repository had written seconds earlier. In a
    live campaign that surfaced as ``inference server for incarnation 0 exited
    with code 1``: the refusal fired inside a spawned process, where the
    message it so carefully composed reached nobody.

    So the assertion is end to end: a bare ``meta``, and the checkpoint both
    carries the encoder's numbers and survives the guard at a real load site.
    """
    net = _campaign_net(monkeypatch)
    _save_stale_checkpoint(tmp_path, net, {"iteration": 4})

    loaded = load_checkpoint(tmp_path)
    assert loaded is not None
    assert loaded.meta["iteration"] == 4, "the caller's own fields survive"
    assert loaded.meta["in_planes"] == observation_size()[0]
    assert loaded.meta["plane_map_version"] == PLANE_MAP_VERSION

    require_matching_plane_count(loaded, net.config.trunk.in_planes)
    factory._load_model_state(net, str(tmp_path))


def test_the_stamp_never_overwrites_a_plane_count_the_caller_stated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Filling a gap is not the same as agreeing with the writer.

    A ``setdefault`` that had been an assignment would rewrite every stale
    file's claim to this process's own on the way past, and the refusal in
    :func:`~komugi.train.checkpoint.require_matching_plane_count` would then be
    unfalsifiable: nothing on disk could ever disagree with it.
    """
    net = _campaign_net(monkeypatch)
    _save_stale_checkpoint(
        tmp_path, net, {"in_planes": PMV1_PLANES, "plane_map_version": 1}
    )

    loaded = load_checkpoint(tmp_path)
    assert loaded is not None
    assert loaded.meta["in_planes"] == PMV1_PLANES
    assert loaded.meta["plane_map_version"] == 1


def test_a_stale_plane_map_version_checkpoint_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review F2: the plane COUNT agrees and the file is still refused, by name.

    This is the one failure the shape provably cannot see. KOM-002 §2 reserves
    planes 121-127, so a later ticket that fills plane 121 leaves ``in_planes``
    at 128 while changing what every plane *means*. ``load_state_dict`` would
    succeed (the weights here are the net's own), the run would resume, and the
    net would read a different board from the one it was trained on -- silently,
    for as long as the campaign lasted.

    ``plane_map_version`` is the field that closes it, and a field that is
    written and never read closes nothing: before this node,
    ``require_matching_plane_count`` compared ``in_planes`` alone and this
    checkpoint loaded without a word.

    **Deleting the version check in
    :func:`~komugi.train.checkpoint.require_matching_plane_count` fires
    ``pytest.raises(ValueError)`` below** -- the guard returns cleanly on a
    version-1 file -- and the ``factory._load_model_state`` call after it.
    """
    net = _campaign_net(monkeypatch)
    _save_stale_checkpoint(
        tmp_path, net, {"in_planes": factory.IN_PLANES, "plane_map_version": 1}
    )

    loaded = load_checkpoint(tmp_path)
    assert loaded is not None
    # The premise: the count agrees exactly, so only the version can refuse it.
    assert loaded.meta["in_planes"] == net.config.trunk.in_planes
    assert loaded.meta["plane_map_version"] == 1 != PLANE_MAP_VERSION

    with pytest.raises(ValueError) as raised:
        require_matching_plane_count(loaded, net.config.trunk.in_planes)
    message = str(raised.value)
    assert "plane map version" in message
    assert "version 1" in message
    assert str(PLANE_MAP_VERSION) in message
    assert "KOM-002" in message

    # And at a real call site, on the object the loader returned.
    with pytest.raises(ValueError, match="KOM-002"):
        factory._load_model_state(net, str(tmp_path))


def test_a_checkpoint_without_a_plane_map_version_is_refused_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review F2, the absence half: no version is not "the current version".

    A file forged the way the pre-KOM-002 codec wrote them carries a plane
    count only. Reading its silence as agreement is the same defect as reading
    a missing ``in_planes`` that way (#22), one field along, and it is the case
    that actually exists on disk in every root started before the ticket.
    """
    net = _campaign_net(monkeypatch)
    _write_pre_kom002_checkpoint(
        tmp_path, net, {"iteration": 1, "in_planes": factory.IN_PLANES}
    )

    loaded = load_checkpoint(tmp_path)
    assert loaded is not None
    assert loaded.meta["in_planes"] == net.config.trunk.in_planes
    assert "plane_map_version" not in loaded.meta

    with pytest.raises(ValueError) as raised:
        require_matching_plane_count(loaded, net.config.trunk.in_planes)
    assert "plane_map_version" in str(raised.value)


def _eight_plane_net() -> KomugiNet:
    """A net whose ``in_planes`` is 8, not the encoder's 128.

    Deliberately tiny elsewhere -- one block, eight channels, a one-plane
    policy head -- because the only property under test is the second axis of
    ``trunk.stem.weight``. Such nets are not exotic: ``in_planes=8`` is one of
    the shapes ``tests/net/behavior/test_scaling_configs.py`` parametrises
    over, and any of them can be checkpointed.
    """
    return KomugiNet(
        NetConfig(
            trunk=TrunkConfig(in_planes=8, channels=8, blocks=1, gpool_every=3, seed=0),
            heads=HeadsConfig(channels=8, num_actions=81, policy_planes=1, seed=0),
        )
    )


def test_the_stamp_describes_the_weights_not_the_process(tmp_path: Path) -> None:
    """Review F5: ``in_planes`` is a claim about THESE weights.

    ``plane_meta`` used to read the running encoder, so saving an 8-plane net
    stamped ``in_planes = 128`` onto weights whose stem is
    ``(channels, 8, 3, 3)``. The stamp was then false, the guard passed on it,
    and the load died in torch with the ``size mismatch for stem.weight``
    traceback the guard exists to replace -- with the checkpoint's own metadata
    agreeing with the net right up to the moment it did not.

    ``plane_map_version`` stays the *encoder's*, and that asymmetry is the
    point: a plane map is a property of the process that produced the
    observations these weights were trained on, and no tensor shape carries it.

    **Reverting ``plane_meta`` to ``observation_size()[0]`` fires the
    ``meta["in_planes"] == 8`` assertion below with 128**, and the refusal after
    it stops firing entirely.
    """
    net = _eight_plane_net()
    stem = net.state_dict()["trunk.stem.weight"]
    assert stem.shape[1] == 8 != observation_size()[0]

    optimiser = torch.optim.SGD(net.parameters(), lr=0.1)
    save_checkpoint(tmp_path, 1, net.state_dict(), optimiser.state_dict(), {"lr": 0.1})

    loaded = load_checkpoint(tmp_path)
    assert loaded is not None
    assert loaded.meta["in_planes"] == 8, (
        "the stamp describes the saved net's stem, not the encoder this "
        "process happens to run"
    )
    assert loaded.meta["plane_map_version"] == PLANE_MAP_VERSION

    # Which is what lets the guard do its job: an 8-plane file into a 128-plane
    # trunk is refused BY NAME rather than inside load_state_dict.
    with pytest.raises(ValueError) as raised:
        require_matching_plane_count(loaded, observation_size()[0])
    message = str(raised.value)
    assert "8 input planes" in message
    assert str(observation_size()[0]) in message
    assert "KOM-002" in message

    # And the stamp is not merely "not 128": a net of the encoder's own shape
    # still stamps the encoder's number, so this is not a check that broke the
    # common case to fix the rare one.
    assert plane_meta(net.state_dict())["in_planes"] == 8
    assert plane_meta()["in_planes"] == observation_size()[0]


# ---------------------------------------------------------------------------
# KOM-036 §1.6 / §5 -- both writers name the arm that produced them (#31)
# ---------------------------------------------------------------------------

_ARM_PLANES: Final[int] = 4
_ARM_ACTIONS: Final[int] = 81
_ARM_ROWS: Final[int] = 4
_ARM_KEY: Final[str] = "fast_policy_weight"

TRAINER_WRITER: Final[str] = "trainer"
EMA_WRITER: Final[str] = "ema"


def _arm_net(seed: int = 3) -> KomugiNet:
    """A one-block, 81-action net -- the ``tests/train`` trainer fixture."""
    trunk = TrunkConfig(
        in_planes=_ARM_PLANES, channels=8, blocks=1, gpool_every=3, seed=seed
    )
    heads = HeadsConfig(
        channels=8, num_actions=_ARM_ACTIONS, policy_planes=1, seed=seed
    )
    return KomugiNet(NetConfig(trunk=trunk, heads=heads))


def _arm_sample(row: int) -> dict[str, torch.Tensor]:
    """One schema-shaped row, a deterministic function of `row`.

    ``search_mode`` is present because ``TrainStep.run`` requires the fast-row
    stratum column **unconditionally**, at every arm including arm C -- the
    mapping is passed on one code path so that ``fast_policy_weight = 1.0`` is
    not a bypass (KOM-036 §1.5, review m8).

    All rows are ``MODE_FULL``, deliberately: this node is about the *stamp*,
    and a batch that happened to be all-fast under arm B would compose to a
    zero-mass policy head -- reachable by design, covered by its own node in
    ``tests/train/behavior/test_fast_policy_weight.py``, and nothing but noise
    here.

    ``termination`` is present on the same terms (KOM-160): ``run`` requires
    the cap-row stratum column at every ``cap_row_value_weight``. All rows are
    decisive, so that multiplier is exactly ones and this node stays about the
    stamp.

    ``ply`` and ``mlh_target`` are present on those terms a third time
    (KOM-159 D4): the moves-left cap-row weight reconstructs the row's game
    length from the pair, and ``run`` demands both at every
    ``mlh_cap_row_weight``. Decisive rows are never censored, so this node
    stays about the stamp here too.
    """
    generator = torch.Generator().manual_seed(900 + row)
    policy = torch.rand(_ARM_ACTIONS, generator=generator) + 0.05
    wdl = torch.zeros(3)
    wdl[row % 3] = 1.0
    return {
        "observation": torch.rand(
            (_ARM_PLANES, 9, 9), generator=generator, dtype=torch.float32
        ),
        "policy_target": policy / policy.sum(),
        "wdl_target": wdl,
        "search_mode": torch.tensor(MODE_FULL, dtype=torch.int64),
        "termination": torch.tensor(TERMINATION_DECISIVE, dtype=torch.int64),
        "ply": torch.tensor(row, dtype=torch.int64),
        "mlh_target": torch.tensor(0.25, dtype=torch.float32),
    }


def _write_trainer_checkpoint(directory: Path, arm: float) -> None:
    """Take one real training iteration under `arm` and let it checkpoint."""
    net = _arm_net()
    training_iteration(
        net,
        [_arm_sample(row) for row in range(_ARM_ROWS)],
        TrainStep(
            net,
            LossAssembler(P0A_RECIPE),
            StepConfig(batch_size=2, lr=1e-3, seed=1, fast_policy_weight=arm),
        ),
        steps=1,
        rng_seed=0,
        checkpoint_dir=directory,
        iteration=0,
        global_step_base=0,
    )


def _write_ema_checkpoint(directory: Path, arm: float) -> None:
    """Publish an EMA checkpoint stamped with `arm`, through the real writer."""
    state = _arm_net(seed=5).state_dict()
    publish_ema(
        directory,
        1,
        state,
        ema_meta(
            iteration=0,
            source_step=1,
            train_sha256="0" * 64,
            ema_sha256="1" * 64,
            fast_policy_weight=arm,
        ),
    )


@pytest.mark.parametrize("writer", [TRAINER_WRITER, EMA_WRITER])
@pytest.mark.parametrize("arm", [0.0, 0.125, 1.0])
def test_a_checkpoint_names_the_arm_that_produced_it(
    tmp_path: Path, writer: str, arm: float
) -> None:
    """#31: **both** writers stamp ``fast_policy_weight``, so no arm is misattributed.

    Parametrised over the two writers deliberately (review M12). Rev 1's node
    covered the trainer only, went green, and left every *served* checkpoint --
    the EMA, which is the artefact most likely to be copied out of a run
    directory and handed to an evaluator -- carrying no arm label at all: a
    fixed six-key dict of ``iteration``/``role``/``decay``/``source_step`` and
    the two digests. An unlabelled served net cannot be attributed to an arm
    after the fact, which is the whole point of a pre-registered A/B.

    Arm ``0.0`` is in the sweep because it is the falsy one: an implementation
    that stamped ``if weight:`` rather than ``if weight is not None:`` would
    lose exactly the discard arm, and would lose it silently.
    """
    directory = tmp_path / writer
    if writer == TRAINER_WRITER:
        _write_trainer_checkpoint(directory, arm)
    else:
        _write_ema_checkpoint(directory, arm)

    loaded = load_checkpoint(directory)
    assert loaded is not None, f"the {writer} wrote no readable checkpoint"
    assert _ARM_KEY in loaded.meta, (
        f"the {writer}'s checkpoint carries no {_ARM_KEY}; the arm that "
        f"produced these weights is unrecoverable from the file"
    )
    assert loaded.meta[_ARM_KEY] == arm
    # Through JSON and back, not merely present: the stamp has to survive the
    # codec, which is what makes it readable by an evaluator on another machine.
    assert json.loads(json.dumps(loaded.meta))[_ARM_KEY] == arm


def test_an_unlabelled_ema_checkpoint_stays_unlabelled(tmp_path: Path) -> None:
    """Absent means the key is **omitted**, never stamped as ``None``.

    The distinction is the one a reader needs: "no arm recorded" (a checkpoint
    written before the stamp existed, or by a caller that has no arm in scope)
    must not read back as "an arm recorded as nothing". ``ema_meta``'s default
    exists so the three pre-existing call sites keep working; this pins that the
    default does not invent a label.
    """
    meta = ema_meta(
        iteration=0, source_step=1, train_sha256="0" * 64, ema_sha256="1" * 64
    )
    assert _ARM_KEY not in meta
    assert set(meta) == {
        "iteration",
        "role",
        "decay",
        "source_step",
        "train_sha256",
        "ema_sha256",
    }

    publish_ema(tmp_path, 1, _arm_net(seed=5).state_dict(), meta)
    loaded = load_checkpoint(tmp_path)
    assert loaded is not None
    assert _ARM_KEY not in loaded.meta
