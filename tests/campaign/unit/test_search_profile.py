"""The durable record of which search a campaign played (KOM-225).

Campaign 6 switched its proof and discovery layers on at iteration 571 of a
run whose first 570 iterations played neither, and the only record of the
switch was a ticket. Nothing in the data said it: ``extras`` are floats, the
shard schema is frozen, and a :class:`~komugi.config.SelfPlayConfig` is never
persisted. The module under test is the fix -- one appended JSON line per
search profile, a per-iteration copy beside the rows, and two float flags on
every record -- and these nodes hold the three claims that make it worth
anything.

**The profile describes the search that actually ran.** ``profile_of`` reads
the modes, the identity they name and the provider discovery uses off the two
configs, so a line is a transcription rather than an operator's intention
(T1).

**The identity is static.** ``static_digest`` is a function of the search and
the recipe and not of the moment, so a hundred relaunches of one configuration
produce one line and every line in the file is a real change (T2). That is
what makes the guard usable: a guard that fired on every restart would be
switched off within a week.

**The guard cannot be walked past by accident.** A relaunch whose search
differs from the recorded one raises, naming the fields and the flag that
authorises the change (T4), and ``run_campaign`` asks before it spawns
anything (T6). The failure this prevents is silent: the rows keep arriving, in
the same directories, produced by a different algorithm, and no later reader
can separate them.

Every node here works on ``tmp_path`` and none of them plays a game.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import warnings
from pathlib import Path

import pytest

from komugi.campaign import loop as campaign_loop
from komugi.campaign.loop import (
    CampaignConfig,
    ResumePoint,
    SEARCH_DISCOVERY_ON,
    SEARCH_PROOF_ON,
    guard_search_profile,
    iteration_root,
    search_profile_extras,
    write_iteration_profile,
)
from komugi.campaign.search_profile import (
    ITERATION_PROFILE_FILENAME,
    PROFILE_FILENAME,
    PROFILE_SCHEMA,
    SearchProfile,
    append_profile,
    canonical_json,
    latest_profile,
    main_sha,
    profile_of,
    read_profiles,
    static_digest,
)
from komugi.config import SelfPlayConfig
from komugi.selfplay.tactical import GameMarshalCaptureProvider


def _selfplay(**overrides: object) -> SelfPlayConfig:
    """The smallest valid self-play config, one field at a time varied."""
    fields: dict[str, object] = {"workers": 1, "process_batch_size": 1}
    fields.update(overrides)
    return SelfPlayConfig(**fields)  # type: ignore[arg-type]


def _armed() -> SelfPlayConfig:
    """Campaign 6 phase b's self-play: both KOM-223 layers on."""
    return _selfplay(proof_mode="interval_v1", discovery_mode="root_marshal_v1")


def _campaign(root: Path, **overrides: object) -> CampaignConfig:
    """A minimal valid campaign rooted at `root`."""
    fields: dict[str, object] = {
        "iterations": 3,
        "run_seed": 11,
        "root": str(root),
        "steps_per_iteration": 4,
        "batch_size": 2,
        "lr": 1e-3,
        "workers": 1,
        "games_per_worker": 1,
    }
    fields.update(overrides)
    return CampaignConfig(**fields)  # type: ignore[arg-type]


def test_a_profile_transcribes_the_search_the_two_configs_select(
    tmp_path: Path,
) -> None:
    """T1: the modes, the algorithm they name, the provider, and the recipe.

    The armed profile is campaign 6b's and the default one is campaigns 1-6a's,
    and the three fields that distinguish them are the three a census needs:
    ``decision_id`` (which rule chose the move), ``target_id`` (which target
    the rows carry) and ``provider_id`` (which tactical provider discovery
    verified captures with). None of them is an input -- they are derived from
    the mode pair by :func:`~komugi.search.modes.search_algorithm_identity`, so
    a profile cannot record an identity the search did not have.

    ``provider_id`` is ``None`` when discovery is off rather than the provider
    with a flag beside it: discovery is the only thing that consumes one, and a
    provider recorded for a run that never called it would read as a claim
    about how those games were played.
    """
    armed = profile_of(_armed(), _campaign(tmp_path, loss_recipe="mlh4"), 571)
    assert armed.schema == PROFILE_SCHEMA
    assert (armed.proof_mode, armed.discovery_mode) == (
        "interval_v1",
        "root_marshal_v1",
    )
    assert armed.decision_id == "gumbel_sh_interval_v1"
    assert armed.target_id == "completed_q_proof_support_v1"
    assert armed.provider_id == "game-marshal-capture-v1"
    assert armed.provider_id == GameMarshalCaptureProvider.provider_id, (
        "the id recorded is the provider's own constant, so a future provider "
        "cannot rename itself while campaigns keep recording the old name"
    )
    assert armed.loss_recipe == "mlh4"
    assert armed.from_iteration == 571

    legacy = profile_of(_selfplay(), _campaign(tmp_path), 0)
    assert (legacy.proof_mode, legacy.discovery_mode) == ("off", "off")
    assert legacy.decision_id == "gumbel_sh_survivor_v1"
    assert legacy.target_id == "completed_q_full_legal_v1"
    assert legacy.provider_id is None
    assert legacy.loss_recipe == campaign_loop.RECIPE_P0B

    # The search knobs travel too: a profile that recorded only the modes would
    # call two runs at different budgets the same search.
    budgeted = profile_of(
        _selfplay(gumbel_n_full=512, draw_score=0.0, min_placements=4),
        _campaign(tmp_path),
        0,
    )
    assert budgeted.gumbel_n_full == 512
    assert budgeted.draw_score == 0.0
    assert budgeted.min_placements == 4
    assert budgeted.repetition_rule == "full", (
        "the rule is recorded as its own string value, so a line decodes back "
        "to an equal profile without a translation table"
    )


def test_the_digest_is_the_search_and_not_the_moment(tmp_path: Path) -> None:
    """T2: canonical rendering, and a digest that ignores when it was written.

    Both halves are load-bearing for the guard. The *rendering* is sorted and
    compact so the bytes are a function of the values alone -- two profiles
    built in different field orders, or by a future call site, hash the same.
    The *blanking* is what makes an unchanged relaunch silent: ``written_at``
    changes every launch, ``main_sha`` every commit, and ``from_iteration`` by
    construction (the guard compares a stored line against a profile for a
    later iteration), so a digest over the whole object would report every
    restart as a changed search and the guard would be noise.

    The three positives are the fields the guard exists to catch: the search
    layer, a budget, and the loss recipe.
    """
    profile = profile_of(_armed(), _campaign(tmp_path, loss_recipe="mlh4"), 571)
    rendered = canonical_json(profile)
    assert rendered == json.dumps(
        json.loads(rendered), sort_keys=True, separators=(",", ":")
    )
    assert "\n" not in rendered and ", " not in rendered, (
        "one line, no whitespace: the file is a JSONL log and the same bytes "
        "are copied into iter_<i>/"
    )
    assert json.loads(rendered)["schema"] == PROFILE_SCHEMA

    # Same search, different occasion -> one digest.
    elsewhere = dataclasses.replace(
        profile,
        written_at="1999-12-31T23:59:59+00:00",
        main_sha="0" * 40,
        from_iteration=9999,
    )
    assert static_digest(elsewhere) == static_digest(profile)
    assert canonical_json(elsewhere) != rendered, (
        "the occasion is still written down -- it is ignored by the digest, "
        "not dropped from the record"
    )

    for changed in (
        dataclasses.replace(profile, proof_mode="off", discovery_mode="off"),
        # Discovery alone, which is exactly the campaign 6 switch at iteration
        # 587 (root_marshal_v1 -> root_win_v1): a digest that only noticed the
        # pair moving *off* would have called those two regimes one search.
        dataclasses.replace(profile, discovery_mode="root_win_v1"),
        dataclasses.replace(profile, gumbel_n_full=512),
        # The moves-left utility's weight alone: it changes what the search
        # maximises without touching a mode name, and it is the knob a census
        # comparing lambda regimes has to be able to separate.
        dataclasses.replace(profile, mlh_utility_lambda=0.5),
        dataclasses.replace(profile, loss_recipe="mlh"),
    ):
        assert static_digest(changed) != static_digest(profile), (
            f"{changed} differs from the stored profile in a field that "
            f"decides what the games are, and the digest is what the guard "
            f"compares"
        )


def test_profiles_append_read_back_in_order_and_refuse_a_damaged_line(
    tmp_path: Path,
) -> None:
    """T3: the log round-trips, the last line is the one in force, torn raises.

    ``read_profiles`` raises where the telemetry readers skip and count, and
    the asymmetry is deliberate: a damaged line in ``selfplay.jsonl`` costs one
    iteration's measurement, while a damaged line here is the provenance of
    every row after it. A reader that skipped it would silently attribute those
    rows to whichever line survived, which is precisely the failure the file
    exists to prevent.

    An absent file is ``()`` and never an error -- that is a campaign that
    predates the record (campaign 6 below iteration 571), and the caller's job
    is to read it as *unrecorded* rather than as observed ``off``/``off``.
    """
    assert read_profiles(tmp_path) == ()
    assert latest_profile(tmp_path) is None

    first = profile_of(_selfplay(), _campaign(tmp_path), 0)
    second = profile_of(_armed(), _campaign(tmp_path, loss_recipe="mlh4"), 571)
    append_profile(tmp_path, first)
    append_profile(tmp_path, second)

    path = tmp_path / PROFILE_FILENAME
    assert path.is_file()
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2

    read = read_profiles(tmp_path)
    assert read == (first, second), (
        "oldest first, and every field survives the round trip -- a profile "
        "read back must be usable as the guard's left-hand side"
    )
    assert latest_profile(tmp_path) == second

    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not json at all\n")
    with pytest.raises(ValueError, match="line 3"):
        read_profiles(tmp_path)

    # A syntactically valid line that is not a profile fails the same way: a
    # JSON object with the wrong keys would otherwise construct nothing and
    # raise a TypeError naming no location at all.
    truncated = tmp_path / "other"
    truncated.mkdir()
    append_profile(truncated, first)
    with (truncated / PROFILE_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"schema": PROFILE_SCHEMA}) + "\n")
    with pytest.raises(ValueError, match="line 2"):
        read_profiles(truncated)


def test_a_torn_trailing_line_is_ignored_and_a_torn_interior_line_is_not(
    tmp_path: Path,
) -> None:
    """F1: the launch that died mid-write must not block every later relaunch.

    ``append_profile`` writes with ``O_APPEND`` and fsyncs before it returns, so
    a line with *no newline after it* can only be a launch that was killed while
    writing its profile -- which means it never got past the guard, never span a
    worker and never wrote a row. Raising on it would turn one interrupted
    launch into a permanent refusal for that campaign root, recoverable only by
    hand-editing the provenance log, which is the worst moment to be editing it.

    A torn line with a line *after* it is the opposite situation: the write that
    followed proves the campaign kept going, so the damaged line is the recorded
    provenance of real rows and skipping it would attribute them to whichever
    line survived. That still raises, names the line, and now says what to do.
    """
    profile = profile_of(_armed(), _campaign(tmp_path, loss_recipe="mlh4"), 571)
    append_profile(tmp_path, profile)
    with (tmp_path / PROFILE_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(profile)[:40])  # killed mid-write: no newline

    with pytest.warns(UserWarning, match="trailing line"):
        assert read_profiles(tmp_path) == (profile,)
    with pytest.warns(UserWarning, match=PROFILE_FILENAME):
        assert guard_search_profile(tmp_path, profile, False) == "unchanged", (
            "the relaunch the torn line interrupted proceeds, against the "
            "profile the last complete line recorded"
        )

    interior = tmp_path / "interior"
    interior.mkdir()
    (interior / PROFILE_FILENAME).write_text(
        canonical_json(profile)[:40] + "\n" + canonical_json(profile) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as damaged:
        read_profiles(interior)
    assert "line 1" in str(damaged.value)
    assert str(damaged.value).endswith("truncate the damaged line and relaunch"), (
        "an operator cannot repair an append-only provenance log by reading it "
        "harder, so the refusal ends with the repair"
    )


def test_the_writer_heals_a_torn_tail_instead_of_fusing_a_line_onto_it(
    tmp_path: Path,
) -> None:
    """F2: a tolerant reader needs a healing writer, or it makes things worse.

    ``read_profiles`` ignores an unterminated trailing fragment, so the launch
    the torn write interrupted gets through the guard -- and then takes the
    ``appended`` or ``switched`` branch and writes. An unconditional
    ``O_APPEND`` would concatenate this profile onto the fragment: the fused
    line *is* newline-terminated, so it is no longer the trailing line the
    reader forgives but an interior line it must raise on, and every later read
    of that root fails. One interrupted launch would have been repaired by the
    next one and instead becomes permanent.

    So the writer truncates the fragment first. It is safe to drop for the
    reason it was safe to ignore: with ``O_APPEND`` and the short-write loop,
    bytes with no newline after them are a launch that died writing its
    profile, which means it never returned from the guard and never played a
    game. Nothing on disk is attributed to them.
    """
    profile = profile_of(_armed(), _campaign(tmp_path, loss_recipe="mlh4"), 571)
    path = tmp_path / PROFILE_FILENAME
    path.write_bytes(canonical_json(profile)[:40].encode("utf-8"))

    with pytest.warns(UserWarning) as torn_only:
        assert guard_search_profile(tmp_path, profile, False) == "appended", (
            "a root whose only content is a fragment has no recorded profile, "
            "so this launch is the first one that records anything"
        )
    assert any(PROFILE_FILENAME in str(w.message) for w in torn_only), (
        "an operator has to be able to find the file the bytes were dropped "
        "from; a warning that named no path would be unactionable"
    )
    assert any("unterminated" in str(w.message) for w in torn_only), (
        "the writer's heal is announced, not silent: bytes left on disk by a "
        "previous launch were deleted"
    )

    assert read_profiles(tmp_path) == (profile,), (
        "the file is readable again -- which is the whole claim: appending "
        "onto the fragment would have made this raise forever"
    )
    contents = path.read_bytes()
    assert contents.endswith(b"\n") and contents.count(b"\n") == 1, (
        "exactly one complete line: the fragment was cut, not padded into a "
        "line of its own"
    )


def test_the_heal_keeps_the_complete_lines_above_a_torn_tail(tmp_path: Path) -> None:
    """F2b: the truncation cuts at the last newline, not at the start of the file.

    The realistic shape of the failure: a campaign recorded a profile, ran,
    and was killed while a later relaunch was writing a second one. The first
    line is the provenance of every row already on disk and must survive
    untouched; only the fragment after it goes. The relaunch then records the
    switch it was authorised to make, and the log reads back as the two
    regimes in the order they took effect.

    A non-torn file is not touched at all -- the heal is a repair, and a repair
    that rewrote healthy files would be a new way to lose provenance.
    """
    first = profile_of(_selfplay(), _campaign(tmp_path), 0)
    second = profile_of(_armed(), _campaign(tmp_path, loss_recipe="mlh4"), 571)
    append_profile(tmp_path, first)
    path = tmp_path / PROFILE_FILENAME
    intact = path.read_bytes()
    with path.open("ab") as handle:
        handle.write(canonical_json(second)[:40].encode("utf-8"))

    with pytest.warns(UserWarning, match=PROFILE_FILENAME):
        assert guard_search_profile(tmp_path, second, True) == "switched"

    assert read_profiles(tmp_path) == (first, second)
    assert latest_profile(tmp_path) == second, (
        "the profile in force is the one this launch will play, not the last "
        "one that survived the crash"
    )
    assert path.read_bytes() == intact + (canonical_json(second) + "\n").encode(
        "utf-8"
    ), "the complete line above the fragment is byte-identical afterwards"


def test_a_file_with_no_torn_tail_is_appended_to_untouched(tmp_path: Path) -> None:
    """F2c: the heal is a no-op on every healthy file, and says nothing.

    The guard against an over-eager repair. A well-terminated log (and an
    absent one) must reach the ``O_APPEND`` below exactly as it did before the
    heal existed: same bytes above, no truncation, and no warning -- a warning
    on the ordinary path would train an operator to ignore the one case that
    matters.
    """
    first = profile_of(_selfplay(), _campaign(tmp_path), 0)
    second = profile_of(_armed(), _campaign(tmp_path, loss_recipe="mlh4"), 571)
    path = tmp_path / PROFILE_FILENAME

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        append_profile(tmp_path, first)  # absent file: nothing to heal
        intact = path.read_bytes()
        append_profile(tmp_path, second)  # terminated file: nothing to heal

    assert path.read_bytes().startswith(intact), (
        "the appended line is added after the existing bytes, which are not "
        "rewritten by the heal"
    )
    assert read_profiles(tmp_path) == (first, second)


def test_the_resume_guard_records_a_change_only_when_one_is_authorised(
    tmp_path: Path,
) -> None:
    """T4: all four branches of the guard, with no campaign anywhere near it.

    The guard is the whole point of the file. A long campaign's configuration
    can change at exactly one moment -- a relaunch -- and the damage is
    invisible when it does: an edited launcher, a flag dropped from a command
    line, and the rows keep arriving in the same directories from a different
    algorithm.

    So: a fresh root records (``appended``); an unchanged relaunch writes
    nothing at all (``unchanged``), which is what keeps the file's line count
    equal to the number of real changes; a deliberate change records both
    regimes (``switched``); and an undeclared change raises, naming what
    differs and the flag that would authorise it. The message has to name both
    -- an operator who cannot see *which* field moved cannot tell a typo from
    the switch they meant to make.
    """
    legacy = profile_of(_selfplay(), _campaign(tmp_path), 0)
    assert guard_search_profile(tmp_path, legacy, False) == "appended"
    assert len(read_profiles(tmp_path)) == 1

    relaunched = profile_of(_selfplay(), _campaign(tmp_path), 300)
    assert guard_search_profile(tmp_path, relaunched, False) == "unchanged"
    assert read_profiles(tmp_path) == (legacy,), (
        "a relaunch of an unchanged campaign leaves the file byte-identical, "
        "however many times it happens"
    )

    armed = profile_of(_armed(), _campaign(tmp_path, loss_recipe="mlh4"), 571)
    with pytest.raises(ValueError) as refusal:
        guard_search_profile(tmp_path, armed, False)
    message = str(refusal.value)
    assert "allow_profile_switch" in message
    for field in ("proof_mode", "discovery_mode", "decision_id", "loss_recipe"):
        assert field in message, f"the refusal must name {field}, which changed"
    assert "written_at" not in message and "main_sha" not in message, (
        "the occasion differs on every launch and naming it would bury the "
        "fields that actually moved"
    )
    assert read_profiles(tmp_path) == (legacy,), "a refusal writes nothing"

    # A discovery-ONLY flip, which is the real campaign 6 change at iteration
    # 587: one field moves, nothing else does, and the message has to name it
    # with both values. This is the shape of mistake the guard exists for -- an
    # operator who edited one word of a launcher sees exactly that word.
    switched_root = tmp_path / "discovery_only"
    switched_root.mkdir()
    append_profile(switched_root, armed)
    with pytest.raises(ValueError) as discovery_refusal:
        guard_search_profile(
            switched_root,
            dataclasses.replace(armed, discovery_mode="root_win_v1"),
            False,
        )
    discovery_message = str(discovery_refusal.value)
    assert "discovery_mode: 'root_marshal_v1' -> 'root_win_v1'" in discovery_message
    assert "proof_mode" not in discovery_message, (
        "only the field that moved is named: a diff that listed the unchanged "
        "fields too would hide the one word that changed"
    )

    assert guard_search_profile(tmp_path, armed, True) == "switched"
    recorded = read_profiles(tmp_path)
    assert recorded == (legacy, armed)
    assert recorded[-1].from_iteration == 571, (
        "the appended line says which iterations played the new search; the "
        "ones below it belong to the line before"
    )


def test_the_iteration_copy_is_the_profile_in_force_or_nothing(
    tmp_path: Path,
) -> None:
    """The per-iteration binding: rows and their provenance travel together.

    A shard tree moved, archived or censused on its own cannot be traced back
    through a root-level log it has been separated from, so the line in force
    is copied into the iteration directory as it is created. Byte-identical to
    the log line, because both are the same canonical rendering.

    A root with no profile gets no file rather than an invented ``off``/``off``
    one: that root's iterations genuinely have no recorded search, and writing
    one would be a measurement nobody made.
    """
    write_iteration_profile(tmp_path, 4)
    assert not (campaign_loop.iteration_root(tmp_path, 4)).exists()

    profile = profile_of(_armed(), _campaign(tmp_path, loss_recipe="mlh4"), 571)
    append_profile(tmp_path, profile)
    write_iteration_profile(tmp_path, 571)
    copied = campaign_loop.iteration_root(tmp_path, 571) / ITERATION_PROFILE_FILENAME
    assert copied.read_text(encoding="utf-8") == canonical_json(profile) + "\n"
    assert SearchProfile(**json.loads(copied.read_text(encoding="utf-8"))) == profile


def test_the_two_search_flags_are_floats_and_ride_on_every_record() -> None:
    """T5: ``search_proof_on`` / ``search_discovery_on``, 1.0 and 0.0.

    :class:`~komugi.telemetry.records.IterationMetrics` extras are floats, so
    the mode *names* cannot ride on a record and these two flags are the index
    into the profile file instead. They are what lets a census filter campaign
    6b's iterations out of 6a's by reading the journal alone -- which is the
    query that could not be asked at all before KOM-225.

    The last assertion is structural, for the reason the loop's other wiring
    nodes give: ``_close_iteration``'s extras are assembled after a real
    self-play iteration, and standing that up to read two floats would test the
    orchestrator.
    """
    assert search_profile_extras(_armed()) == {
        SEARCH_PROOF_ON: 1.0,
        SEARCH_DISCOVERY_ON: 1.0,
    }
    assert search_profile_extras(_selfplay()) == {
        SEARCH_PROOF_ON: 0.0,
        SEARCH_DISCOVERY_ON: 0.0,
    }
    # The proof layer without discovery is a configuration the search validates
    # and campaign 6b passed through: the flags must not move together.
    assert search_profile_extras(_selfplay(proof_mode="interval_v1")) == {
        SEARCH_PROOF_ON: 1.0,
        SEARCH_DISCOVERY_ON: 0.0,
    }
    for value in search_profile_extras(_armed()).values():
        assert isinstance(value, float), "the journal's extras are float-only"

    source = inspect.getsource(campaign_loop._close_iteration)
    assert "search_profile_extras" in source, (
        "the boundary record is where a census reads the flags off; computing "
        "them in a helper nothing calls would leave the journal unchanged"
    )


class _Sentinel(RuntimeError):
    """Raised in place of the first thing ``run_campaign`` does after the guard."""


def _stop_after_the_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, start: int
) -> None:
    """Point ``run_campaign`` at `start` and abort it the statement after the guard.

    ``archive_telemetry`` is the call immediately below the guard and the first
    thing in ``run_campaign`` that touches a net, a checkpoint or a process, so
    replacing it stops the call in the narrow window where the guard has run
    and nothing has been spawned. ``resume_point`` is stubbed because the real
    one reads a checkpoint directory that a test with no training in it cannot
    fill.
    """
    monkeypatch.setattr(
        campaign_loop,
        "resume_point",
        lambda campaign: ResumePoint(iteration=start, checkpoint=None),
    )

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise _Sentinel("the guard has run and nothing has been spawned")

    monkeypatch.setattr(campaign_loop, "archive_telemetry", _boom)


def test_run_campaign_binds_its_profile_before_it_spawns_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T6: the guard is wired into the launch, and it is wired in early.

    Two claims, and the *order* is the one that matters. A guard asked after
    the first iteration would have let the run it was supposed to refuse play a
    whole iteration of games under an unrecorded search -- and those games are
    exactly the ones nobody could attribute later. So the campaign is entered
    for real and stopped at ``archive_telemetry``, the statement below the
    guard: the profile of a fresh root is on disk by then, and a root whose
    recorded profile disagrees raises ``ValueError`` and never reaches the
    sentinel at all.
    """
    selfplay = _armed()
    campaign = _campaign(tmp_path, loss_recipe="mlh4")

    # (1) a fresh root: the guard appended before the run got any further.
    _stop_after_the_guard(monkeypatch, tmp_path, start=0)
    with pytest.raises(_Sentinel):
        campaign_loop.run_campaign(campaign, selfplay)
    recorded = read_profiles(tmp_path)
    assert len(recorded) == 1
    assert recorded[0].decision_id == "gumbel_sh_interval_v1"
    assert recorded[0].from_iteration == 0

    # (2) the same root, relaunched with the search turned back off and no
    # authorisation: refused, before the sentinel and before any spawn.
    _stop_after_the_guard(monkeypatch, tmp_path, start=1)
    with pytest.raises(ValueError, match="allow_profile_switch"):
        campaign_loop.run_campaign(campaign, _selfplay())
    assert read_profiles(tmp_path) == recorded, "the refused launch wrote nothing"

    # (3) the same change, declared: recorded, and the run proceeds to the
    # sentinel exactly as the authorised 570->571 switch proceeds to self-play.
    with pytest.raises(_Sentinel):
        campaign_loop.run_campaign(
            _campaign(tmp_path, loss_recipe="mlh4", allow_profile_switch=True),
            _selfplay(),
        )
    switched = read_profiles(tmp_path)
    assert len(switched) == 2
    assert switched[-1].proof_mode == "off"
    assert switched[-1].from_iteration == 1


def test_the_iteration_copy_is_written_between_the_stale_rmtree_and_self_play(
    tmp_path: Path,
) -> None:
    """F3: the per-iteration copy is wired in, and wired in at the one safe spot.

    Structural, for the reason the loop's other wiring nodes give: exercising
    this stretch of ``run_campaign`` for real means a net, a checkpoint and a
    spawned orchestrator. What the ordering buys is not decoration. *Before* the
    rmtree, the copy is deleted again on every re-run of a crashed iteration and
    the tree keeps its rows with no provenance beside them; *after* the
    self-play call, an iteration that dies mid-games leaves shards whose search
    is unrecorded -- and those are precisely the games nobody can attribute
    later. Deleting the call entirely passed every other node in this file.
    """
    source = inspect.getsource(campaign_loop.run_campaign)
    rmtree = source.index("shutil.rmtree(iteration_root")
    copy = source.index("write_iteration_profile(root, iteration)")
    selfplay = source.index("run_selfplay(")
    assert rmtree < copy < selfplay, (
        "the profile copy belongs after the stale-attempt rmtree (which would "
        "delete it) and before the first game of the iteration it describes"
    )


def test_the_refused_launch_always_names_the_field_the_digest_moved_on(
    tmp_path: Path,
) -> None:
    """N1: the diff and the digest are taken over the same rendering.

    ``_profile_diff`` compares the JSON rendering of each field because that is
    what ``static_digest`` hashes. Comparing the Python values instead let the
    two disagree: ``-0.0 == 0.0`` and ``1 == 1.0`` in Python while their
    renderings differ, so the guard would refuse a relaunch and then name *no
    field at all* -- a message that tells an operator their search changed and
    refuses to say how. The invariant is now structural: digests differ if and
    only if some field's rendering differs, and that field is in the message.
    """
    profile = profile_of(_armed(), _campaign(tmp_path, loss_recipe="mlh4"), 571)
    recorded = dataclasses.replace(profile, mlh_utility_lambda=-0.0)
    launched = dataclasses.replace(profile, mlh_utility_lambda=0.0, from_iteration=600)
    assert static_digest(recorded) != static_digest(launched), (
        "the canonical rendering keeps the sign of the zero, so the two are a "
        "different recorded identity -- the guard is right to refuse"
    )

    append_profile(tmp_path, recorded)
    with pytest.raises(ValueError) as refusal:
        guard_search_profile(tmp_path, launched, False)
    assert "mlh_utility_lambda: -0.0 -> 0.0" in str(refusal.value), (
        "the fields the message names are the fields the digest disagreed on, "
        "never an empty list"
    )

    # The converse half of the same invariant: renderings that agree are a
    # digest that agrees, and nothing is written.
    assert (
        guard_search_profile(
            tmp_path, dataclasses.replace(recorded, from_iteration=600), False
        )
        == "unchanged"
    )
    assert campaign_loop._profile_diff(recorded, recorded) == ""


# --------------------------------------------------------------------------
# T7 (KOM-249): schema 2, and the four knobs KOM-245/246 are written against
# --------------------------------------------------------------------------

_SCHEMA_2_KEYS: tuple[str, ...] = (
    "cap_adjudication",
    "cap_adjudication_p_max",
    "cap_adjudication_d_full",
    "min_placements_max",
)
"""What schema 2 added -- and therefore what a schema-1 line does not have."""


def _schema_1_line(profile: SearchProfile) -> str:
    """Render `profile` the way the build that wrote schema 1 would have.

    Derived from a real profile rather than typed out: a literal would pin one
    author's memory of the schema-1 key set, and would keep passing after a
    field was added to the middle of the record. Taking today's object and
    removing exactly the keys schema 2 introduced is the migration stated as
    data.
    """
    payload = dataclasses.asdict(profile)
    for key in _SCHEMA_2_KEYS:
        del payload[key]
    payload["schema"] = 1
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _write_line(root: Path, line: str) -> None:
    """Put one already-rendered line in `root`'s profile log."""
    (root / PROFILE_FILENAME).write_text(line + "\n", encoding="utf-8")


def test_a_schema_1_line_decodes_with_the_new_fields_at_their_defaults(
    tmp_path: Path,
) -> None:
    """A1 (i): an old line is read as the behaviour it was played under.

    The campaign that wrote it ran on a build with no cap adjudication and no
    drawn draft threshold, so ``off``/``0.8``/``8``/``0`` is a *transcription*
    of that run and not a guess about it -- which is the only reason filling a
    missing key is allowed at all. Read through the module's own reader, not
    through the dataclass: the line has to survive the path a resuming campaign
    actually takes.
    """
    profile = profile_of(_armed(), _campaign(tmp_path, loss_recipe="mlh4"), 571)
    line = _schema_1_line(profile)
    assert all(key not in json.loads(line) for key in _SCHEMA_2_KEYS)
    _write_line(tmp_path, line)

    (decoded,) = read_profiles(tmp_path)
    assert decoded.schema == 1, "the record still says which build wrote it"
    assert decoded.cap_adjudication == "off"
    assert decoded.cap_adjudication_p_max == 0.8
    assert decoded.cap_adjudication_d_full == 8
    assert decoded.min_placements_max == 0
    assert decoded == dataclasses.replace(profile, schema=1), (
        "every key the old line did carry is unchanged: the migration fills "
        "the gap and touches nothing else"
    )

    # A key the schema-1 build DID write is still an error when it is missing:
    # that is a damaged line, not an old one, and the two must not be confused.
    damaged = json.loads(line)
    del damaged["draw_score"]
    _write_line(tmp_path, json.dumps(damaged))
    with pytest.raises(ValueError, match="line 1"):
        read_profiles(tmp_path)

    # And an unknown key is still an error -- a line from a LATER build, whose
    # search this code cannot describe.
    unknown = json.loads(line)
    unknown["cap_adjudication_floor"] = 0.1
    _write_line(tmp_path, json.dumps(unknown))
    with pytest.raises(ValueError, match="line 1"):
        read_profiles(tmp_path)


@pytest.mark.parametrize("key", _SCHEMA_2_KEYS)
def test_a_current_line_missing_a_schema_2_key_is_damaged_not_old(
    tmp_path: Path, key: str
) -> None:
    """Review F1: the rule the migration documents, enforced where it is read.

    The four schema-2 fields carry defaults on the dataclass -- that is what
    lets a migrated schema-1 payload construct at all -- so the constructor
    cannot tell a line that never had the key from one that lost it. Left to it,
    a current line with a hole in it would be read as a campaign that ran
    ``off``/``0.8``/``8``/``0``: a search nobody can name, transcribed as one
    somebody did. Parametrised over all four because the check is a loop over
    the migration table, and a table this test did not walk is a key the reader
    still guesses at.
    """
    payload = json.loads(canonical_json(profile_of(_selfplay(), _campaign(tmp_path), 0)))
    assert payload["schema"] == PROFILE_SCHEMA, "the line still claims this build"
    del payload[key]
    _write_line(tmp_path, json.dumps(payload, sort_keys=True, separators=(",", ":")))

    with pytest.raises(ValueError) as refusal:
        read_profiles(tmp_path)
    message = str(refusal.value)
    assert key in message, "the operator is told which key the line lacks"
    assert "line 1" in message


def test_the_iteration_copy_of_a_migrated_line_is_readable_and_says_which_schema(
    tmp_path: Path,
) -> None:
    """Review F2: the per-iteration copy is the profile IN FORCE, migrated.

    A relaunch on a schema-2 build of a campaign whose log holds one schema-1
    line appends nothing (the digest ignores ``schema``), so the copy written
    beside the iteration's rows is the *migrated* object rather than the bytes
    of the stored line: schema 1, with the four newer keys at the values that
    run played. Both halves matter -- the schema is how a later reader knows
    which build recorded the search, and the filled keys are how a shard tree
    carried off on its own is readable without this build's migration table.
    """
    profile = profile_of(_armed(), _campaign(tmp_path, loss_recipe="mlh4"), 571)
    _write_line(tmp_path, _schema_1_line(profile))
    (stored,) = read_profiles(tmp_path)

    campaign_loop.write_iteration_profile(tmp_path, 601)
    copy_path = iteration_root(tmp_path, 601) / ITERATION_PROFILE_FILENAME
    copied = json.loads(copy_path.read_text(encoding="utf-8"))
    assert copied["schema"] == 1, "the copy says which build wrote the profile"
    assert set(_SCHEMA_2_KEYS) <= copied.keys(), (
        "and carries the migrated keys, so the copy needs no migration table"
    )

    # The copy is itself a legal profile line: a tree salvaged into a fresh root
    # reads back as exactly the profile that was in force when it was written.
    salvaged = tmp_path / "salvaged"
    salvaged.mkdir()
    _write_line(salvaged, copy_path.read_text(encoding="utf-8").strip())
    assert read_profiles(salvaged) == (stored,)


def test_a_schema_1_line_with_todays_knobs_is_unchanged_on_relaunch(
    tmp_path: Path,
) -> None:
    """A1 (ii)/(iii), KOM-237 A2 and review F6: a schema bump alone is not a switch.

    ``schema`` says which *writer* produced the line. If the digest counted it,
    the first relaunch after deploying schema 2 -- of a campaign nobody
    touched -- would be refused with ``schema: 1 -> 2``, and the operator would
    have to set ``allow_profile_switch`` and record a switch that changed
    nothing. That is the guard training its own users to walk past it, which is
    the failure the guard exists to prevent.
    """
    profile = profile_of(_armed(), _campaign(tmp_path, loss_recipe="mlh4"), 571)
    _write_line(tmp_path, _schema_1_line(profile))
    (stored,) = read_profiles(tmp_path)

    assert stored.schema != profile.schema == PROFILE_SCHEMA
    assert static_digest(stored) == static_digest(profile), (
        "identical knobs, two schemas: one identity"
    )
    assert canonical_json(stored) != canonical_json(profile), (
        "the schema is still written down -- ignored by the digest, not dropped"
    )

    relaunched = dataclasses.replace(profile, from_iteration=600)
    assert guard_search_profile(tmp_path, relaunched, False) == "unchanged"
    assert read_profiles(tmp_path) == (stored,), "nothing was appended"
    assert "schema" not in campaign_loop._profile_diff(stored, profile), (
        "a field the digest blanks can never be reported as what changed"
    )


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"cap_adjudication": "material"}, id="mode"),
        pytest.param({"cap_adjudication_p_max": 0.6}, id="p_max"),
        pytest.param({"cap_adjudication_d_full": 5}, id="d_full"),
        pytest.param(
            {"min_placements": 1, "min_placements_max": 14}, id="min_placements_max"
        ),
    ],
)
def test_the_digest_moves_when_any_of_the_four_knobs_moves(
    tmp_path: Path, overrides: dict[str, object]
) -> None:
    """A1 (iii), the positive half: each knob is identity, one at a time.

    Parametrised rather than asserted as a set, because the failure this
    catches is a single field omitted from ``profile_of`` -- a campaign that
    turned cap adjudication on and whose record said it had not, which is the
    exact confusion KOM-225 exists to make impossible.

    The ``min_placements_max`` case moves the low edge to 1 alongside it: a
    DRAWN range may not start at 0, which is the mask switched off, so 14 on
    the fixture's zero low edge is a config the constructor refuses. Both
    knobs are then asserted on the record, so the case still shows the high
    edge reaching it.
    """
    base = profile_of(_selfplay(), _campaign(tmp_path), 0)
    moved = profile_of(_selfplay(**overrides), _campaign(tmp_path), 0)
    assert static_digest(moved) != static_digest(base), (
        f"{overrides} decides what a game is or what its rows mean, and the "
        f"digest is what the guard compares"
    )
    for key, value in overrides.items():
        assert getattr(moved, key) == value, f"{key} never reached the record"


def test_the_guard_refuses_an_unauthorised_cap_adjudication_switch(
    tmp_path: Path,
) -> None:
    """A1 (iv): the labelling rule changes, so the refusal names it.

    A relaunch that switched the cap rule on would keep writing rows into the
    same directories with a different *meaning* -- the draws below it are draws
    and the ones above are soft material labels -- so it is precisely a change
    that must be declared. ``allow_profile_switch`` then records both regimes,
    which is what lets a later census cut the run at the right iteration.
    """
    stored = profile_of(_selfplay(), _campaign(tmp_path), 0)
    append_profile(tmp_path, stored)
    adjudicated = profile_of(
        # The low edge moves to 1 with the high one: a drawn range may not
        # start at the mask's off switch, so the pair is what a real switch
        # would carry -- and the refusal has to report both.
        _selfplay(
            cap_adjudication="material", min_placements=1, min_placements_max=12
        ),
        _campaign(tmp_path),
        700,
    )

    with pytest.raises(ValueError) as refusal:
        guard_search_profile(tmp_path, adjudicated, False)
    message = str(refusal.value)
    assert "cap_adjudication: 'off' -> 'material'" in message
    assert "min_placements_max: 0 -> 12" in message
    assert "min_placements: 0 -> 1" in message
    assert "allow_profile_switch" in message
    assert read_profiles(tmp_path) == (stored,), "a refusal writes nothing"

    assert guard_search_profile(tmp_path, adjudicated, True) == "switched"
    recorded = read_profiles(tmp_path)
    assert recorded == (stored, adjudicated)
    assert recorded[-1].cap_adjudication == "material"
    assert recorded[-1].from_iteration == 700


def test_the_head_is_recorded_when_it_can_be_read_and_never_raises() -> None:
    """``main_sha`` is provenance, and provenance never fails a campaign.

    Every way it can go wrong -- no git, no repository, a hung invocation -- is
    ``None``, because a campaign that refused to start over an unreadable
    ``HEAD`` would be a worse outcome than a line without one. In this tree it
    is readable, so the positive case is asserted rather than assumed.
    """
    sha = main_sha()
    assert sha is None or (len(sha) == 40 and sha.strip() == sha)
