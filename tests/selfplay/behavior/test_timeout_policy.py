"""A request that is never answered times out, and the client stays dead.

Design D2 makes the timeout *fatal* on purpose, and the reasoning is worth
restating because the test looks harsher than it needs to be otherwise: a
worker that times out and then keeps submitting is the origin of every
stale-response and torn-read race in this architecture.  Its slot may be
written by the server at any later moment (the server does not know the client
gave up), so a second submission into the same segment can read the *previous*
request's answer as if it were the new one.  The design closes that by
construction -- "the worker exits nonzero; the orchestrator ... NEVER reuses a
segment across incarnations" -- which only holds if the client refuses to
continue rather than politely retrying.

So this module pins two things:

1. ``submit`` raises ``InferenceTimeout`` on a server that never starts, and
   does so on the ``request_timeout_s`` clock -- not immediately (a client that
   gives up in 5ms turns every GPU hiccup into a dead worker) and not much
   later (a client that waits forever cannot be killed by the liveness gate,
   design D10).  The walls are 0.5x and 2x: generous, because a spawned child
   importing torch under CI load is not a real-time system.
2. After that timeout the client makes NO further submissions.  The check is
   that a second ``submit`` fails *fast*: any exception is acceptable
   (``InferenceTimeout`` again, ``RuntimeError``, a custom poisoned-client
   error), but blocking for another full timeout window is not, and returning
   a value is not -- that is the race above, live.

The whole scenario runs in a spawned child so that a client which ignores its
timeout entirely cannot hang ``ci/check.sh``: the parent joins inside a wall
and fails.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

import inference_rig as rig

sp = pytest.importorskip("komugi.selfplay")

if not hasattr(sp, "ShardServer"):
    pytest.skip(
        "komugi.selfplay exports no ShardServer yet (KOM-021a)",
        allow_module_level=True,
    )

#: Short enough that the test costs seconds, long enough that a slow CI box
#: cannot make a correct client look early.
_REQUEST_TIMEOUT_S = 4.0

#: The child does two submissions, each bounded by the timeout above, plus a
#: torch import.
_CHILD_WALL_S = 120.0


def _child_main(config: Any, handle: Any, results: Any) -> None:
    """Ring a doorbell nobody is listening for, twice; report what happened."""
    client = rig.make_client(config, handle)
    obs = rig.encode_obs(7)
    legal = rig.all_legal()

    started = time.monotonic()
    try:
        client.submit(obs, legal)
    except BaseException as exc:
        results.put(("first", time.monotonic() - started, type(exc).__name__))
    else:
        results.put(("first-returned", time.monotonic() - started, ""))
        # Always two messages, so the parent never waits out its wall for a
        # report the child was never going to send.
        results.put(("not-attempted", 0.0, ""))
        return

    started = time.monotonic()
    try:
        client.submit(obs, legal)
    except BaseException as exc:
        results.put(("second", time.monotonic() - started, type(exc).__name__))
    else:
        results.put(("second-returned", time.monotonic() - started, ""))


def test_timeout_is_bounded_and_fatal() -> None:
    """No server, ever: the client times out on schedule and stays refused."""
    config = rig.make_config(request_timeout_s=_REQUEST_TIMEOUT_S)

    with rig.slots(config, [0]) as handles:
        results = rig.CTX.Queue()
        # No server process is started.  That is the whole scenario: the
        # client's ``request_seq`` climbs and nothing ever moves
        # ``response_seq`` to match it.
        child = rig.spawn(_child_main, config, handles[0], results, name="lonely")
        reported = rig.collect(results, 2, _CHILD_WALL_S)
        failures = rig.join_all([child], rig.JOIN_GRACE_S)

    assert failures == [], failures
    assert reported, "the child reported nothing -- it hung or died on import"

    kind, elapsed, exc_name = reported[0]
    assert kind == "first", f"submit returned a value with no server: {reported[0]}"
    assert exc_name == "InferenceTimeout", (
        f"submit raised {exc_name}, not InferenceTimeout -- the spec names the "
        f"exception because the worker's exit path branches on it (D2)"
    )
    assert _REQUEST_TIMEOUT_S * 0.5 <= elapsed <= _REQUEST_TIMEOUT_S * 2.0, (
        f"timed out after {elapsed:.3f}s, want ~{_REQUEST_TIMEOUT_S}s "
        f"(request_timeout_s is the liveness gate's only handle on a wedged "
        f"worker -- design D10)"
    )

    assert len(reported) == 2, (
        "the client made no second submission report -- expected it to refuse "
        "the second submit, not to disappear"
    )
    kind, elapsed, exc_name = reported[1]
    assert kind == "second", (
        f"a timed-out client answered a second submit ({reported[1]}): D2 makes "
        f"the timeout fatal precisely so this cannot read a stale response"
    )
    assert elapsed <= _REQUEST_TIMEOUT_S * 0.5, (
        f"the second submit blocked for {elapsed:.3f}s before failing: a "
        f"poisoned client must refuse immediately, not wait out another window"
    )
