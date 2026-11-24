"""``python -m komugi.eval.v1runner`` -- the process the harness spawns.

Thin by design: everything testable lives in
:func:`komugi.eval.v1runner.runner.main`, which takes its three streams as
parameters so the unit suite can drive a whole handshake in-process. This file
is the only place in the package that touches ``sys``.

Nothing here reconfigures buffering, and that is a decision rather than an
omission. The harness reads this process line by line off a pipe
(``v1bridge._Channel._pump_stdout``) and Python block-buffers a pipe by
default, so an unflushed response would sit in an 8 KB buffer until it filled
and the first ``pick`` would be observed as §D6 M2 -- a hang the harness caused
itself. The fix is placed at every write instead: ``runner.serve``,
``runner.write_refusal`` and the ``--check-imports`` branch each flush the
stream they wrote to, which is a property the unit suite can check on a
``StringIO`` and a ``reconfigure`` call here would not be.
"""

from __future__ import annotations

import sys

from komugi.eval.v1runner.runner import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr))
