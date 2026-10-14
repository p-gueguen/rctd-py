"""Every RCTDConfig field must be read somewhere in the package.

A field nothing reads is a setting users change with no effect and no warning. Three such
fields exist today; they are listed here so the list cannot grow silently, and so wiring one
up forces this list to be updated too.
"""

import re
from pathlib import Path

from rctd import RCTDConfig

SRC = Path(__file__).resolve().parents[1] / "src" / "rctd"

# Not read yet: the solvers hard-code 50 iterations (25 for scoring), step 0.3 and min_change 0.001.
KNOWN_UNREAD = {"max_iter", "step_size", "MIN_CHANGE_REG"}


def test_every_config_field_is_read():
    code = "\n".join(p.read_text() for p in SRC.rglob("*.py") if p.name != "_types.py")
    unread = {f for f in RCTDConfig._fields if not re.search(rf"\bconfig\.{f}\b", code)}
    assert unread == KNOWN_UNREAD, (
        f"newly unread: {sorted(unread - KNOWN_UNREAD)}; "
        f"now read, drop from KNOWN_UNREAD: {sorted(KNOWN_UNREAD - unread)}"
    )
