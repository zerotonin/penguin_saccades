"""Open the SLEAP labelling GUI: ``python sleap/gui.py <project.slp>``.

The ``sleap-label`` console script aborts on the pkg_resources conflict in
the 1.4.1 conda package and ``sleap.gui.app`` has no main guard.
"""

from __future__ import annotations

import sys

from sleap.gui.app import main

if __name__ == "__main__":
    main(sys.argv[1:])
