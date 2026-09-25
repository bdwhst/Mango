"""Launch the Mango GUI (docs/DESIGN.md section 6.7) without installing the package.
Double-click `Mango.cmd` (Windows) or `Mango.command` (macOS) in the repository root, or:

    python scripts/gui.py --run runs/9x9-r0                 # play the run's best model
    python scripts/gui.py --run runs/9x9-r0 --mode analysis # play both colours, ask the search
    python scripts/gui.py --model runs/9x9-r0/models/0015-4da2388b --sims 400 --colour W
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from mango.gui import run  # noqa: E402

if __name__ == "__main__":
    sys.exit(run())
