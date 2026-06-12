"""Zero-dependency test runner (use this when pytest is not installed).

    python3 tests/run_tests.py

Discovers ``test_*`` functions in the ``tests`` package and runs them. The same
files are also valid pytest tests.
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    test_files = sorted(Path(__file__).parent.glob("test_*.py"))
    passed = 0
    failed = 0
    for path in test_files:
        module = importlib.import_module(f"tests.{path.stem}")
        for name in dir(module):
            if not name.startswith("test_"):
                continue
            func = getattr(module, name)
            if not callable(func):
                continue
            try:
                func()
                passed += 1
                print(f"PASS {path.stem}.{name}")
            except Exception:  # noqa: BLE001 - report all failures
                failed += 1
                print(f"FAIL {path.stem}.{name}")
                traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
