#!/usr/bin/env python3
"""Legacy entry point — forwards to `python -m eloquent`."""

import sys
from eloquent.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
