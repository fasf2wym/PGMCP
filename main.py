"""Entry point for `python main.py` (documented run command).

Delegates to the real server entry point in pg_mcp.__main__.
"""

from pg_mcp.__main__ import main

if __name__ == "__main__":
    main()
