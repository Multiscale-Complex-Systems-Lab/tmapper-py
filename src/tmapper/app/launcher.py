"""Console-script entry point: ``tmapper-app``.

A Streamlit app is a *script Streamlit runs*, not a program you import and
call, so this hands the app file to Streamlit's own CLI rather than
executing it directly -- running it as a plain script would just print
Streamlit's "missing ScriptRunContext" warnings and do nothing useful.

Any extra arguments are passed straight through, so the usual Streamlit
options still work::

    tmapper-app --server.port 8600 --server.address 0.0.0.0
"""

import sys
from pathlib import Path

APP_FILE = Path(__file__).resolve().parent / "streamlit_app.py"


def main(argv=None):
    try:
        from streamlit.web import cli as stcli
    except ImportError:  # pragma: no cover - depends on install extras
        raise SystemExit(
            "The interactive app needs Streamlit, which is not installed.\n"
            "Install it with:  pip install 'tmapper[app]'"
        )

    args = sys.argv[1:] if argv is None else list(argv)
    sys.argv = ["streamlit", "run", str(APP_FILE), *args]
    return stcli.main()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
