"""The interactive Streamlit app, and its launcher.

Lives inside the package so that ``pip install "tmapper[app]"`` ships it --
otherwise the app would only be runnable from a clone of the repository,
which is exactly the friction it exists to remove.
"""

from .launcher import main

__all__ = ["main"]
